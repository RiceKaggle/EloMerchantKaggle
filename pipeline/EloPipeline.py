import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import BayesianRidge
from model.LGBModel import LGBModel
from preprocess.Dataset import Dataset
import lightgbm as lgb
from model.CatBoostModel import CatBoostModel

START_ID = 11

class EloPipeline(object):
    def __init__(self, data_dir = '../data',
                 submission_dir = '../submission',
                 train_file = 'train_agg_id1.csv',
                 test_file = 'test_agg_id1.csv',
                 combine_mode = 'whole', shuffle = True,
                 random_state = 15):
        '''

        :param data_dir:
        :param submission_dir:
        :param train_file:
        :param test_file:
        :param combine_mode: when combine_mode is "whole", use the whole dataset to train, when the combine_mode is "outliers",
        only combine the result on outlier data, when the combine_mode is "without_outliers" the combine result on training data
        without the outlier
        '''
        self.submission_dir = submission_dir
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.combine_mode = combine_mode
        self.shuffle = shuffle
        self.random_state = random_state

    def get_train_target(self):
        train_df = pd.read_csv(os.path.join(self.data_dir, self.train_file))
        test_df = pd.read_csv(os.path.join(self.data_dir, self.test_file))
        if self.combine_mode == 'outliers':
            train_df = train_df[train_df['outliers'] == 1]
            target = train_df['target']
            del train_df['target']
        elif self.combine_mode == 'without_outliers':
            train_df = train_df[train_df['outliers'] == 0]
            target = train_df['target']
            del train_df['target']

        return target, test_df, train_df


    def stack_model(self, prediction_list_name,
                    method = "BayesianRidge",
                    split_method = "kFold",
                    n_splits = 5):

        target, test_df, train_df= self.get_train_target()

        if len(prediction_list_name) == 0:
            print("no prediction result ...")
            return
        else:
            oof_list = []
            prediction_list = []

            for name in prediction_list_name:

                pred_path = os.path.join(self.submission_dir, name)
                oof_path = os.path.join(self.submission_dir+'/oof', 'oof_'+name)
                if not os.path.isfile(pred_path):
                    print("{} is not a prediction result path".format(pred_path))
                elif not os.path.isfile(oof_path):
                    print("{} is not a oof result path".format(oof_path))
                else:
                    oof = pd.read_csv(oof_path)
                    prediction = pd.read_csv(pred_path)
                    prediction_list.append(prediction['target'].values)
                    oof_list.append(oof['target'].values)

            train_stack = np.vstack(oof_list).transpose()
            test_stack = np.vstack(prediction_list).transpose()

            if split_method == 'kFold':
                kfold = KFold(n_splits=n_splits, random_state=self.random_state, shuffle=self.shuffle)
                iterator = enumerate(kfold.split(train_stack))

            elif split_method == 'StratifiedKFold':
                kfold = StratifiedKFold(n_splits=n_splits, random_state=self.random_state, shuffle=self.shuffle)
                iterator = enumerate(kfold.split(train_stack,target.values))

            oof_stack = np.zeros(train_stack.shape[0])
            predictions_stack = np.zeros(test_stack.shape[0])

            for fold_, (trn_idx, val_idx) in iterator:
                print("fold n°{}".format(fold_))
                trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
                val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

                print("-" * 10 + "Stacking " + str(fold_) + "-" * 10)
                #     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
                #     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
                if method == 'BayesianRidge':
                    clf = BayesianRidge()
                    clf.fit(trn_data, trn_y)

                oof_stack[val_idx] = clf.predict(val_data)
                predictions_stack += clf.predict(test_stack) / 5

            print("cv score : ",np.sqrt(mean_squared_error(target.values, oof_stack)))
            print('save stacked oof file and prediction file ...')
            oof_file_name = '_'.join([name[:len(name)-4] for name in prediction_list_name]).strip()
            oof_file_name = 'oof_merge_'+oof_file_name
            pred_file_name = '_'.join([name[:len(name)-4] for name in prediction_list_name]).strip()
            pred_file_name = 'merge_'+pred_file_name

            stack_result = pd.DataFrame({'card_id':test_df['card_id']})
            stack_result['target'] = predictions_stack
            stack_result.to_csv(os.path.join(self.submission_dir,pred_file_name), index=False)

            oof_stack = pd.DataFrame({'target': oof_stack})
            oof_stack.to_csv(os.path.join(self.submission_dir + '/oof', 'oof_' + oof_file_name), index=False)
            print('stacked oof and prediction file save successfully ...')


    # this method use lgb to do binary classification
    def binary_classification(self,  metrics = 'binary_logloss', n_splits =5, split_method = 'KFold'):
        lgb_model = LGBModel(random_state=15,objective = 'binary',metric = metrics ,debug=False,verbose_eval=False)

        if not hasattr(self, 'train_X') or not hasattr(self, 'train_Y'):

            self.set_train_outlier()

        target = self.train_X['outliers']
        df_train = self.train_X.copy()
        del df_train['target']
        print(len(target))
        lgb_model.set_train_test(df_train, target, self.test, self.features, self.cate_features)

        param = {'num_leaves': 31,
                 'min_data_in_leaf': 30,
                 'objective': 'binary',
                 'max_depth': 6,
                 'learning_rate': 0.01,
                 "boosting": "rf",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9,
                 "bagging_seed": 11,
                 "metric": 'binary_logloss',
                 "lambda_l1": 0.1,
                 "verbosity": -1,
                 "random_state": 2333}

        predictions = lgb_model.predict_with_param(param, read_data=False, file_name='lgb_outlier_classifier_id'+(str(START_ID+1))+'.csv')
        outlier_classify = pd.DataFrame({'card_id':self.test['card_id']})
        outlier_classify['target'] = predictions
        return outlier_classify


    def separate_prediction(self, outlier_classfier, model_without_outlier, best_model, submission_name):
        outlier_id = pd.DataFrame(outlier_classfier.sort_values(by='target', ascending=False).head(25000)['card_id'])

        most_likely_liers = best_model.merge(outlier_id, how='right')
        print(most_likely_liers.head(50))
        for card_id in most_likely_liers['card_id']:
            model_without_outlier.loc[model_without_outlier['card_id'] == \
                                      card_id,'target'] = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values

        model_without_outlier.to_csv(os.path.join(self.submission_dir, submission_name), index=False)
        return model_without_outlier


    def set_train_outlier(self):
        dataset = Dataset(train_path=self.train_file, test_path=self.test_file)

        self.train_X, self.train_Y, self.test, self.features, self.cate_features = dataset.preprocess(reload=True)
        self.train_X['target'] = self.train_Y

        if 'outliers' not in self.train_X.columns:
            dataset.set_outlier_col(self.train_X)

        train_df = self.train_X[self.train_X['outliers'] == 0]

        target = train_df['target']
        del train_df['target']

        return train_df, target

    
    def train_without_outlier_cat(self):
        cat_model = CatBoostModel()
        param = {
            "iterations": 10000,
            "learning_rate": 0.005,
            "depth": 6,
            "eval_metric": 'RMSE',
            "bagging_temperature": 0.9,
            "od_type": 'Iter',
            "metric_period": 100,
            "od_wait": 50,
            "random_state": 2333
        }
        train_df, target = self.set_train_outlier()
        cat_model.set_train_test(train_df, target, self.test, self.features, self.cate_features, 'outliers')
        # if you use the set_train_test, you need to set read_data to false in case the read_data() method override the train and test data
        prediction = cat_model.predict_with_param(param=param,read_data=False,file_name='cat_without_outlier_id'+(str(START_ID))+'.csv')
        model_without_outliers = pd.DataFrame({"card_id": self.test["card_id"].values})
        model_without_outliers["target"] = prediction
        return model_without_outliers


    def train_without_outlier_lgb(self):
        lgb_model = LGBModel(contain_cate=False, random_state=2333,
                    debug=False, verbose_eval=False, split_method='StratifiedKFold')

        # set_outlier will set the training and testing data
        train_df, target = self.set_train_outlier()

        # set training and testing data for lgb model
        lgb_model.set_train_test(train_df, target ,self.test ,self.features,self.cate_features,'outliers')

        param = {'objective': 'regression',
                 'num_leaves': 31,
                 'min_data_in_leaf': 25,
                 'max_depth': 7,
                 'learning_rate': 0.01,
                 'lambda_l1': 0.13,
                 "boosting": "gbdt",
                 "feature_fraction": 0.85,
                 'bagging_freq': 8,
                 "bagging_fraction": 0.9,
                 "metric": 'rmse',
                 "verbosity": -1,
                 "random_state": 2333}

        prediction = lgb_model.predict_with_param(param, read_data=False, file_name='lgb_without_outlier_id'+(str(START_ID))+'.csv')
        model_without_outliers = pd.DataFrame({"card_id": self.test["card_id"].values})
        model_without_outliers["target"] = prediction
        return model_without_outliers


if __name__ == "__main__":

    # 2.predict without outlier
    pipeline = EloPipeline(train_file='train_clean.csv',test_file='test_clean.csv',combine_mode='whole')

    # train a cat boost model without outlier
    cat_model_without_outliers = pipeline.train_without_outlier_cat()

    #model_without_outliers = pipeline.train_without_outlier_lgb()
    lgb_model_without_outliers = pd.read_csv('../submission/lgb_without_outlier_id'+(str(START_ID))+'.csv')

    predict_list = ['lgb_without_outlier_id'+(str(START_ID))+'.csv', 'cat_without_outlier_id'+(str(START_ID))+'.csv']
    pipeline.stack_model(predict_list)


    #outlier_likehood = pipeline.binary_classification()
    #outlier_likehood = pd.read_csv('../submission/lgb_outlier_classifier_id'+(str(START_ID+1))+'.csv')

    #best_model = pd.read_csv('../submission/3.695.csv')
    #pipeline = pipeline.separate_prediction(outlier_likehood, model_without_outliers,best_model,'without_outlier_id'+(str(START_ID+2))+'.csv')


