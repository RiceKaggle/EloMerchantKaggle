import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import BayesianRidge
from model.LGBModel import LGBModel
from preprocess.Dataset import Dataset

class EloPipeline(object):
    def __init__(self, data_dir = '../data',
                 submission_dir = '../submission',
                 train_file = 'train_agg_id1.csv',
                 test_file = 'test_agg_id1.csv'):
        self.submission_dir = submission_dir
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file

    def get_train_target(self):
        train_df = pd.read_csv(os.path.join(self.data_dir, self.train_file))
        test_df = pd.read_csv(os.path.join(self.data_dir, self.test_file))
        target = train_df['target']
        target.to_csv(os.path.join(self.data_dir, 'target.csv'))
        return target, test_df, train_df

    def stack_model(self, prediction_list_name,
                    method = "BayesianRidge",
                    split_method = "kFold",
                    n_splits = 5,
                    random_state = 4520):
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
                kfold = KFold(n_splits=n_splits, random_state=random_state)
                iterator = enumerate(kfold.split(train_stack))

            elif split_method == 'StratifiedKFold':
                kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state)
                iterator = enumerate(kfold.split(train_stack,target.values))

            oof_stack = np.zeros(train_stack.shape[0])
            predictions_stack = np.zeros(test_stack.shape[0])

            for fold_, (trn_idx, val_idx) in enumerate(kfold.split(train_stack, target)):
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
            oof_file_name = '_'.join(prediction_list_name).strip()
            oof_file_name = 'oof_merge_'+oof_file_name
            pred_file_name = '_'.join(prediction_list_name).strip()
            pred_file_name = 'merge_'+pred_file_name

            stack_result = pd.DataFrame({'card_id':test_df['card_id']})
            stack_result['target'] = predictions_stack
            stack_result.to_csv(os.path.join(self.submission_dir,pred_file_name), index=False)

            oof_stack = pd.DataFrame({'target': oof_stack})
            oof_stack.to_csv(os.path.join(self.submission_dir + '/oof', 'oof_' + oof_file_name), index=False)
            print('stacked oof and prediction file save successfully ...')


    # this method use lgb to do binary classification
    def binary_classification(self, methon = 'lgb', metrics = 'auc',
                              params = None, n_splits =5, split_method = 'KFold'):
        lgb_model = LGBModel(objective = 'binary',metric = metrics ,debug=False,verbose_eval=False)

        if not hasattr(self, 'train_X') or not hasattr(self, 'train_Y'):

            self.set_train_outlier()

        target = self.train_X['outlier']
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

        predictions = lgb_model.predict_with_param(param, read_data=False, file_name='lgb_outlier_classifier_id12.csv')
        outlier_classify = pd.DataFrame({'card_id':self.test['card_id']})
        outlier_classify['target'] = predictions
        outlier_classify.sort_values(by=['target'], ascending=False).reset_index(drop=True, inplace = True)
        return outlier_classify


    def separate_prediction(self, outlier_classfier, model_without_outlier, best_model, submission_name):
        outlier_id = pd.DataFrame(outlier_classfier.head(1310)['card_id'])
        most_likely_liers = best_model.merge(outlier_id, how='right')
        for card_id in most_likely_liers['card_id']:
            model_without_outlier.loc[model_without_outlier['card_id'] == \
                                      card_id,'target'] = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values

        model_without_outlier.to_csv(os.path.join(self.submission_dir, submission_name))
        return model_without_outlier


    def set_train_outlier(self):
        dataset = Dataset(train_path=self.train_file, test_path=self.test_file)

        self.train_X, self.train_Y, self.test, self.features, self.cate_features = dataset.preprocess(reload=True)
        self.train_X['target'] = self.train_Y
        dataset.set_outlier_col(self.train_X)

        train_df = self.train_X[self.train_X['outlier'] == 0]
        target = train_df['target']
        return train_df, target


    def train_without_outlier(self):
        lgb_model = LGBModel(random_state=2333,  debug=False, verbose_eval=False, split_method='StratifiedKFold')
        # set_outlier will set the training and testing data
        train_df, target = self.set_train_outlier()
        # set training and testing data for lgb model
        lgb_model.set_train_test(train_df, target ,self.test ,self.features,self.cate_features,'outlier')

        param = {'num_leaves': 31,
                 'min_data_in_leaf': 32,
                 'objective': 'regression',
                 'max_depth': -1,
                 'learning_rate': 0.005,
                 "min_child_samples": 20,
                 "boosting": "gbdt",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9,
                 "bagging_seed": 11,
                 "metric": 'rmse',
                 "lambda_l1": 0.1,
                 "nthread": 4,
                 "verbosity": -1}

        prediction = lgb_model.predict_with_param(param, read_data=False, file_name='lgb_without_outlier_id11.csv')
        model_without_outliers = pd.DataFrame({"card_id": self.test["card_id"].values})
        model_without_outliers["target"] = prediction
        return model_without_outliers


if __name__ == "__main__":
    # 1. stack models together
    # pipeline = EloPipeline()
    # predict_list = ['lgb_id1.csv','cat_id4.csv','lgb_id2.csv']
    # pipeline.stack_model(predict_list)
    pass

    # 2.predict without outlier
    pipeline = EloPipeline(train_file='train_clean.csv',test_file='test_clean.csv')
    model_without_outliers = pd.read_csv('../submission/lgb_id14.csv')
    outlier_likehood = pipeline.binary_classification()
    best_model = pd.read_csv('../submission/3.695.csv')
    pipeline = pipeline.separate_prediction(outlier_likehood, model_without_outliers,best_model,'without_outlier_id13.csv')


