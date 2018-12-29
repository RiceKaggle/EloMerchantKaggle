from model.MLModel import MLModel
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np
import pandas as pd
from preprocess.Dataset import Dataset
from model.bayesian_optimization import bayesian_optimisation
import os, json

class LGBModel(MLModel):
    def __init__(self, non_numeric_param=None,
                 objective='regression',debug = False,
                 metric='rmse', bayesian_optimisation=False,
                 bayesian_iteration = 50,verbose_eval=True,
                 split_method='KFold', n_splits=5,
                 random_state=2018, num_round=10000,
                 shuffle=True, data_dir="../data",
                 submission_dir='../submission',
                 best_param_dir = '../best_param',
                 train_file_name='df_train_agg1.csv',
                 test_file_name='df_test_agg1.csv'):
        '''

        :param non_numeric_param: non-numeric parameter, do not optimize these parameter
        :param objective: the objective of the model
        :param debug: if set debug equals to true, use debug model (load part of training and test data)
        :param metric:
        :param bayesian_optimisation: if true, use bayesian optimization to search best parameter, otherwise, use grid search
        :param baysian_iteration: num of iterations bayesian optimization run
        :param verbose_eval
        :param split_method: 'KFold' and 'StratifiedKFold' are available
        :param n_splits: number of fold you want to have in you train/val set
        :param random_state:
        :param num_round:
        :param shuffle:
        :param data_dir: this is the training and testing data directory
        :param submission_dir: the directory stores the final result
        :param best_param_dir: load the best param form best_param_dir
        :param train_file_name: processed training file in data_dir
        :param test_file_name: test file in data_dir

        '''
        super(LGBModel, self).__init__()
        self.submission_dir = submission_dir
        self.objective = objective
        self.debug = debug
        self.metric = metric
        self.num_round = num_round
        self.split_method = split_method
        self.bayesian_iteration = bayesian_iteration
        self.verbose_eval = verbose_eval
        self.bayesian_optimisation = bayesian_optimisation
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.so_far_best_params = None
        self.best_param_dir = best_param_dir
        # this can add additional features to the param list
        self.non_numeric_param = {'objective': self.objective, 'metric': self.metric}

        if not verbose_eval:
            self.non_numeric_param['verbosity'] = -1

        self.train_file_name = train_file_name
        self.test_file_name = test_file_name

    def predict_with_best_param(self, file_name = 'submission_lgb.csv', best_param_name = 'lgb_param.json'):
        self.read_data()
        param_path = os.path.join(self.best_param_dir,best_param_name)
        print('lgb::: load best param so far form {} ...'.format(param_path))
        with open(param_path) as readfile:
            self.so_far_best_params = json.loads(readfile)
        print('lgb::: load successfully, begin to train and prediction ...')
        cv_error, prediction = self._train(params=self.so_far_best_params, predict=True)

        print('lgb::: best cv_error is {} ...'.format(cv_error))
        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)
        print('lgb::: save the prediction result in {} ...'.format(os.path.join(self.submission_dir, file_name)))

        return prediction

    def predict_with_param(self, param, file_name = 'prediction_lgb_id_default.csv'):
        self.read_data()
        cv_error, prediction = self._train(params=param, predict= True)
        print('cv_error with provided parameters is {} ...'.format(cv_error))
        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)
        print('save the prediction result in {} ...'.format(os.path.join(self.submission_dir, file_name)))
        return prediction


    def predict(self, file_name='submission_lgb.csv', best_param_name = 'lgb_param.json'):
        # use the best parameter to predict and save as file_name
        cv_error, prediction = self._train(params=self.so_far_best_params, predict=True)

        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)

        # after perdicting with the best parameter, save the best parameter
        with open(os.path.join(self.best_param_dir,best_param_name), 'w') as outputfile:
            json.dump(self.so_far_best_params, outputfile)

        return prediction

    def _train(self, params=None, predict=False):
        oof_lgb = np.zeros(len(self.train_X))
        prediction = np.zeros(len(self.test))
        feature_importance_df = pd.DataFrame()
        if self.split_method == 'KFold':
            kfold = KFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X))

        elif self.split_method == 'StratifiedKFold':
            kfold = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X, self.train_Y.values))

        for fold_, (train_index, val_index) in iterator:

            train_x, val_x = self.train_X.loc[train_index, self.train_features], \
                             self.train_X.loc[val_index, self.train_features]
            train_y, val_y = self.train_Y[train_index], self.train_Y[val_index]

            print('lgb fold_{}'.format(fold_ + 1))
            train_set = lgb.Dataset(train_x, label=train_y, categorical_feature=self.cate_features)
            val_set = lgb.Dataset(val_x, label=val_y, categorical_feature=self.cate_features)

            model = lgb.train(params, train_set, self.num_round, valid_sets=[train_set, val_set], verbose_eval=100,
                              early_stopping_rounds=200)
            oof_lgb[val_index] = model.predict(val_x, num_iteration=model.best_iteration)
            # only when predict is true calculate the featrue importance and predict on test set
            if predict:
                fold_importance_feature = pd.DataFrame()
                fold_importance_feature['Feature'] = self.train_features
                fold_importance_feature['importance'] = model.feature_importance()
                fold_importance_feature['fold'] = fold_ + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_feature], axis=0)
                prediction += model.predict(self.test.loc[:,self.train_features], num_iteration=model.best_iteration) / kfold.n_splits
        print('CV score: {:<8.5f}'.format(mean_squared_error(oof_lgb, self.train_Y) ** 0.5))
        return mean_squared_error(oof_lgb, self.train_Y) ** 0.5, prediction

    def sample_loss(self, params):
        # change the numeric parameter list to parameter dict
        param_dict = {}
        for index, key in enumerate(['num_leaves', 'max_depth', 'min_child_weight', 'learning_rate',
                                     'bagging_fraction', 'feature_fraction', 'bagging_freq']):
            if key in ['num_leaves', 'max_depth', 'min_child_weight', 'bagging_freq']:
                param_dict[key] = int(params[index])
            else:
                param_dict[key] = float(params[index])
        # set some non-numeric parameters too
        for key in self.non_numeric_param:
            param_dict[key] = self.non_numeric_param[key]

        print(param_dict)
        cv_error,_ = self._train(param_dict)
        return cv_error, param_dict

    def read_data(self):
        data_set = Dataset(self.train_file_name, self.test_file_name, base_dir=self.data_dir)

        self.train_X, self.train_Y, self.test, self.train_features, self.cate_features = data_set.preprocess(
            reload=True)

    # overload the train method
    def train(self, params_list=None):
        self.read_data()

        # debug mode only load part of data, to test whether the whole pipeline works
        if self.debug:
            self.train_X = self.train_X[:1000]
            self.test = self.test[:1000]
            self.train_Y = self.train_Y[:1000]
            self.bayesian_iteration = 1

        if not self.bayesian_optimisation:
            # frid search the best parameter
            params_list = {
                "objective": ["regression"],
                "metric": ["rmse"],
                'boosting': ["gbdt"],
                "num_leaves": [10, 30, 50],
                "min_child_weight": [40, 50, 60],
                "max_depth": [5, 7, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.06],
                "bagging_fraction": [0.6, 0.7, 0.8],
                "feature_fraction": [0.6, 0.7, 0.8],
                "bagging_frequency": [4, 5, 6],
            }

            super(LGBModel, self).train(params_list)
        else:
            func_param = self.prepare_baysian_optimization()
            xp, yp, param_candidate = bayesian_optimisation(**func_param)
            self.so_far_best_rmse = yp[0]
            for i in range(1, len(param_candidate)):
                if yp[i] < self.so_far_best_rmse:
                    self.so_far_best_rmse = yp[i]
                    self.so_far_best_params = param_candidate[i]

    def prepare_baysian_optimization(self):
        n_iters = self.bayesian_iteration
        # each bounds corresponding to ['num_leaves','max_depth','min_child_weight','learning_rate','bagging_fraction','feature_fraction','bagging_freq']
        bounds = [[10, 25], [5, 10], [10, 30], [0.03, 0.08], [0.5, 0.8], [0.5, 0.8], [3, 10]]

        return {'n_iters': n_iters, 'bounds': np.array(bounds), 'sample_loss': self.sample_loss}


if __name__ == '__main__':
    model = LGBModel(bayesian_optimisation=True,debug=False,verbose_eval=False)
    model.train()
    model.predict()
