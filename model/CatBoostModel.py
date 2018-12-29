from model.MLModel import MLModel
from preprocess.Dataset import Dataset
import numpy as np
from model.bayesian_optimization import bayesian_optimisation
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import os
import json

class CatBoostModel(MLModel):
    def __init__(self,non_numeric_param=None,
                 bayesian_iteration = 50,
                 bayesian_optimisation = False,
                 debug = False,
                 split_method='KFold',
                 n_splits=5,
                 data_dir="../data",
                 submission_dir='../submission',
                 best_param_dir = '../best_param',
                 train_file_name='train_agg_id1.csv',
                 test_file_name='test_agg_id1.csv',
                 random_state=1052):
        super(CatBoostModel, self).__init__()
        self.bayesian_optimisation = bayesian_optimisation
        self.debug = debug
        self.split_method=split_method
        self.n_splits = n_splits
        self.bayesian_iteration = bayesian_iteration
        self.data_dir =data_dir
        self.submission_dir=submission_dir
        self.best_param_dir = best_param_dir
        self.train_file_name=train_file_name
        self.test_file_name = test_file_name
        self.non_numeric_param = non_numeric_param
        self.random_state = random_state
        self.non_numeric_param = {"objective":"regression", "eval_metric":"RMSE"}


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
                "learning_rate" : ["0.02,0.03,0.04,0.05"],
                "depth":[4,5,6,7,8],
                "eval_metric":['RMSE'],
                "bagging_temperature":[0.8,0.9],
                "metric_period":[80,90,100],
                "od_wait":[50,60]
            }

            super(CatBoostModel, self).train(params_list)
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
        # each bounds corresponding to ['learning_rate','depth','bagging_temperature','metric_period','od_wait']
        bounds = [[0.02, 0.06], [4, 9], [0.8, 1.0], [80, 100], [50, 60]]

        return {'n_iters': n_iters, 'bounds': np.array(bounds), 'sample_loss': self.sample_loss}

    def _train(self, params, predict=False, save_oof = False):
        oof_cat = np.zeros(len(self.train_X))
        prediction = np.zeros(len(self.test))
        feature_importance_df = pd.DataFrame()
        if self.split_method == 'KFold':
            kfold = KFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X))

        elif self.split_method == 'StratifiedKFold':
            kfold = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X, self.train_Y.values))

        for fold_, (train_index, val_index) in iterator:
            print('cat fold_{}'.format(fold_ + 1))

            model_cat = CatBoostRegressor(**params)

            model_cat.fit(self.train_X.loc[train_index, self.train_features], self.train_Y[train_index],
                          eval_set=(self.train_X.loc[val_index, self.train_features], self.train_Y[val_index]))
            oof_cat[val_index] = model_cat.predict(self.train_X.loc[val_index, self.train_features])
            # only when predict is true calculate the featrue importance and predict on test set
            if predict:
                prediction += model_cat.predict(self.test.loc[:,self.train_features]) / kfold.n_splits
        print('CV score: {:<8.5f}'.format(mean_squared_error(oof_cat, self.train_Y) ** 0.5))


        return mean_squared_error(oof_cat, self.train_Y) ** 0.5, oof_cat, prediction



    def predict_with_param(self, param, file_name = 'prediction_cat_id_default.csv'):
        self.read_data()

        cv_error, oof_cat , prediction = self._train(params=param, predict=True)

        # for debug purpose
        # cv_error = pd.DataFrame({'id':[1,2,3]})
        # oof_cat = np.zeros(7)
        # prediction =  pd.DataFrame({'id':[1,2,3]})

        oof_cat = pd.DataFrame({'target':oof_cat})
        oof_cat.to_csv(os.path.join(self.submission_dir+'/oof','oof_'+file_name), index=False)

        print('cv_error with provided parameters is {} ...'.format(cv_error))
        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)
        print('save the prediction result in {} ...'.format(os.path.join(self.submission_dir, file_name)))
        return prediction


    def predict_with_best_param(self, file_name = 'submission_cat.csv', best_param_name = 'cat_param.json'):
        self.read_data()
        param_path = os.path.join(self.best_param_dir,best_param_name)
        print('catboost::: load best param so far form {} ...'.format(param_path))
        with open(param_path) as readfile:
            self.so_far_best_params = json.loads(readfile)
        print('catboost::: load successfully, begin to train and prediction ...')
        cv_error, oof_cat, prediction = self._train(params=self.so_far_best_params, predict=True)

        print('catboost::: best cv_error is {} ...'.format(cv_error))
        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)
        print('catboost::: save the prediction result in {} ...'.format(os.path.join(self.submission_dir, file_name)))

        return prediction


    def predict(self, file_name='submission_cat.csv', best_param_name = 'cat_param.json'):
        # use the best parameter to predict and save as file_name
        cv_error, oof_cat, prediction = self._train(params=self.so_far_best_params, predict=True)

        result = pd.DataFrame({'card_id': self.test['card_id']})
        result['target'] = prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)

        # after perdicting with the best parameter, save the best parameter
        with open(os.path.join(self.best_param_dir,best_param_name), 'w') as outputfile:
            json.dump(self.so_far_best_params, outputfile)

        return prediction

    def read_data(self):
        data_set = Dataset(self.train_file_name, self.test_file_name, base_dir=self.data_dir)

        self.train_X, self.train_Y, self.test, self.train_features, self.cate_features = data_set.preprocess(
            reload=True)

    def sample_loss(self, params):
        # change the numeric parameter list to parameter dict
        param_dict = {}
        for index, key in enumerate(['learning_rate','depth','bagging_temperature','metric_period','od_wait']):
            param_dict[key] = params[index]
        # set some non-numeric parameters too
        for key in self.non_numeric_param:
            param_dict[key] = self.non_numeric_param[key]

        print(param_dict)
        cv_error,_,_ = self._train(param_dict)
        return cv_error, param_dict

if __name__ == '__main__':
    model = CatBoostModel()
    param = {
        "iterations" : 10000,
        "learning_rate": 0.02,
        "depth" : 6,
        "eval_metric" : 'RMSE',
        "bagging_temperature" : 0.9,
        "od_type" : 'Iter',
        "metric_period" : 100,
        "od_wait" : 50
    }
    model.predict_with_param(param=param, file_name="cat_id4.csv")