from abc import abstractmethod
from util.util import map_list_combination
import json
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, log_loss
import numpy as np
from preprocess.Dataset import Dataset
import pandas as pd
from tqdm import tqdm
import os
from model.bayesian_optimization import bayesian_optimisation


class MLModel(object):

    @abstractmethod
    def __init__(self):
        self.so_far_best_rmse = 1000
        self.so_far_best_params = None

    @abstractmethod
    def train(self, params_list = None):
        list_params = map_list_combination(params_list)

        for params in tqdm(list_params):
            print("Current Params:{}".format(json.dumps(params)))
            cv = self._train(params)
            if cv < self.so_far_best_rmse:
                self.so_far_best_rmse = cv
                self.so_far_best_params = params

    @abstractmethod
    def _train(self, params):
        pass

    @abstractmethod
    def predict(self):
        pass

class LGBModel(MLModel):
    def __init__(self,  non_numeric_param = None ,
                 objective = 'regression',
                 metric = 'rmse',bayesian_optimisation = False,
                 split_method = 'KFold',n_splits = 5,
                 random_state=2018, num_round = 10000,
                 shuffle=True,data_dir="../data",
                 submission_dir = '../submission',
                 train_file_name = 'df_train_agg1.csv',
                 test_file_name = 'df_test_agg1.csv'):
        super(LGBModel,self).__init__()
        self.submission_dir = submission_dir
        self.objective = objective
        self.metric = metric
        self.num_round = num_round
        self.split_method = split_method
        self.bayesian_optimisation = bayesian_optimisation
        self.n_splits =n_splits
        self.random_state =random_state
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.so_far_best_params=None
        # this can add additional features to the param list
        self.non_numeric_param = {'objective': self.objective,  'metric': self.metric}
        data_set = Dataset(train_file_name,test_file_name)
        self.train_X, self.train_Y , self.test, self.train_features, self.cate_features= data_set.preprocess(reload=True)

    def predict(self, file_name = 'submission.csv'):
        # use the best parameter to predict and save as file_name
        self._train(params=self.so_far_best_params, predict=True )

        result = pd.DataFrame({'card_id':self.test['card_id']})
        result['target'] = self.prediction
        result.to_csv(os.path.join(self.submission_dir, file_name), index=False)

        return self.prediction

    def _train(self, params=None, predict = False):
        oof_lgb = np.zeros(len(self.train_X))
        self.prediction = np.zeros(len(self.test))
        feature_importance_df = pd.DataFrame()
        if self.split_method == 'KFold':
            kfold = KFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X))

        elif self.split_method == 'StratifiedKFold':
            kfold = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
            iterator = enumerate(kfold.split(self.train_X, self.train_Y.values))

        for fold_, (train_index, val_index) in iterator:

            train_x, val_x = self.train_X.loc[train_index,self.train_features], \
                             self.train_X.loc[val_index,self.train_features]
            train_y, val_y = self.train_Y[train_index], self.train_Y[val_index]

            print('lgb fold_{}'.format(fold_+1))
            train_set = lgb.Dataset(train_x, label=train_y, categorical_feature=self.cate_features)
            val_set = lgb.Dataset(val_x, label=val_y, categorical_feature= self.cate_features)


            model = lgb.train(params, train_set, self.num_round, valid_sets=[train_set, val_set], verbose_eval=100, early_stopping_rounds=200)
            oof_lgb[val_index] = model.predict(val_x, num_iteration=model.best_iteration)
            # only when predict is true calculate the featrue importance and predict on test set
            if predict:
                fold_importance_feature = pd.DataFrame()
                fold_importance_feature['Feature'] = self.train_features
                fold_importance_feature['importance'] = model.feature_importance()
                fold_importance_feature['fold'] = fold_ + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_feature], axis=0)
                self.prediction += model.predict(self.test,num_iteration=model.best_iteration) / KFold.n_splits
        print('CV score: {:<8.5f}'.format(mean_squared_error(oof_lgb, self.train_Y)**0.5))
        return mean_squared_error(oof_lgb, self.train_Y)**0.5

    def sample_loss(self, params):
        # change the numeric parameter list to parameter dict
        param_dict = {}
        for index, key in enumerate(['num_leaves','max_depth','min_child_weight','learning_rate',
                           'bagging_fraction','feature_fraction','bagging_freq']):
            if key in ['num_leaves','max_depth','min_child_weight','bagging_freq']:
                param_dict[key] = int(params[index])
        # set some non-numeric parameters too
        for key in self.non_numeric_param:
            param_dict[key] = self.non_numeric_param[key]

        print(param_dict)
        return self._train(param_dict), param_dict

    # overload the train method
    def train(self, params_list = None):
        if not self.bayesian_optimisation:
            # frid search the best parameter
            params_list = {
                "objective": ["regression"],
                "metric": ["rmse"],
                'boosting': ["gbdt"],
                "num_leaves": [10,30,50],
                "min_child_weight": [40,50,60],
                "max_depth":[5,7,10],
                "learning_rate": [0.01,0.03, 0.05, 0.06],
                "bagging_fraction": [0.6,0.7,0.8],
                "feature_fraction": [0.6,0.7,0.8],
                "bagging_frequency": [4,5,6],
            }


            super(LGBModel, self).train(params_list)
        else:
            func_param = self.prepare_baysian_optimization()
            xp, yp = bayesian_optimisation(**func_param)
            self.so_far_best_rmse = yp[0]
            for i in range(1,len(xp)):
                if yp[i] < self.so_far_best_rmse:
                    self.so_far_best_rmse = yp[i]
                    self.so_far_best_params = xp[i]


    def prepare_baysian_optimization(self):
        n_iters = 30
        sample_loss = None
        # each bounds corresponding to ['num_leaves','max_depth','min_child_weight','learning_rate','bagging_fraction','feature_fraction','bagging_freq']
        bounds = [[10,20],[5,10],[10,30],[0.03,0.08],[0,0.7],[0,0.7],[1,10]]

        return {'n_iters' : n_iters,'bounds':np.array(bounds), 'sample_loss':self.sample_loss}


if __name__ == '__main__':
    model = LGBModel(bayesian_optimisation=True)
    model.train()
    model.predict()














