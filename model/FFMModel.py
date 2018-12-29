import pandas as pd
from preprocess.Dataset import Dataset
import os
from model.MLModel import MLModel
from model.bayesian_optimization import bayesian_optimisation
import numpy as np
import xlearn as xl

'''
    This class is the implementation of Field-Aware Factorization Machines model, the pipeline follows these steps:
    1. load training and test dataset and select field to process
    2. process the training and dataset to libSVM format. The format of the training and test data is <label><feature1>:<value1><feature2>:<value2>
    3. use grid search of bayesian optimization to find the best parameter.
    4. predict the result
'''
class FFMModel(MLModel):
    def __init__(self,task='reg',debug = False,
                 metric='rmse', bayesian_optimisation=False,
                 bayesian_iteration = 30, n_splites = 5,
                 train_name='alltrainffm.txt',
                 test_name='alltestffm.txt',
                 data_dir='../data',
                 model_dir = '../pretrained_model',
                 model_name = 'model_output.out',
                 submission_dir='../submission',
                 ):
        super(FFMModel, self).__init__()
        self.debug = debug
        self.data_dir = data_dir
        self.prepared = False
        self.train_name = train_name
        self.test_name = test_name
        self.bayesian_optimisation = bayesian_optimisation
        self.metric = metric
        self.task = task
        self.debug = debug
        self.bayesian_optimisation = 30
        self.bayesian_iteration = bayesian_iteration
        self.n_splites = n_splites
        self.model_dir = model_dir
        self.model_name = model_name
        self.submission_dir = submission_dir
        self.non_numeric_param =  {'task': self.task, 'metric': self.metric, 'fold':self.n_splites}

    def train(self, params_list=None):
        if self.debug:
            params = {
                "task":"reg",
                "metric":"rmse",
                "lr":0.2,
                "lambda":0.005
            }
            self._train(params)
            return
        if not self.bayesian_optimisation:
            # frid search the best parameter
            params_list = {
                "task": ["reg"],
                "metric": ["rmse"],
                'lr': [0.05,0.1,0.15,0.2,0.25,0.3],
                'lambda': [0.002,0.003,0.004,0.005]
            }
            super(FFMModel, self).train(params_list)
        else:
            func_param = self.prepare_baysian_optimization()
            xp, yp, param_candidate = bayesian_optimisation(**func_param)
            self.so_far_best_rmse = yp[0]
            # iterate all the result from bayesian process and choose the best one
            for i in range(1, len(param_candidate)):
                if yp[i] < self.so_far_best_rmse:
                    self.so_far_best_rmse = yp[i]
                    self.so_far_best_params = param_candidate[i]

    def prepare_baysian_optimization(self):
        n_iters = self.bayesian_iteration
        # each bounds corresponding to ['lr','lambda']
        bounds = [[0.05,0.4], [0.001, 0.009]]

        return {'n_iters': n_iters, 'bounds': np.array(bounds), 'sample_loss': self.sample_loss}

    def _train(self, params):
        ffm_model = xl.create_ffm()
        ffm_model.setTrain(os.path.join(self.data_dir,self.train_name))

        print(params)
        ffm_model.cv(params)

        ffm_model.fit(params,os.path.join(self.model_dir,self.model_name))

        ffm_model.setTest(os.path.join(self.data_dir,self.test_name))
        ffm_model.predict(os.path.join(self.model_dir,self.model_name),
                          os.path.join(self.submission_dir, 'ffmoutput.txt'))

        # wait for the answer of how to get loss value in xlearn
        cv_error = 0
        return cv_error, params


    def sample_loss(self, params):
        # change the numeric parameter list to parameter dict
        param_dict = {}
        for index, key in enumerate(['lr','lambda']):
            param_dict[key] = params[index]
        # set some non-numeric parameters too
        for key in self.non_numeric_param:
            param_dict[key] = self.non_numeric_param[key]

        print(param_dict)
        cv_error,_ = self._train(param_dict)
        return cv_error, param_dict

    def predict(self):
        pass


if __name__ == "__main__":

    dataset = Dataset(train_path='df_train_agg1.csv', test_path='df_test_agg1.csv')
    dataset.format_transformer(train_file_name='alltrainffm_agg1.txt',
                               test_file_name='alltestffm_agg1.txt',
                               fields=['feature_1', 'feature_2', 'feature_3', 'elapsed_time', 'hist_month_lag_max', 'hist_category_1_sum',
                                       'hist_weekend_sum','hist_category_3_mean_mean','hist_category_1_sum','hist_category_1_mean',
                                       'hist_authorized_flag_sum','hist_authorized_flag_mean','hist_purchase_date_max'])

    model = FFMModel(debug=True, train_name = 'alltrainffm_agg1.txt', test_name='alltestffm_agg1.txt')
    model.train()


