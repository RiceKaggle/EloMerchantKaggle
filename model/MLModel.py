from abc import abstractmethod
from util.util import map_list_combination
import json
import pandas as pd
from tqdm import tqdm
import os


# add this line to avoid warning of OpenKMP if your system have multiple OpenKMP lib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MLModel(object):
    @abstractmethod
    def __init__(self):
        self.so_far_best_rmse = 1000
        self.so_far_best_params = None

    @abstractmethod
    def train(self, params_list=None):
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
