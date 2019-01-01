import gc
gc.enable()
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class FeatureSelection(object):
    def __init__(self, feature_score_name):
        self.feature_score_name = feature_score_name
        pass

    def load_data(self):
        pass


    def get_feature_score(self):
        pass




