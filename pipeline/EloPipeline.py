import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import BayesianRidge

class EloPipeline(object):
    def __init__(self, data_dir = '../data',
                 submission_dir = '../submission'):
        self.submission_dir = submission_dir
        self.data_dir = data_dir

    def get_train_target(self):
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train_agg_id1.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test_agg_id1.csv'))
        target = train_df['target']
        target.to_csv(os.path.join(self.data_dir, 'target.csv'))
        return target, test_df

    def stack_model(self, prediction_list_name,
                    method = "BayesianRidge",
                    split_method = "kFold",
                    n_splits = 5,
                    random_state = 4520):
        target, test_df= self.get_train_target()

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

if __name__ == "__main__":
    pipeline = EloPipeline()
    predict_list = ['lgb_id1.csv','cat_id4.csv','lgb_id2.csv']
    pipeline.stack_model(predict_list)
