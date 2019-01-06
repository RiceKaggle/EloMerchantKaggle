import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, BayesianRidge

train_df = pd.read_csv('../data/train_clean.csv')
test_df = pd.read_csv('../data/test_clean.csv')

train = train_df[train_df['outliers'] == 0]
target = train['target']
del train['target']
len(train)

features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'outliers','Unnamed: 0']]
cate_features = [c for c in features if 'feature_' in c]

print(features)
from catboost import CatBoostRegressor
import catboost as cb
from sklearn.model_selection import KFold
import numpy as np


FOLDs = KFold(n_splits=5, shuffle=True, random_state=15)
param = {
    "iterations": 10000,
    "learning_rate": 0.002,
    "depth": 5,
    "eval_metric": 'RMSE',
    "bagging_temperature": 0.8,
    "od_type": 'Iter',
    "metric_period": 50,
     "od_wait": 20,
     "random_state": 2333
}

oof_cb = np.zeros(len(train))
predictions_cb = np.zeros(len(test_df))


for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = cb.Pool(train.iloc[trn_idx][features], target.iloc[trn_idx])
    val_data = cb.Pool(train.iloc[val_idx][features], target.iloc[val_idx])

    print("CB " + str(fold_) + "-" * 50)
    num_round = 10000
    #cb_model = cb.CatBoostRegressor(max_depth=11, learning_rate=0.005, eval_metric = 'RMSE', iterations=num_round, early_stopping_rounds=50)
    cb_model = cb.CatBoostRegressor(**param)
    cb_model.fit(trn_data, verbose_eval = 400, eval_set = val_data)
    oof_cb[val_idx] = cb_model.predict(train.iloc[val_idx][features])
    predictions_cb += cb_model.predict(test_df[features]) / FOLDs.n_splits

np.sqrt(mean_squared_error(oof_cb, target))

oof_cat_df = pd.DataFrame({'target':oof_cb})
oof_cat_df.to_csv('../submission/oof/oof_cat_without_outlier_id11#2.csv', index=False)

oof_cat_prediction_df = pd.DataFrame({'card_id':test_df['card_id']})
oof_cat_prediction_df['target'] = predictions_cb
oof_cat_prediction_df.to_csv('../submission/cat_without_outlier_id11#2.csv', index = False)