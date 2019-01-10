import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

train_df = pd.read_csv('../data/df_train_agg1.csv')
train_df = train_df[:int(len(train_df)/2)]
train = train_df
target = train_df['target']
del train['target']

features = [c for c in train.columns if c not in ['card_id', 'first_active_month', 'outliers','Unnamed: 0']]
print(len(features))
cate_features = [c for c in features if 'feature_' in c]

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(train[features])

new_train = pd.DataFrame(data = data_with_imputed_values[:,:], columns=train[features].columns.values)

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()

target = lab_enc.fit_transform(target)


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc = RandomForestClassifier( n_jobs=-1, class_weight='balanced')
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
x=new_train[features].values
y=target
boruta_selector.fit(x,y)

print("==============BORUTA==============")
print (boruta_selector.n_features_)