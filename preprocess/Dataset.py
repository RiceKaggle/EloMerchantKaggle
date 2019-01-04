import os
import pandas as pd
import numpy as np
import datetime
import gc

class Dataset(object):
    def __init__(self,  train_path = 'train.csv', test_path = 'test.csv', hist_trans_path = 'historical_transactions.csv', new_trans_path='new_merchant_transactions.csv',
                 new_merc_path='merchants.csv', base_dir='../data'):
        self.train_path = os.path.join(base_dir,train_path)
        self.test_path = os.path.join(base_dir,test_path)
        self.hist_trans_path = os.path.join(base_dir, hist_trans_path)
        self.new_trans_path = os.path.join(base_dir, new_trans_path)
        self.new_merc_path = os.path.join(base_dir, new_merc_path)
        self.base_dir = base_dir

    def load_train(self):
        print('load train data ...')
        if not os.path.isfile(self.train_path):
            print('{} - train path not found ! '.format(self.train_path))
            return
        return pd.read_csv(self.train_path, parse_dates=['first_active_month'])

    def set_outlier_col(self, df_train):
        # simply set
        print('set train outlier ...')
        df_train['outliers'] = 0
        df_train.loc[df_train['target'] < -30, 'outliers'] = 1
        print('set outlier successfully')

    def load_test(self):
        print('load test data ... ')
        if not os.path.isfile(self.test_path):
            print('{} - test path not found ! '.format(self.test_path))
            return
        return pd.read_csv(self.test_path, parse_dates=['first_active_month'])

    def get_new_columns(self, name, aggs):
        return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

    def fill_hist_missing(self,df_hist_trans,df_new_merchant_trans):
        print('filling the missing value in hist ...')
        for df in [df_hist_trans, df_new_merchant_trans]:
            df['category_2'].fillna(1.0, inplace=True)
            df['category_3'].fillna('A', inplace=True)
            df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

    def load_hist_new_merchant(self):
        print('load history data ...')
        if not os.path.isfile(self.hist_trans_path):
            print('hist trans path not found ! ')
            return
        if not os.path.isfile(self.new_merc_path):
            print('new merchant path not found !')
            return
        if not os.path.isfile(self.new_trans_path):
            print('new hist trans path not found !')
            return

        df_hist_trans = pd.read_csv(self.hist_trans_path)
        df_new_merchant_trans = pd.read_csv(self.new_trans_path)
        self.fill_hist_missing(df_hist_trans, df_new_merchant_trans)

        for df in [df_hist_trans, df_new_merchant_trans]:
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            df['year'] = df['purchase_date'].dt.year
            df['weekofyear'] = df['purchase_date'].dt.weekofyear
            df['month'] = df['purchase_date'].dt.month
            df['dayofweek'] = df['purchase_date'].dt.dayofweek
            df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
            df['hour'] = df['purchase_date'].dt.hour
            df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
            df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
            # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
            df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
            df['month_diff'] += df['month_lag']

        print('reduce hist_trans & new_merchant_trans memory usage...')
        self.reduce_mem_usage(df_hist_trans)
        self.reduce_mem_usage(df_new_merchant_trans)


        return df_hist_trans, df_new_merchant_trans


    def agg1(self,df_hist_trans, df_new_merchant_trans):

        aggs = {}
        for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
                    'merchant_category_id']:
            aggs[col] = ['nunique']

        aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['purchase_date'] = ['max', 'min']
        aggs['month_lag'] = ['max', 'min', 'mean', 'var']
        aggs['month_diff'] = ['mean']
        aggs['authorized_flag'] = ['sum', 'mean']
        aggs['weekend'] = ['sum', 'mean']
        aggs['category_1'] = ['sum', 'mean']
        aggs['card_id'] = ['size']

        for col in ['category_2', 'category_3']:
            df_hist_trans[col + '_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
            aggs[col + '_mean'] = ['mean']


        new_columns = self.get_new_columns('hist', aggs)
        df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
        df_hist_trans_group.columns = new_columns
        df_hist_trans_group.reset_index(drop=False, inplace=True)
        df_hist_trans_group['hist_purchase_date_diff'] = (
        df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
        df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff'] / \
                                                            df_hist_trans_group['hist_card_id_size']
        df_hist_trans_group['hist_purchase_date_uptonow'] = (
        datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days



        aggs = {}
        for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
                    'merchant_category_id']:
            aggs[col] = ['nunique']
        aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
        aggs['purchase_date'] = ['max', 'min']
        aggs['month_lag'] = ['max', 'min', 'mean', 'var']
        aggs['month_diff'] = ['mean']
        aggs['weekend'] = ['sum', 'mean']
        aggs['category_1'] = ['sum', 'mean']
        aggs['card_id'] = ['size']

        for col in ['category_2', 'category_3']:
            df_new_merchant_trans[col + '_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform(
                'mean')
            aggs[col + '_mean'] = ['mean']

        new_columns = self.get_new_columns('new_hist', aggs)
        df_new_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
        df_new_trans_group.columns = new_columns
        df_new_trans_group.reset_index(drop=False, inplace=True)
        df_new_trans_group['new_hist_purchase_date_diff'] = (
        df_new_trans_group['new_hist_purchase_date_max'] - df_new_trans_group['new_hist_purchase_date_min']).dt.days
        df_new_trans_group['new_hist_purchase_date_average'] = df_new_trans_group['new_hist_purchase_date_diff'] / \
                                                               df_new_trans_group['new_hist_card_id_size']
        df_new_trans_group['new_hist_purchase_date_uptonow'] = (
        datetime.datetime.today() - df_new_trans_group['new_hist_purchase_date_max']).dt.days
        return df_hist_trans_group, df_new_trans_group



    def convert_feature_to_outlier_mean(self,df_train, df_test):
        for f in ['feature_1','feature_2','feature_3']:
            feature_mapping = df_train.groupby([f])['outlier'].mean()
            df_train[f] = df_train[f].map(feature_mapping)
            df_test[f] = df_test[f].map(feature_mapping)

    def preprocess_train_test(self, df_train, df_test):
        self.set_outlier_col(df_train)

        # add date related attr
        for df in [df_train, df_test]:
            df['first_active_month'] = pd.to_datetime(df['first_active_month'])
            df['dayofweek'] = df['first_active_month'].dt.dayofweek
            df['weekofyear'] = df['first_active_month'].dt.weekofyear
            df['month'] = df['first_active_month'].dt.month
            df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
            df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
            df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
            for f in ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_hist_purchase_date_max',
                      'new_hist_purchase_date_min']:
                df[f] = df[f].astype(np.int64) * 1e-9
            df['card_id_total'] = df['new_hist_card_id_size'] + df['hist_card_id_size']
            df['purchase_amount_total'] = df['new_hist_purchase_amount_sum'] + df['hist_purchase_amount_sum']
        self.convert_feature_to_outlier_mean(df_train, df_test)

    def reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
        start_mem - end_mem) / start_mem))
        return df


    def preprocess(self, reload = False, version='1.0'):
        df_train = self.load_train()

        df_test = self.load_test()

        if not reload:
            if version == '1.0':
                df_hist_trans, df_new_merchant_trans = self.load_hist_new_merchant()
                df_hist_trans_group, df_new_trans_group = self.agg1(df_hist_trans, df_new_merchant_trans)

                df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
                df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
                del df_hist_trans_group
                gc.collect()

                df_train = df_train.merge(df_new_trans_group, on='card_id', how='left')
                df_test = df_test.merge(df_new_trans_group, on='card_id', how='left')
                del df_new_trans_group
                gc.collect()

                del df_hist_trans;
                gc.collect()
                del df_new_merchant_trans;
                gc.collect()

                self.preprocess_train_test(df_train, df_test)

                df_train.to_csv('df_train_agg1.csv', index=False)
                df_test.to_csv('df_test_agg1.csv', index=False)

        train_Y = df_train['target']
        test = df_test
        del df_train['target']
        train_X = df_train

        features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers','Unnamed: 0']]
        cate_features = [c for c in features if 'feature_' in c]

        #features = [col for col in df_train.columns.values if col not in ['card_id','first_active_month','Unnamed: 0','outliers']]
        #cate_features = [col for col in df_train.columns.values if 'feature' in col]

        return train_X, train_Y, test, features, cate_features

    def format_transformer(self, unwanted = None ,fields = None,
                           train_file_name = 'alltrainffm.txt',
                           test_file_name = 'alltestffm.txt',
                           numeric_features = None,cate_feature=None):
        '''
        :param unwanted: unwanted list
        :param fields: fields to select
        :param fields: which field to select
        :param train_file_name: training file name
        :param test_file_name: test file name
        :param numeric_features
        :param cate_feature
        :return: processed training and testing dataset, format is <label><feature1>:<value1><feature2>:<value2>, for classification label is an integer
        indicating the class label, for regression, label is a the target value which can be any real number
        '''
        # add an extra column to the test_df

        train_df = self.load_train()
        test_df = self.load_test()

        unwanted = ['card_id','first_active_month','target']



        test_df.insert(test_df.shape[1],'target',0)


        # concat them together
        train_test_df = pd.concat([train_df, test_df])
        train_test_df = train_test_df.reset_index(drop = True)


        # default use all the features as candidate, select the features that has lower unique value which can be treated as categoricial data

        features = []
        if fields != None:
            for col in train_test_df.columns.values:
                if col in fields and col not in unwanted:
                    features.append(col)
        else:
            for col in train_test_df.columns.values:
                if col in unwanted:
                    continue
                else:
                    features.append(col)

        for col in features:
            train_no = len(train_test_df.loc[:train_df.shape[0], col].unique())
            test_no = len(train_test_df.loc[train_df.shape[0]:, col].unique())
            if train_no >= 30 or test_no >= 30:
                train_test_df.loc[:, col] = pd.cut(train_test_df.loc[:, col], 30, labels=False)


        train = train_test_df.loc[:train_df.shape[0]].copy()
        test = train_test_df.loc[train_df.shape[0]:].copy()

        categories = features

        print(categories)

        if numeric_features != None:
            numerics = numeric_features
        else:
            numerics = []

        currentcode = len(numerics)

        catdict = {}
        catcode = {}

        for x in numerics:
            catdict[x] = 0
        for x in categories:
            catdict[x] = 1

        print(catdict)



        # transform training file
        num_rows = len(train)
        if fields != None:
            num_cols = len(fields)
        else:
            num_cols = len(features)

        train_path = os.path.join(self.base_dir,train_file_name)
        with open(train_path,'w') as text_file:
            for index, row in enumerate(range(num_rows)):
                if ((index % 100000) == 0):
                    print('Train Row', index)
                datastring = ""
                datarow = train.iloc[row].to_dict()
                datastring += str(datarow['target'])

                for i,x in enumerate(catdict.keys()):
                    # if it is numeric feature
                    if catdict[x] == 0:
                        datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                    else:
                        if x not in catcode:
                            catcode[x] = {}
                            currentcode += 1
                            catcode[x][datarow[x]] = currentcode
                        elif datarow[x] not in catcode[x]:
                            currentcode += 1
                            catcode[x][datarow[x]] = currentcode

                        code = catcode[x][datarow[x]]
                        datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
                datastring += '\n'
                text_file.write(datastring)


        # transform test file
        num_rows = len(test)
        if fields != None:
            num_cols = len(fields)
        else:
            num_cols = len(features)

        test_path = os.path.join(self.base_dir,test_file_name)
        with open(test_path, 'w') as text_file:
            for index, row in enumerate(range(num_rows)):
                if ((index % 100000) == 0):
                    print('Test Row', index)
                datastring = ""
                datarow = train.iloc[row].to_dict()
                datastring += str(datarow['target'])

                for i,x in enumerate(catdict.keys()):
                    # if it is numeric feature
                    if catdict[x] == 0:
                        datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                    else:
                        if x not in catcode:
                            catcode[x] = {}
                            currentcode += 1
                            catcode[x][datarow[x]] = currentcode
                        elif datarow[x] not in catcode[x]:
                            currentcode += 1
                            catcode[x][datarow[x]] = currentcode

                        code = catcode[x][datarow[x]]
                        datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
                datastring += '\n'
                text_file.write(datastring)
        print('successfully transform the data to libSvm format ...')

    def ffmtxt2csv(self, file_name = '../submission/ffmoutput.txt', dest_name = '../submission/ffmoutput.csv'):
        test_df = pd.read_csv(self.test_path)
        #print("test_df {}".format(len(test_df)))

        submission = pd.read_csv(file_name)
        #print("submission {}".format(len(submission)))

        self.recoverdf(submission)

        final_submission = pd.DataFrame({'card_id':test_df['card_id']})
        #print("final_submission {}".format(len(final_submission)))
        final_submission['target'] = submission.loc[:,submission.columns.values[0]]
        final_submission.to_csv(dest_name,index=False)




    def recoverdf(self,df):
        first_value = df.columns.values[0]

        if 'target' not in first_value:
            # df.rename({first_value: 'target'}, axis='columns', inplace=True)
            df.loc[-1] = float(first_value)
            df.index = df.index + 1
            df.sort_index(inplace=True)


# test
if __name__ == '__main__':
    # dataset = Dataset('train.csv','test.csv')
    #
    # train_X, train_Y, test = dataset.preprocess(reload=True)

    dataset = Dataset('df_train_agg1.csv','df_test_agg1.csv')
    dataset.ffmtxt2csv()















