from preprocess.FeatureSelection import FeatureSelection

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from preprocess.Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

class PIMP(FeatureSelection):
    def __init__(self, feature_score_name, corr_score_name = 'corr_score.csv', nb_runs=80):
        super(PIMP, self).__init__(feature_score_name)
        self.nb_runs = nb_runs
        self.corr_score_name=corr_score_name

    def load_data(self):
        dataset = Dataset('train_agg_id1.csv', 'test_agg_id1.csv')
        self.train_X, self.target, self.test, self.features, self.cate_features = dataset.preprocess(reload=True)

    def get_feature_importances(self, data, shuffle, target, seed=2016, train_features = None, categorical_feats = None):
        if train_features == None:
        # Gather real features
            train_features = [f for f in data.columns.values if f not in ['target', 'card_id']]
        # Go over fold and keep track of CV score (train and valid) and feature importances

        # Shuffle target if required
        y = target.copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = target.copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)

        lgb_params = {'num_leaves': 32,
                 'min_data_in_leaf': 20,
                 'objective': 'regression',
                 'max_depth': 4,
                 'learning_rate': 0.005,
                 "min_child_samples": 20,
                 "boosting": "rf",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9,
                 "bagging_seed": 11,
                 "metric": 'rmse',
                 "lambda_l1": 0.1,
                 "nthread": 4,
                      "verbosity":-1}

        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        predictions = clf.predict(data[train_features])
        imp_df['trn_score'] = mean_squared_error(predictions, y) ** 0.5

        return imp_df

    def permutation_importance(self):
        self.load_data()
        # Seed the unexpected randomness of this world
        np.random.seed(123)
        # Get the actual importance, i.e. without shuffling
        self.actual_imp_df = self.get_feature_importances(self.train_X, False, self.target,
                                                     train_features=self.features, categorical_feats= self.cate_features)
        self.null_imp_df = self.get_null_importance()
        score_df,corr_scores_df = self.get_feature_score()

        print('successfully calculate feature score, saving feature score df and corr score df ...')
        score_df.to_csv(os.path.join('./feature_score',self.feature_score_name), index = False)
        corr_scores_df.to_csv(os.path.join('./feature_score',self.corr_score_name), index = False)
        print('saving successfully ...')


        print('saving actual and null importance ...')
        self.save_feature_distribution()
        print('saving successfully ...')


    def get_null_importance(self):
        null_imp_df = pd.DataFrame()

        import time
        start = time.time()
        dsp = ''
        for i in range(self.nb_runs):
            # Get current run importances
            imp_df = self.get_feature_importances(self.train_X, True, self.target,
                                                  train_features=self.features, categorical_feats= self.cate_features)
            imp_df['run'] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, self.nb_runs, spent)
            print(dsp, end='', flush=True)
        return null_imp_df

    def get_feature_score(self):
        print("calculate feature_score ...")
        feature_scores = []
        for _f in self.actual_imp_df['feature'].unique():
            f_null_imps_gain = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

        print("calculate correlation score")
        correlation_scores = []
        for _f in self.actual_imp_df['feature'].unique():
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = self.null_imp_df.loc[self.null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = self.actual_imp_df.loc[self.actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

        return scores_df, corr_scores_df

    def show_score_df(self, feature_dir = './feature_score'):
        if os.path.isfile(os.path.join(feature_dir,self.feature_score_name)):
            score = pd.read_csv(os.path.join(feature_dir,self.feature_score_name))

            score_split = score.sort_values(by = ['split_score'], ascending=False).reset_index(drop=True)
            score_gain = score.sort_values(by = ['gain_score'], ascending=False).reset_index(drop=True)
            score_both = score.sort_values(by = ['split_score','gain_score'], ascending=False).reset_index(drop=True)


            corr_score = pd.read_csv(os.path.join(feature_dir,self.corr_score_name))

            return score_split, score_gain, score_both, corr_score
        else:
            print("no score file exist ...")

    def save_feature_distribution(self):
        self.actual_imp_df.to_csv(os.path.join('./feature_score/null_and_actual_importance','actual_imp_df.csv'))
        self.null_imp_df.to_csv(os.path.join('./feature_score/null_and_actual_importance', 'null_imp_df.csv'))

    def plot_feature_score(self, df):
        fig = plt.figure(figsize=(16,16))
        gs = gridspec.GridSpec(1,2)
        ax = plt.subplot(gs[0,0])
        sns.barplot(x='split_score', y = 'feature', data = df.sort_values(by=['split_score'], ascending=False).iloc[0:70], ax= ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        ax = plt.subplot(gs[0,1])
        sns.barplot(x='gain_score', y='feature', data=df.sort_values(by=['gain_score'],ascending=False).iloc[0:70],ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)


if __name__ == "__main__":
    pimp = PIMP("pimp_score.csv",corr_score_name='corr_score.csv')
    #pimp.permutation_importance()
    score_split, score_gain, score_both, corr_score = pimp.show_score_df()
    # print("score_split")
    # print(score_split.head())
    # print("score_gain")
    # print(score_gain.head())
    # print("score_both")
    # print(score_both.head())
    # print("corr_score")
    # print(corr_score.head())
    pimp.plot_feature_score(score_both)
