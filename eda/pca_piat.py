import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('ggplot')

FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/encrypted_traffic/output/piat_features_labels.csv')
FEATURES_FILE.is_file()

data_df = pd.read_csv(FEATURES_FILE)
data_df.describe()

np.unique(data_df.loc[:, 'label'])
PIAT_FEATURES = {
    'number_of_piats_in_time_window': '# piats',
    'number_of_unique_piats_in_time_window': '# unq piats',
    'min_piat': 'min',
    'max_piat': 'max',
    'mean_piat': 'mean',
    'std_piat': 'std',
    'q1_piat': 'q1',
    'q2_piat': 'q2',
    'q3_piat': 'q3',
}
len(PIAT_FEATURES)
PIAT_FEATURES.keys()
feat_df = data_df.loc[:, PIAT_FEATURES.keys()]
feat_df = feat_df.rename(columns=PIAT_FEATURES)
feat_df

feat_df = (feat_df - feat_df.mean()) / feat_df.std()
feat_df

feat_corr_df = feat_df.corr()
feat_corr_df

sns.heatmap(feat_corr_df)

pca = PCA(n_components=feat_corr_df.shape[1])
feat_pca_tbl = pd.DataFrame(pca.fit_transform(feat_corr_df))
feat_pca_tbl

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(len(feat_pca_tbl.columns))], index=feat_pca_tbl.columns)
loadings

exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)
cum_exp_var

fig, ax = plt.subplots()
ax.plot(np.arange(0, len(cum_exp_var)), pca.explained_variance_ratio_)
ax.set(xlabel='Primary Component', ylabel='Cummulative Explained Variance')
ax.set_xticks(labels=np.arange(1, len(cum_exp_var) + 1), ticks=np.arange(0, len(cum_exp_var)))

sns.boxenplot(feat_pca_tbl)
