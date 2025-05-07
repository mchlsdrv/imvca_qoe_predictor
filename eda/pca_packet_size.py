import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('ggplot')

FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/encrypted_traffic/output/packet_size_features_labels.csv')
FEATURES_FILE.is_file()

data_df = pd.read_csv(FEATURES_FILE)
data_df.describe()

np.unique(data_df.loc[:, 'label'])
PACKET_SIZE_FEATURES = {
    'number_of_packet_sizes_in_time_window': '# pck sz',
    'number_of_unique_packet_sizes_in_time_window': '# unq pck sz',
    'min_packet_size': 'min',
    'max_packet_size': 'max',
    'mean_packet_size': 'mean',
    'std_packet_size': 'std',
    'q1_packet_size': 'q1',
    'q2_packet_size': 'q2',
    'q3_packet_size': 'q3',
}
len(PACKET_SIZE_FEATURES)
PACKET_SIZE_FEATURES.keys()
feat_df = data_df.loc[:, PACKET_SIZE_FEATURES.keys()]
feat_df = feat_df.rename(columns=PACKET_SIZE_FEATURES)
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
