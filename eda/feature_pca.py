import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def calc_data_reduction(original_size, reduced_size):
    reduction_pct = 100 - 100 * reduced_size / original_size
    return reduction_pct

def remove_outliers(dataset: pd.DataFrame, columns: list, std_th: int):
    L = len(dataset)
    dataset = dataset.loc[(np.abs(scipy.stats.zscore(dataset.loc[:, columns])) < std_th).all(axis=1)]
    N = len(dataset)
    reduct_pct = calc_data_reduction(L, N)
    print(f'''
Outliers
    Total before reduction: {L}
    Total after reduction: {N}
    > Present reduced: {reduct_pct:.3f}%
''')

    return dataset, reduct_pct
    
plt.style.use('ggplot')
FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/all_features.csv')
FEATURES_FILE.is_file()
data_df = pd.read_csv(FEATURES_FILE)
data_df
data_df.drop(columns=['number_of_piats_in_time_window'], inplace=True)
data_df.columns
FEATURE_NAMES = {
    'number_of_packet_sizes_in_time_window': '# pckts',
    'number_of_unique_packet_sizes_in_time_window': '# unq pckt sz',
    'min_packet_size': 'min pckt sz',
    'max_packet_size': 'max pckt sz',
    'mean_packet_size': 'mean pckt sz',
    'std_packet_size': 'std pckt sz',
    'q1_packet_size': 'Q1 pckt sz',
    'q2_packet_size': 'Q2 pckt sz',
    'q3_packet_size': 'Q3 pckt sz',
    'number_of_unique_piats_in_time_window': '# unq piats',
    'min_piat': 'min piat',
    'max_piat': 'max piat',
    'mean_piat': 'mean piat',
    'std_piat': 'std piat',
    'q1_piat': 'Q1 piat',
    'q2_piat': 'Q2 piat',
    'q3_piat': 'Q3 piat',
}
data_df.rename(columns=FEATURE_NAMES, inplace=True)
data_df.columns.values

PIAT_FEATURES = ['# unq piats', 'min piat', 'max piat', 'mean piat', 'std piat', 'Q1 piat', 'Q2 piat', 'Q3 piat']
PCK_SZ_FEATURES = ['# unq pckt sz', 'min pckt sz', 'max pckt sz', 'mean pckt sz', 'std pckt sz', 'Q1 pckt sz', 'Q2 pckt sz', 'Q3 pckt sz'] 
LABELS = [
    'brisque',
    'piqe',
    'fps',
]
data_df.describe()

data_df
piat_feats_df = data_df.loc[:, PIAT_FEATURES]
piat_feats_df = (piat_feats_df - piat_feats_df.mean()) / piat_feats_df.std() 
piat_feats_df, _ = remove_outliers(piat_feats_df, PIAT_FEATURES, 3)
pca = PCA(n_components=piat_feats_df.shape[1])
feat_pca_tbl = pd.DataFrame(pca.fit_transform(piat_feats_df))
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

sns.boxenplot(piat_feats_df)
plt.xticks(ticks=np.arange(len(PIAT_FEATURES)), labels=PIAT_FEATURES, rotation=45, ha='right')



pckt_sz_feats_df = data_df.loc[:, PCK_SZ_FEATURES]
pckt_sz_feats_df = (pckt_sz_feats_df - pckt_sz_feats_df.mean()) / pckt_sz_feats_df.std() 
pckt_sz_feats_df, _ = remove_outliers(pckt_sz_feats_df, PCK_SZ_FEATURES, 3)
pca = PCA(n_components=pckt_sz_feats_df.shape[1])
feat_pca_tbl = pd.DataFrame(pca.fit_transform(pckt_sz_feats_df))
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

sns.boxenplot(pckt_sz_feats_df)
plt.xticks(ticks=np.arange(len(PCK_SZ_FEATURES)), labels=PCK_SZ_FEATURES, rotation=45, ha='right')


