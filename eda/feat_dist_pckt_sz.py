import os
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, mutual_info_regression, r_regression
from tqdm import tqdm

plt.style.use('ggplot')

DATA_TYPE = 'pckt_sz'

OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/eda/plots/{DATA_TYPE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_features_labels.csv')
data_df = pd.read_csv(FEATURES_FILE)

# - Remove invalid samples (NaNs and kbps < 0)
n_samples = len(data_df)
n_samples

data_df.dropna(inplace=True)
n_na = n_samples - len(data_df)
n_no_na_samples = len(data_df)
n_na

data_df = data_df.loc[data_df.loc[:, 'kbps'] > 0]
n_invalid = n_no_na_samples - len(data_df) 
n_invalid

print(f'''
    Samples:
        - Total: {n_samples}
        - NAs: {n_na} ({100 * n_na / n_samples}%)
        - Invalid: {n_invalid} ({100 * n_invalid / n_samples}%)
    ''')

    
# - SUMMARY STATISTICS
summ_stats_out_dir = OUTPUT_DIR / 'summary_stats'
os.makedirs(summ_stats_out_dir, exist_ok=True)

# -1- Number of packets
fig = sns.histplot(data=data_df, x=f'# pckts', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'n_{DATA_TYPE}_dist.png')
plt.close(fig)

# -2- Number of unique packets
fig = sns.histplot(data=data_df, x=f'# unq {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'n_unq_{DATA_TYPE}_dist.png')
plt.close(fig)

# -3- Minimal packet sizes
fig = sns.histplot(data=data_df, x=f'min {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'min_{DATA_TYPE}_dist.png')
plt.close(fig)

# -4- Maximal packet sizes
fig = sns.histplot(data=data_df, x=f'max {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'max_{DATA_TYPE}_dist.png')
plt.close(fig)

# -5- Mean packet sizes
fig = sns.histplot(data=data_df, x=f'mean {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'mean_{DATA_TYPE}_dist.png')
plt.close(fig)

# -4- STD packet sizes
fig = sns.histplot(data=data_df, x=f'std {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'std_{DATA_TYPE}_dist.png')
plt.close(fig)

# -5- Q1 
fig = sns.histplot(data=data_df, x=f'Q1 {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'q1_{DATA_TYPE}_dist.png')
plt.close(fig)

# -6- Q2 
fig = sns.histplot(data=data_df, x=f'Q2 {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'q2_{DATA_TYPE}_dist.png')
plt.close(fig)

# -7- Q3 
fig = sns.histplot(data=data_df, x=f'Q3 {DATA_TYPE.replace('_', ' ')}', hue='kbps', kde=True).get_figure()
fig.savefig(summ_stats_out_dir / f'q3_{DATA_TYPE}_dist.png')
plt.close(fig)

# - MICRO FEATURES
micro_feats_out_dir = OUTPUT_DIR / 'micro_feats'
os.makedirs(micro_feats_out_dir, exist_ok=True)

MICRO_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]

feats_df = data_df.loc[:, MICRO_FEATURES]

# -1- Correlation
MAX_COL = -1
data_corr = feats_df.iloc[:MAX_COL, :].corr()
corr_plot = sns.heatmap(data_corr)
corr_fig = corr_plot.get_figure()
corr_fig.savefig(micro_feats_out_dir / f'{DATA_TYPE}_micro_feats_corr.png')
plt.close(corr_fig)

MAX_COL = 300
data_corr = feats_df.iloc[:MAX_COL, :MAX_COL].corr()
data_corr
corr_plot = sns.heatmap(data_corr)
corr_fig = corr_plot.get_figure()
corr_fig.savefig(micro_feats_out_dir / f'{DATA_TYPE}_micro_feats_low_corr_n_{MAX_COL}.png')
plt.close(corr_fig)

# -2- PCA
pca = PCA(n_components=feats_df.shape[1])
feat_pca_tbl = pd.DataFrame(pca.fit_transform(feats_df))
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(len(feat_pca_tbl.columns))], index=feat_pca_tbl.columns)
exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)
PC_TH = 0.963
print(f'- Number of PCs < {PC_TH} = {len(cum_exp_var[cum_exp_var < PC_TH])}')
fig, ax = plt.subplots()
ax.plot(np.arange(0, len(cum_exp_var)), pca.explained_variance_ratio_)
ax.set(xlabel='Primary Component', ylabel='Cumulative Explained Variance')
ax.set_xticks(labels=np.arange(1, len(cum_exp_var) + 1), ticks=np.arange(0, len(cum_exp_var)))
fig.savefig(micro_feats_out_dir / f'{DATA_TYPE}_micro_feats_pca.png')


# -4- Distribution
micro_feats_dist_out_dir = micro_feats_out_dir / 'distributions'
os.makedirs(micro_feats_dist_out_dir, exist_ok=True)

for feat in tqdm(feats_df.columns):
    dist_plot = sns.histplot(data=feats_df, x=feat, kde=True)
    fig = dist_plot.get_figure()
    fig.savefig(micro_feats_dist_out_dir / f'{DATA_TYPE}_{feat}_dist.png')
    plt.close(fig)
    
# - FEATURE IMPORTANCE
# -1- X2
X = feats_df.values
y = data_df.loc[:, 'fps'].values

stats, p = chi2(X, y)
stats, p

# -2- Mutual Information
# (a) brisque
X = feats_df.values
y = data_df.loc[:, 'brisque'].values

mut_info = mutual_info_regression(X, y)
plt.bar(np.arange(len(mut_info)), mut_info)

# (b) piqe
X = feats_df.values
y = data_df.loc[:, 'piqe'].values

mut_info = mutual_info_regression(X, y)
plt.bar(np.arange(len(mut_info)), mut_info)

# (c) fps
X = feats_df.values
y = data_df.loc[:, 'fps'].values

mut_info = mutual_info_regression(X, y)
plt.bar(np.arange(len(mut_info)), mut_info)

# -3- R Regression
# (a) brisque
X = feats_df.values
y = data_df.loc[:, 'brisque'].values

r = r_regression(X, y)
plt.bar(np.arange(len(r)), r)

# (b) piqe
X = feats_df.values
y = data_df.loc[:, 'piqe'].values

r = r_regression(X, y)
plt.bar(np.arange(len(r)), r)

# (c) fps
X = feats_df.values
y = data_df.loc[:, 'fps'].values

r = r_regression(X, y)
plt.bar(np.arange(len(r)), r)
