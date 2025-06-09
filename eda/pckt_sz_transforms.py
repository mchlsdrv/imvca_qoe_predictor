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

    
MICRO_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]

feats_df = data_df.loc[:, MICRO_FEATURES]
feats_df
import scipy
X = feats_df.values
idx = 0
X_idx = X[idx]
plt.bar(np.arange(len(X_idx)), X_idx)

X_idx_bc, _ = scipy.stats.boxcox(X_idx[X_idx > 0])
X_idx[X_idx > 0] = X_idx_bc
plt.bar(np.arange(len(X_idx)), X_idx)
