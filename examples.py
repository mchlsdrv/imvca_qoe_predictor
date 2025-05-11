import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
from scipy.stats import gaussian_kde
EPSILON = 1e-10
B = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'B': [10, 11, 12, 13, 14, 15, 16, 17, 18],
    'C': [19, 20, 21, 22, 23, 24, 25, 26, 27],
    'D': [28, 29, 30, 31, 32, 33, 34, 35, 36],
    'E': [37, 38, 39, 40, 41, 42, 43, 44, 45],
    'F': [46, 47, 48, 49, 50, 51, 52, 53, 54],
    'G': [55, 56, 57, 58, 59, 60, 61, 62, 63],
    'H': [64, 65, 66, 67, 68, 69, 70, 71, 72],
    'I': [73, 74, 75, 76, 77, 78, 79, 80, 81]
}
)
B.iloc[0:3, :]
b = B.iloc[0:3, :].T.values.reshape(-1, 3, 3)
b
c = np.array(list(map(lambda x: x.T, b)))
c

a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
a.iloc[0:3, :].T.values.flatten()
a.iloc[0:3, :].values.flatten()
a.flatten()

DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_features_labels.csv')
dt_df = pd.read_csv(DATA_PATH)
dt_df
dt_df.dropna(inplace=True)
dt_df
FEATURES = [f'piat_{i}' for i in range(1, 351)]
LABELS = ['brisque', 'piqe', 'fps']
feats = dt_df.loc[:, FEATURES]
feats
lbls = dt_df.loc[:, LABELS]
lbls
# - Standardize the values
feats_mu, feats_std = feats.mean(), feats.std()
feats = (feats - feats_mu) / (feats_std + EPSILON)
feats.values.T
feats.T
feats.isna().sum()
lbls.isna().sum()
# - Fit a random forest regressor
feats.values.shape
rf = RF(n_estimators=100, max_depth=None, random_state=42)
rf.fit(
    feats.values, 
    lbls.values
)

lbls.values.shape
x = feats.iloc[:, :].values
x.shape
x = x + np.random.randn(*x.shape)
rf.predict(x)
kde = gaussian_kde(feats.T.values)

new_samps = kde.resample(size=1000)
new_samps.shape
new_samps = (new_samps * np.expand_dims(feats_std, -1)) + np.expand_dims(feats_mu, -1) 
new_samps
