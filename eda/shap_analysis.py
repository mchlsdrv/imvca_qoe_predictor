import pathlib
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn import svm
import tqdm
import seaborn as sns

RANDOM_SEED = 0

DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_features_labels.csv')
OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/eda/plots/shap')
SAVE_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted')


PACKET_SIZE_FEATURES = [
    '# pckts',
    '# unq pckt sz',
    'min pckt sz',
    'max pckt sz',
    'mean pckt sz',
    'std pckt sz',
    'Q1 pckt sz',
    'Q2 pckt sz',
    'Q3 pckt sz',
]

PIAT_FEATURES = [
    '# pckts',
    '# unq piats',
    'min piat',
    'max piat',
    'mean piat',
    'std piat',
    'Q1 piat',
    'Q2 piat',
    'Q3 piat',
]

LABELS = ['brisque', 'piqe', 'fps']

def calc_mean_inference_time(model, data: pd.DataFrame, model_name: str):
    t_exec_arr = np.array([])
    for idx in tqdm.tqdm(range(len(data) - 2)):
        t_strt = time.time()
        model.predict(data.iloc[idx + 1:idx + 2, :])
        t_exec_arr = np.append(t_exec_arr, time.time() - t_strt)
    exec_mu, exec_std = t_exec_arr.mean(), t_exec_arr.std()
    print(f'''
        === {model_name} === 
            Mean inference time on {len(data)} samples:
                {exec_mu:.4f}+/-{exec_std:.5f}
        ''')
    return t_exec_arr
    
# - DATA
data_df = pd.read_csv(DATA_PATH)
data_df = data_df.loc[:, list(set([*PACKET_SIZE_FEATURES, *PIAT_FEATURES, *LABELS]))].dropna(axis=0)
len(data_df)
data_df.head()
data_df.describe()
sns.histplot(data_df, x='brisque')
# - FEATURES
X = data_df.loc[:, PACKET_SIZE_FEATURES]
X.to_csv(SAVE_PATH / 'pckt_sz_X.csv')
X = data_df.loc[:, PIAT_FEATURES]
X.to_csv(SAVE_PATH / 'piat_X.csv')
X = data_df.loc[:, list(set([*PIAT_FEATURES, *PACKET_SIZE_FEATURES]))]
X.to_csv(SAVE_PATH / 'all_X.csv')

# - LABEL
# y = data_df.loc[:, ['brisque']]
# y = data_df.loc[:, ['piqe']]
y = data_df.loc[:, ['fps']]
Y = data_df.loc[:, LABELS]
Y.to_csv(SAVE_PATH / 'all_Y.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
X_test.head()
y_test.head()
# - MODELS
# -- SVM
reg = svm.SVR()
reg.fit(X=X_train.values[:, :], y=y_train.values[:, 0])
t_exec_arr = calc_mean_inference_time(model=reg, data=X_test, model_name='SVM')
explainer = shap.KernelExplainer(reg.predict, X_test)
# explainer = shap.KernelExplainer(reg.predict, shap.sample(X_test, 100))
shap_vals = explainer.shap_values(X_test.values)
shap.summary_plot(shap_vals, X_test.iloc[:10, :])
shap.summary_plot(shap_vals, X_test)

# -- RANDOM FOREST
reg = RandomForestRegressor()
reg.fit(X_train, y_train.values.flatten())

t_exec_arr = calc_mean_inference_time(model=reg, data=X_test, model_name='Random Forest')

explainer = shap.Explainer(reg)
hap_vals = explainer.shap_values(X_test)
shap.summary_plot(hap_vals, X_test)

# -- XGBoost
reg = XGBRegressor()
reg.fit(X=X_train.values, y=y_train.values)
t_exec_arr = calc_mean_inference_time(model=reg, data=X_test, model_name='XGB')
reg.predict(X_test)
explainer = shap.Explainer(reg)
hap_vals = explainer.shap_values(X_test.values)
shap.summary_plot(hap_vals, X_test)


# -- CatBoost
reg = CatBoostRegressor()
reg.fit(X=X_train.values, y=y_train.values, silent=True)
t_exec_arr = calc_mean_inference_time(model=reg, data=X_test, model_name='CatBoost')

explainer = shap.Explainer(reg)
hap_vals = explainer.shap_values(X_test)
shap.summary_plot(hap_vals, X_test)

