import os
import pathlib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import svm
from configs.params import PIAT_FEATURES, RANDOM_SEED

DATA_PATH = pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/train_data.csv')
data_df = pd.read_csv(DATA_PATH)
data_df.head()

for feat in data_df.columns:
    print(feat)
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

X = data_df.loc[:, PIAT_FEATURES]#.values
y = data_df.loc[:, ['fps']]#.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

explainer = shap.Explainer(reg)
hap_vals = explainer.shap_values(X_test)
shap.summary_plot(hap_vals, X_test)



reg = XGBRegressor()
reg.fit(X=X_train.values, y=y_train.values)
y_pred = reg.predict(X_test)

explainer = shap.Explainer(reg)
hap_vals = explainer.shap_values(X_test)
shap.summary_plot(hap_vals, X_test)
