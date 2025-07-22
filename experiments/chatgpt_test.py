
import joblib
import pandas as pd

model = joblib.load("random_forest_qoe_model.joblib")
DATA_FILE = '/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/seperated_features_and_labels/cv1/all/test_features.csv'

X = pd.read_csv(DATA_FILE)
model(X)
