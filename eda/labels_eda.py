import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
LABELS_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/piat_features_labels.csv')
LABELS_FILE.is_file()
data_df = pd.read_csv(LABELS_FILE)
LABELS = [
    'brisque',
    'piqe',
    'fps',
]
data_df = data_df.loc[:, LABELS]

data_df
# - Correlation
data_corr = data_df.corr()
data_corr

# - Correlation
sns.heatmap(data_corr)
aplt.xticks(ticks=np.arange(len(FEATURE_NAMES)), labels=data_corr.columns.values, rotation=45, ha='right')
data_corr

sns.displot(data_df, x='brisque', kde=True)
sns.displot(data_df, x='piqe', kde=True)
sns.displot(data_df, x='fps', kde=True)

