import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/encrypted_traffic/output/piat_features_labels.csv')
FEATURES_FILE.is_file()

data_df = pd.read_csv(FEATURES_FILE)
data_df.describe()

data_df.loc[:, 'label']

vpn_pckts = data_df[data_df.loc[:, 'label'] == 'VPN']
len(vpn_pckts)
vpn_pckts.describe()


non_vpn_pckts = data_df[data_df.loc[:, 'label'] == 'NonVPN']
len(non_vpn_pckts)
non_vpn_pckts.describe()


mlwr_pckts = data_df[data_df.loc[:, 'label'] == 'Malware']
len(mlwr_pckts)
mlwr_pckts.describe()

data_type = data_df.loc[:, 'label'].apply(lambda x: 'Benign' if (x != 'Malware') else 'Malware')
np.unique(data_type)
data_df.loc[:, 'data_type'] = data_type
_, cnts = np.unique(data_df.loc[:, 'data_type'].values, return_counts=True)
print(f'''
    Stats:
        - Benign: {cnts[0]}
        - Malware: {cnts[-1]}
    ''')
    
data_df.columns
# -1- Inter-paket time means
sns.histplot(data=data_df, x='number_of_piats_in_time_window', hue='data_type', kde=True)

# -2- Burst duration
sns.histplot(data=data_df, x='number_of_unique_piats_in_time_window', hue='data_type', kde=True)

# -3- SRC -> DST IAT
sns.histplot(data=data_df, x='min_piat', hue='data_type', kde=True)

# -4- DST -> SRC
sns.histplot(data=data_df, x='max_piat', hue='data_type', kde=True)

# -4- DST -> SRC
sns.histplot(data=data_df, x='mean_piat', hue='data_type', kde=True)


# -4- DST -> SRC
sns.histplot(data=data_df, x='std_piat', hue='data_type', kde=True)


# -4- DST -> SRC
sns.histplot(data=data_df, x='q1_piat', hue='data_type', kde=True)

# -4- DST -> SRC
sns.histplot(data=data_df, x='q2_piat', hue='data_type', kde=True)

# -4- DST -> SRC
sns.histplot(data=data_df, x='q3_piat', hue='data_type', kde=True)

