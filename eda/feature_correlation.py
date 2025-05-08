import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
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
LABELS = [
    'brisque',
    'piqe',
    'fps',
]
data_df.describe()

data_df
# - Correlation
data_corr = data_df.corr()
data_corr

# - Correlation
sns.heatmap(data_corr)
plt.xticks(ticks=np.arange(len(FEATURE_NAMES)), labels=data_corr.columns.values, rotation=45, ha='right')
data_corr



PIAT_FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/piat_features_labels.csv')
PCKT_SZ_FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/packet_size_features_labels.csv')
piat_data_df = pd.read_csv(PIAT_FEATURES_FILE)
piat_data_df.drop(columns=[*LABELS, 'window_end_time_stamp', 'number_of_piats_in_time_window', 'limiting_parameter', 'date', 'kbps', 'file_name', 'max_packets_in_time_window'], inplace=True)
piat_data_df.columns
pckt_sz_data_df = pd.read_csv(PCKT_SZ_FEATURES_FILE)
pckt_sz_data_df.columns
all_feats_lbls_df = pd.concat([piat_data_df, pckt_sz_data_df], axis=1)
np.setdiff1d(all_feats_lbls_df.columns.values, LABELS)
all_feats_lbls_df = all_feats_lbls_df.loc[:, ['window_end_time_stamp', *np.setdiff1d(all_feats_lbls_df.columns.values, ['window_end_time_stamp', *LABELS]), *LABELS]]
all_feats_lbls_df.columns
all_feats_lbls_df.loc[:, LABELS]
all_feats_lbls_df.to_csv('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/all_features_labels.csv', index=False)
all_feats_lbls_df.columns
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
all_feats_lbls_df.rename(columns=FEATURE_NAMES, inplace=True)
all_feats_lbls_df.columns
