import pathlib

import numpy as np
import pandas as pd

ALL_FEATURE_COLS = ['number_of_piats_in_time_window', 'number_of_unique_piats_in_time_window', 'min_piat', 'max_piat', 'mean_piat', 'std_piat', 'q1_piat', 'q2_piat', 'q3_piat', 'number_of_packet_sizes_in_time_window', 'number_of_unique_packet_sizes_in_time_window', 'min_packet_size', 'max_packet_size', 'mean_packet_size', 'std_packet_size', 'q1_packet_size', 'q2_packet_size', 'q3_packet_size']
LABELS = ['brisque', 'piqe', 'fps']
PCKT_SZ_DATA = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/packet_size_features_labels.csv')

PIAT_DATA = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/piat_features_labels.csv')

pckt_sz_feats_df = pd.read_csv(PCKT_SZ_DATA)
lbls_df = pckt_sz_feats_df.loc[:, LABELS]

# - Get only the packet size features
pckt_sz_feats_df = pckt_sz_feats_df.loc[:, np.setdiff1d(pckt_sz_feats_df.columns.values, LABELS)]

# - Get the PIAT features and labels
piat_feats_df = pd.read_csv(PIAT_DATA)
# - Get only the PIAT features
piat_feats_df = piat_feats_df.loc[:, np.setdiff1d(piat_feats_df.columns.values, LABELS)]

# - Combine the packet size and the piat features and labels
all_feats_df = pd.concat([pckt_sz_feats_df, piat_feats_df, lbls_df], axis=1).loc[:, ALL_FEATURE_COLS]

all_feats_df.to_csv('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/all_features_labels.csv', index=False)

# - Combine the packet size and the piat features and labels
all_feats_only_df = pd.concat([pckt_sz_feats_df, piat_feats_df], axis=1).loc[:, ALL_FEATURE_COLS]
all_feats_df
all_labels_only_df = pd.concat([pckt_sz_feats_df, piat_feats_df, lbls_df], axis=1).loc[:, LABELS]
all_labels_only_df

all_feats_df.to_csv('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/all_features.csv', index=False)
all_feats_df.to_csv('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/all_lables.csv', index=False)
