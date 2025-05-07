import os
import numpy as np
import pathlib
import pandas as pd
df = pd.read_csv('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data/bandwidth/2024_06_16_12_56_250KBps/pcap_2024_06_16_12_56_250KBps.csv')
df.columns
df.dtypes
EPSILON = 1e-10
CV_FOLD = 1
ALL_FEATURE_COLS = ['number_of_piats_in_time_window', 'number_of_unique_piats_in_time_window', 'min_piat', 'max_piat', 'mean_piat', 'std_piat', 'q1_piat', 'q2_piat', 'q3_piat', 'number_of_packet_sizes_in_time_window', 'number_of_unique_packet_sizes_in_time_window', 'min_packet_size', 'max_packet_size', 'mean_packet_size', 'std_packet_size', 'q1_packet_size', 'q2_packet_size', 'q3_packet_size']
PCKT_SZ_FEATURE_COLS = ['number_of_packet_sizes_in_time_window', 'number_of_unique_packet_sizes_in_time_window', 'min_packet_size', 'max_packet_size', 'mean_packet_size', 'std_packet_size', 'q1_packet_size', 'q2_packet_size', 'q3_packet_size']
PIAT_FEATURE_COLS = ['number_of_piats_in_time_window', 'number_of_unique_piats_in_time_window', 'min_piat', 'max_piat', 'mean_piat', 'std_piat', 'q1_piat', 'q2_piat', 'q3_piat']
LABEL_COLS = ['brisque', 'piqe', 'fps']

# - Data paths
DATA_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output')
DATA_DIR.is_dir()

ALL_CV_DIR = DATA_DIR / 'all_cv_10_folds'
ALL_CV_DIR.is_dir()

PCKT_SZ_CV_DIR = DATA_DIR / 'packet_size_cv_10_folds'
PCKT_SZ_CV_DIR.is_dir()

PIAT_CV_DIR = DATA_DIR / 'piat_cv_10_folds'
PIAT_CV_DIR.is_dir()

ALL_OUTPUT_DIR = DATA_DIR / f'feats_labels' / f'cv{CV_FOLD}' / 'all'
os.makedirs(ALL_OUTPUT_DIR, exist_ok=True)
ALL_OUTPUT_DIR.is_dir()

PCKT_SZ_OUTPUT_DIR = DATA_DIR / f'feats_labels' / f'cv{CV_FOLD}' / 'pckt_sz'
os.makedirs(PCKT_SZ_OUTPUT_DIR, exist_ok=True)

PIAT_OUTPUT_DIR = DATA_DIR / f'feats_labels' / f'cv{CV_FOLD}' / 'piat'
os.makedirs(PIAT_OUTPUT_DIR, exist_ok=True)

# - All Feature-Label extraction
# > TRAIN
all_train_data_df = pd.read_csv(ALL_CV_DIR / f'train_test{CV_FOLD}' / 'train_data.csv')
all_train_data_df

all_train_feats_df = all_train_data_df.loc[:, ALL_FEATURE_COLS]
all_train_feats_df

all_train_lbls_df = all_train_data_df.loc[:, LABEL_COLS]
all_train_lbls_df

# - Save the features and labels
all_train_feats_df.to_csv(ALL_OUTPUT_DIR / f'train_features.csv')
all_train_lbls_df.to_csv(ALL_OUTPUT_DIR / f'train_labels.csv')

# > TEST
all_test_data_df = pd.read_csv(ALL_CV_DIR / f'train_test{CV_FOLD}' / 'test_data.csv')
all_test_data_df

all_test_feats_df = all_test_data_df.loc[:, ALL_FEATURE_COLS]
all_test_feats_df

all_test_lbls_df = all_test_data_df.loc[:, LABEL_COLS]
all_test_lbls_df

# - Save the features and labels
all_test_feats_df.to_csv(ALL_OUTPUT_DIR / f'test_features.csv')
all_test_lbls_df.to_csv(ALL_OUTPUT_DIR / f'test_labels.csv')

# - Evaluate - PIAT
all_test_preds_df = pd.read_csv(ALL_OUTPUT_DIR / f'test_predictions.csv')
all_test_preds_df
all_test_lbls_df

all_errs_abs_pct_df = (100 - (all_test_preds_df * 100 / all_test_lbls_df)).abs()
all_errs_abs_pct_df
all_errs_abs_pct_df = all_errs_abs_pct_df.replace([np.inf, -np.inf], np.nan)
all_errs_abs_pct_df.describe()

# - Final Results
all_errs_abs_pct_df.mean()

# - Packet Size Feature-Label extraction
# > TRAIN
pckt_sz_train_data_df = pd.read_csv(PCKT_SZ_CV_DIR / f'train_test{CV_FOLD}' / 'train_data.csv')
pckt_sz_train_data_df

pckt_sz_train_feats_df = pckt_sz_train_data_df.loc[:, PCKT_SZ_FEATURE_COLS]
pckt_sz_train_feats_df

pckt_sz_train_lbls_df = pckt_sz_train_data_df.loc[:, LABEL_COLS]
pckt_sz_train_lbls_df

# - Save the features and labels
pckt_sz_train_feats_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'train_features.csv')
pckt_sz_train_lbls_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'train_labels.csv')

# > TEST
pckt_sz_test_data_df = pd.read_csv(PCKT_SZ_CV_DIR / f'train_test{CV_FOLD}' / 'test_data.csv')
pckt_sz_test_data_df

pckt_sz_test_feats_df = pckt_sz_test_data_df.loc[:, PCKT_SZ_FEATURE_COLS]
pckt_sz_test_feats_df

pckt_sz_test_lbls_df = pckt_sz_test_data_df.loc[:, LABEL_COLS]
pckt_sz_test_lbls_df

# - Save the features and labels
pckt_sz_test_feats_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'test_features.csv')
pckt_sz_test_lbls_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'test_labels.csv')

# - Evaluate - Packet Size
pckt_sz_test_preds_df = pd.read_csv(PCKT_SZ_OUTPUT_DIR / f'test_predictions.csv')
pckt_sz_test_preds_df
pckt_sz_test_lbls_df = pd.read_csv(PCKT_SZ_OUTPUT_DIR / f'pckt_sz_test_labels_cv0.csv')

pckt_sz_errs_abs_pct_df = (100 - (pckt_sz_test_preds_df * 100 / pckt_sz_test_lbls_df)).abs()
pckt_sz_errs_abs_pct_df
pckt_sz_errs_abs_pct_df = pckt_sz_errs_abs_pct_df.replace([np.inf, -np.inf], np.nan)
pckt_sz_errs_abs_pct_df.describe()

# - Final Results
pckt_sz_errs_abs_pct_df.mean()

# - PIAT Feature-Label extraction
# > TRAIN
piat_train_data_df = pd.read_csv(PIAT_CV_DIR / f'train_test{CV_FOLD}' / 'train_data.csv')
piat_train_data_df

piat_train_feats_df = piat_train_data_df.loc[:, PIAT_FEATURE_COLS]
piat_train_feats_df

piat_train_lbls_df = piat_train_data_df.loc[:, LABEL_COLS]
piat_train_lbls_df

# - Save the features and labels
piat_train_feats_df.to_csv(PIAT_OUTPUT_DIR / f'train_features.csv')
piat_train_lbls_df.to_csv(PIAT_OUTPUT_DIR / f'train_labels.csv')

# > TEST
piat_test_data_df = pd.read_csv(PIAT_CV_DIR / f'train_test{CV_FOLD}' / 'test_data.csv')
piat_test_data_df

piat_test_feats_df = piat_test_data_df.loc[:, PIAT_FEATURE_COLS]
piat_test_feats_df

piat_test_lbls_df = piat_test_data_df.loc[:, LABEL_COLS]
piat_test_lbls_df

# - Save the features and labels
piat_test_feats_df.to_csv(PIAT_OUTPUT_DIR / f'test_features.csv')
piat_test_lbls_df.to_csv(PIAT_OUTPUT_DIR / f'test_labels.csv')


# - Packet Size Feature-Label extraction
# > TRAIN
pckt_sz_train_data_df = pd.read_csv(PCKT_SZ_CV_DIR / f'train_test{CV_FOLD}' / 'train_data.csv')
pckt_sz_train_data_df

pckt_sz_train_feats_df = pckt_sz_train_data_df.loc[:, PCKT_SZ_FEATURE_COLS]
pckt_sz_train_feats_df

pckt_sz_train_lbls_df = pckt_sz_train_data_df.loc[:, LABEL_COLS]
pckt_sz_train_lbls_df

# - Save the features and labels
pckt_sz_train_feats_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'train_features.csv')
pckt_sz_train_lbls_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'train_labels.csv')

# > TEST
pckt_sz_test_data_df = pd.read_csv(PCKT_SZ_CV_DIR / f'train_test{CV_FOLD}' / 'test_data.csv')
pckt_sz_test_data_df

pckt_sz_test_feats_df = pckt_sz_test_data_df.loc[:, PCKT_SZ_FEATURE_COLS]
pckt_sz_test_feats_df

pckt_sz_test_lbls_df = pckt_sz_test_data_df.loc[:, LABEL_COLS]
pckt_sz_test_lbls_df

# - Save the features and labels
pckt_sz_test_feats_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'test_features.csv')
pckt_sz_test_lbls_df.to_csv(PCKT_SZ_OUTPUT_DIR / f'test_labels.csv')

# - Evaluate - Packet Size
pckt_sz_test_preds_df = pd.read_csv(PCKT_SZ_OUTPUT_DIR / f'test_predictions.csv')
pckt_sz_test_preds_df
pckt_sz_test_lbls_df

pckt_sz_errs_abs_pct_df = (100 - (pckt_sz_test_preds_df * 100 / pckt_sz_test_lbls_df)).abs()
pckt_sz_errs_abs_pct_df
pckt_sz_errs_abs_pct_df = pckt_sz_errs_abs_pct_df.replace([np.inf, -np.inf], np.nan)
pckt_sz_errs_abs_pct_df.describe()

# - Final Results
pckt_sz_errs_abs_pct_df.mean()

# - Evaluate - PIAT
piat_test_preds_df = pd.read_csv(PIAT_OUTPUT_DIR / f'test_predictions.csv')
piat_test_preds_df
piat_test_lbls_df

piat_errs_abs_pct_df = (100 - (piat_test_preds_df * 100 / piat_test_lbls_df)).abs()
piat_errs_abs_pct_df
piat_errs_abs_pct_df = piat_errs_abs_pct_df.replace([np.inf, -np.inf], np.nan)
piat_errs_abs_pct_df.describe()

# - Final Results
piat_errs_abs_pct_df.mean()
