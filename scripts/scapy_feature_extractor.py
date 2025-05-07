import os
import pathlib
import numpy as np
import pandas as pd
from scapy.utils import rdpcap
from tqdm import tqdm

from configs.params import FEATURE_NAMES

DEBUG = False
# DEBUG = True
EPSILON = 1e-9

DATA_ROOT = pathlib.Path('./data')
OUTPUT_DIR = pathlib.Path('./output')

os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_WINDOW = 1
MAX_PACKETS_IN_TIME_WINDOW = 350
PIAT_FACTOR = 1e3


class FeatureExtractor:
    def __init__(self, time_window, max_packets_in_time_window):
        self.time_window = time_window
        self.max_packets_in_time_window = max_packets_in_time_window

    def get_window_end_time_stamps(self, pcap_df: pd.DataFrame):
        ts = pcap_df.loc[:, 'arrival_time'].values
        frst_ts = ts[0]

        # - Get relative times by subtracting the first timestamp
        rel_ts = ts - frst_ts

        # - Calculate the time windows by whole-dividing each relative timestamp by the time_window
        t_wndws = list(map(lambda x: x // self.time_window, rel_ts))

        # - Construct a DataFrame containing time_window -> time_stamp relationship
        wndw_2_rel_ts_df = pd.DataFrame({
            'time_window': t_wndws,
            'time_stamp': ts
        })

        # - Calculate the end_time_stamp for each time_window
        wndw_2_rel_ts_gb = wndw_2_rel_ts_df.groupby('time_window')['time_stamp'].agg(end_time_stamp='max')
        wndw_2_rel_ts_gb['end_time_stamp'] += self.time_window
        wndw_2_rel_ts_gb['time_window'] = wndw_2_rel_ts_gb.index

        # - Produce a sequence of the original size with time_window -> end_time_stamp relationship
        wndw_2_end_time_df = wndw_2_rel_ts_df.join(wndw_2_rel_ts_gb.set_index('time_window'), on='time_window', how='outer')

        # - Return only the end_time_stamp values of the original size
        return wndw_2_end_time_df.loc[:, 'end_time_stamp'].values

    def truncate_feature_by_time_window(self, feature_df: pd.DataFrame, feature: str):
        for idx, pckts in feature_df.iterrows():
            pckts_in_tw = pckts.iloc[0]
            new_pckts_in_tw = []

            # - If the number of packets that arrived in time window is less than the n_feature_size - pad it with trailing zeros
            n_pckts_in_tw = len(pckts_in_tw)
            if n_pckts_in_tw < self.max_packets_in_time_window:
                padded_pckts = np.zeros(self.max_packets_in_time_window)
                padded_pckts[:n_pckts_in_tw] = pckts_in_tw
                new_pckts_in_tw = padded_pckts

            # - Otherwise, take only n_feature_size
            else:
                new_pckts_in_tw = pckts_in_tw[:self.max_packets_in_time_window]
            feature_df.loc[idx, feature] = new_pckts_in_tw

    def extract_features(self, pcap_df):

        # - Extract the features related to the packet inter-arrival times
        piat_feat_df = self.extract_piat_features(pcap_df=pcap_df)

        # - Extract the features related to the packet size
        pckt_size_feat_df = self.extract_packt_size_features(pcap_df=pcap_df)

        return piat_feat_df, pckt_size_feat_df

    def extract_summary_stats(self, data_group_by: pd.DataFrame, feature: str, window_end_time_stamps: list, reduce_file_weight=False):
        """
        Method that receives a grouped by time stamp (unique time stamp) pcap data in a pd.DataFrame format,
        and extracts micro and macro (statistical) features related to the size of the packets:
        Inputs:
            - data_group_by: pd.DataFrame containing the grouped by time stamp pcap data
            - feature: the feature for calculation
            - window_end_time_stamps: list of relationship time_window -> time_stamp
            - (optional) reduce_file_weight: casts to int16 data os small numbers that is represented by integers
        Outputs:
            - packet_size_features_df: pd.DataFrame containing the following columns:
                * window_end_time_stamp (int64): the time stamp of the end of the time window defined by the self.time_window
                * number_of_{feature}_in_time_window (int16): number of packets that arrived in each time window
                * number_of_unique_{feature}_in_time_window (int16): number of unique {feature} that arrived in each time window
                * min_{feature} (int16): the minimal {feature} in each time window
                * max_{feature} (int16): the maximal {feature} in each time window
                * mean_{feature} (float32): the mean {feature} in each time window
                * std_{feature} (float32): the standard deviation of the {feature} in each time window
                * q1_{feature} (float32): the first (< 25%) quantile of the {feature} in each time window
                * q2_{feature} (float32): the second (< 50%) quantile of the {feature} in each time window
                * q3_{feature} (float32): the third (< 75%) quantile of the {feature} in each time window
                * {feature}_1 - packet_size_{self.max_packets_in_time_window} (int16): the actual {feature} of the first self.max_packets_in_time_window packets in each time window
        """

        n_pckts_in_tw = data_group_by.map(lambda x: len(x)).values
        n_unq_pckts_in_tw = data_group_by.map(lambda x: len(np.unique(x))).values

        # - Make sure that in each time window there are exactly max_packets_in_time_window packets.
        #   * If there are less - pad with trailing zeroes
        #   * If there are more - truncate
        self.truncate_feature_by_time_window(feature_df=data_group_by, feature=feature)

        # - Split the grouped in lists sizes into separate columns
        col_names = [f'{feature}_{idx}' for idx in range(1, self.max_packets_in_time_window + 1)]
        features_df = pd.DataFrame(data_group_by.loc[:, feature].to_list(), columns=col_names)
        if reduce_file_weight:
            features_df = features_df.astype(np.int16)

        # - Calculate the summary statistics of the size data
        try:
            max_feat = features_df.max(axis=1)
            min_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.min(x.values[np.argwhere(x.values > 0)]), axis=1)
            mean_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.mean(x.values[np.argwhere(x.values > 0)]), axis=1)
            std_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.std(x.values[np.argwhere(x.values > 0)]), axis=1)
            q1_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.quantile(x.values[x.values > 0], 0.75), axis=1)
            q2_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.quantile(x.values[x.values > 0], 0.5), axis=1)
            q3_feat = features_df.apply(lambda x: np.nan if len(x.values[np.argwhere(x.values > 0)]) == 0 else np.quantile(x.values[x.values > 0], 0.25), axis=1)

            features_df[f'q3_{feature}'] = q3_feat
            features_df[f'q2_{feature}'] = q2_feat
            features_df[f'q1_{feature}'] = q1_feat

            features_df[f'std_{feature}'] = std_feat
            features_df[f'mean_{feature}'] = mean_feat
            features_df[f'max_{feature}'] = max_feat
            features_df[f'min_{feature}'] = min_feat

            features_df[f'number_of_unique_{feature}s_in_time_window'] = n_unq_pckts_in_tw
            features_df[f'number_of_{feature}s_in_time_window'] = n_pckts_in_tw

            features_df[f'window_end_time_stamp'] = np.unique(window_end_time_stamps)

            # - Change the dtypes of the columns
            features_df = features_df.astype({
                f'window_end_time_stamp': np.int64,
                f'number_of_{feature}s_in_time_window': np.float32,
                f'number_of_unique_{feature}s_in_time_window': np.float32,
                f'min_{feature}': np.float32,
                f'max_{feature}': np.float32,
                f'mean_{feature}': np.float32,
                f'std_{feature}': np.float32,
                f'q1_{feature}': np.float32,
                f'q2_{feature}': np.float32,
                f'q3_{feature}': np.float32,
            })

            # - Change the order of the columns to have the summary statistics first
            features_df = features_df[
                [
                    *features_df.columns.values[-10:][::-1],
                    *features_df.columns.values[:-10]
                ]
            ]
        except ValueError as err:
            print(f'Value Error: {err}')

        features_df.dropna(axis=0, inplace=True)

        return features_df

    def extract_packt_size_features(self, pcap_df: pd.DataFrame):
        """
        Method that receives a pcap data in a pd.DataFrame format, and extracts micro and macro (statistical) features related to the size of the packets:
        Inputs:
            - pcap_df: pd.DataFrame containing the pcap data
        Outputs:
            - packet_size_features_df: pd.DataFrame containing the following columns:
                * window_end_time_stamp (int64): the time stamp of the end of the time window defined by the self.time_window
                * number_of_packet_sizes_in_time_window (int16): number of packets that arrived in each time window
                * number_of_unique_packet_sizes_in_time_window (int16): number of unique packet sizes that arrived in each time window
                * min_packet_size (int16): the minimal packet size in each time window
                * max_packet_size (int16): the maximal packet size in each time window
                * mean_packet_size (float32): the mean packet size in each time window
                * std_packet_size (float32): the standard deviation of the packet size in each time window
                * q1_packet_size (float32): the first (< 25%) quantile of the packet sizes in each time window
                * q2_packet_size (float32): the second (< 50%) quantile of the packet sizes in each time window
                * q3_packet_size (float32): the third (< 75%) quantile of the packet sizes in each time window
                * packet_size_1 - packet_size_{self.max_packets_in_time_window} (int16): the actual packet sizes of the first self.max_packets_in_time_window packets in each time window
        """
        window_end_time_stamps = self.get_window_end_time_stamps(pcap_df=pcap_df)

        pckt_size_df = pd.DataFrame(
            {
                'time_window': window_end_time_stamps,
                'packet_size': pcap_df.loc[:, 'packet_size']
             }
        )

        pckt_size_gb = pckt_size_df.groupby('time_window').agg(list)
        packet_size_features_df = self.extract_summary_stats(
            data_group_by=pckt_size_gb,
            feature='packet_size',
            window_end_time_stamps=window_end_time_stamps,
            reduce_file_weight=True
        )

        return packet_size_features_df

    def extract_piat_features(self, pcap_df: pd.DataFrame):
        """
        Method that receives a pcap data in a pd.DataFrame format, and extracts micro and macro (statistical) features related to the size of the packets:
        Inputs:
            - pcap_df: pd.DataFrame containing the pcap data
        Outputs:
            - packet_size_features_df: pd.DataFrame containing the following columns:
                * window_end_time_stamp (int64): the time stamp of the end of the time window defined by the self.time_window
                * number_of_piats_in_time_window (int16): number of packets in each time window
                * number_of_unique_piats_in_time_window (int16): number of unique piats in each time window
                * min_piat (int16): the minimal piat in each time window
                * max_piat (int16): the maximal piat in each time window
                * mean_piat (float32): the mean piat in each time window
                * std_piat (float32): the standard deviation of the piat in each time window
                * q1_piat (float32): the first (< 25%) quantile of the piats in each time window
                * q2_piat (float32): the second (< 50%) quantile of the piats in each time window
                * q3_piat (float32): the third (< 75%) quantile of the piats in each time window
                * piat_1 - packet_size_{self.max_packets_in_time_window} (int16): the actual piats of the first self.max_packets_in_time_window packets in each time window
        """
        # - Get time window identifications for the current pcap
        window_end_time_stamps = self.get_window_end_time_stamps(pcap_df=pcap_df)

        # - Calculate the Packet Inter-Arrival Times (PIAT)
        piats = np.zeros(len(pcap_df))
        piats[1:] = pcap_df.loc[:, 'arrival_time'].values[1:] + pcap_df.loc[:, 'arrival_time_fraction'].values[1:] - pcap_df.loc[:, 'arrival_time'].values[:-1] + pcap_df.loc[:, 'arrival_time_fraction'].values[:-1]

        # - Combine the Time Windows and the PIAT data to form the dataframe
        piat_df = pd.DataFrame(
            {
                'time_window': window_end_time_stamps,
                'piat': piats
            }
        )

        # - Group by time window
        piat_gb = piat_df.groupby('time_window').agg(list)

        piat_features_df = self.extract_summary_stats(
            data_group_by=piat_gb,
            feature='piat',
            window_end_time_stamps=window_end_time_stamps,
            reduce_file_weight=False
        )

        return piat_features_df


def extract_features(feature_extractor: FeatureExtractor, pcap_fl: str or pathlib.Path, feature_names: dict):
    pckts = rdpcap(str(pcap_fl))

    pckt_arr_times_splt = [str(pckt.time).split('.') for pckt in pckts]
    pckt_arr_time_whole = [int(t_pckt[0]) for t_pckt in pckt_arr_times_splt]
    pckt_arr_time_frac = [float('0.' + t_pckt[1]) for t_pckt in pckt_arr_times_splt]
    pckt_sizes = [len(pckt) for pckt in pckts]

    pcap_df = pd.DataFrame(
        {
            'arrival_time': pckt_arr_time_whole,
            'arrival_time_fraction': pckt_arr_time_frac,
            'packet_size': pckt_sizes
        }
    )

    pckt_arr_times_splt = [str(pckt.time).split('.') for pckt in pckts]
    pckt_arr_time_whole = [int(t_pckt[0]) for t_pckt in pckt_arr_times_splt]
    pckt_arr_time_frac = [float('0.' + t_pckt[1]) for t_pckt in pckt_arr_times_splt]
    pckt_sizes = [len(pckt) for pckt in pckts]

    pcap_df.loc[:, 'arrival_time'] = pckt_arr_time_whole
    pcap_df.loc[:, 'arrival_time_fraction'] = pckt_arr_time_frac
    pcap_df.loc[:, 'packet_size'] = pckt_sizes

    # - Extract the features
    piat_feats_df, pckt_size_feats_df = feature_extractor.extract_features(pcap_df=pcap_df)

    return piat_feats_df, pckt_size_feats_df

EXCLUDED_DIRS = ['Malware-PCAPs-01', 'data']

def main():
    feat_extrctr = FeatureExtractor(
        time_window=TIME_WINDOW,
        max_packets_in_time_window=MAX_PACKETS_IN_TIME_WINDOW
    )

    if DEBUG:
        extract_features(
            feature_extractor=feat_extrctr,
            pcap_fl='/Users/mchlsdrv/Desktop/projects/phd/malware_detection/data/Malware-PCAPs-01/botnet-capture-20110810-neris.pcap',
            feature_names=FEATURE_NAMES
        )

    piat_features_labels_df = pd.DataFrame()
    packet_size_features_labels_df = pd.DataFrame()
    fl_idx = 0
    for root, dirs, files in os.walk(DATA_ROOT):
        # - Get the information regarding the data collection
        data_params = root.split('/')
        lbl = data_params[-1].split('-')[0]
        root_dir = pathlib.Path(root)
        if root_dir.name not in EXCLUDED_DIRS:
            for file in tqdm(files):
                if file.endswith('pcap'):
                    print(f'\n- Extracting features for {file} ...')
                    pcap_fl = root_dir / file
                    new_piat_feats_lbls_df, new_pckt_size_feats_lbls_df = extract_features(
                        feature_extractor=feat_extrctr,
                        pcap_fl=pcap_fl,
                        feature_names=FEATURE_NAMES
                    )

                    # - Get the filename by stripping the file extension
                    file_name = file.split('.')[0]

                    # - Add the information of the file
                    new_piat_feats_lbls_df['label'] = lbl
                    new_piat_feats_lbls_df['file_name'] = file_name
                    new_piat_feats_lbls_df['max_packets_in_time_window'] = MAX_PACKETS_IN_TIME_WINDOW

                    f = piat_features_labels_df.loc[:, piat_features_labels_df.columns[10:]]

                    # - Append to the final PIAT feature - label file
                    piat_features_labels_df = pd.concat(
                        [
                            piat_features_labels_df,
                            new_piat_feats_lbls_df
                        ]
                    )

                    # - Add the information of the file
                    new_pckt_size_feats_lbls_df['label'] = lbl
                    new_pckt_size_feats_lbls_df['file_name'] = file_name
                    new_pckt_size_feats_lbls_df['max_packets_in_time_window'] = MAX_PACKETS_IN_TIME_WINDOW

                    # - Append to the final packet size feature - label file
                    packet_size_features_labels_df = pd.concat(
                        [
                            packet_size_features_labels_df,
                            new_pckt_size_feats_lbls_df
                        ]
                    )

    # - Reset the index to be increasing
    piat_features_labels_df = piat_features_labels_df.reset_index(drop=True)

    # save_dir = OUTPUT_DIR / lbl
    save_dir = OUTPUT_DIR

    # - Save the PIAT features - labels data
    save_file = save_dir / f'piat_features_labels.csv'
    piat_features_labels_df.to_csv(save_file, index=False)
    print(f'> Saved the PIAT features to {save_file} ...')

    # - Reset the index to be increasing
    packet_size_features_labels_df = packet_size_features_labels_df.reset_index(drop=True)

    # - Save the packet size features - labels data
    save_file = save_dir / f'packet_size_features_labels.csv'
    packet_size_features_labels_df.to_csv(save_file, index=False)
    print(f'> Saved the packet size features to {save_file} ...')

    print('Done!')


if __name__ == '__main__':
    main()

