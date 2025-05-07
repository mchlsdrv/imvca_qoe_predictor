from copy import deepcopy

import numpy as np
import pandas as pd
import pywt
# - Install as follows:
#   > brew install autoconf automake libtool pkg-config gettext json-c
#   > git clone --recurse-submodules https://github.com/nfstream/nfstream.git
#   > cd nfstream
#   > python3 -m pip install --upgrade pip
#   > python3 -m pip install -r dev_requirements.txt
#   > python3 -m pip install .
from nfstream import NFStreamer
from scapy.all import *
from tqdm import tqdm

DATA_ROOT = pathlib.Path('./data')
OUTPUT_DIR = pathlib.Path('./output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
EPSILON = 1e-9
TIME_DELTA_MIN = 10
TIME_DELTA_MAX = 1000
N_FIRST_PACKETS = 30
# - ACCOUNTING_MODE is used for packet and byte related features:
# 0 - link layer
# 1 - IP layer
# 2 - Transport layer
# 3 - Payload
ACCOUNTING_MODE = 0

def calc_wavelet(flow):
    pckt_szs = np.array(flow.splt_ps)
    wvlt_type = 'sym6'
    coeffs = pywt.wavedec(pckt_szs, wvlt_type)
    coeffs_approx = coeffs[0]
    wvlt_ldrs = np.abs(coeffs_approx) * np.sqrt(np.arange(1, len(coeffs_approx) + 1))
    wvlt_ldrs = [round(val, 3) for val in wvlt_ldrs]

    # - Pad the wavelet with 0's
    wvlt_ldrs.extend(np.zeros(20 - len(wvlt_ldrs)))

    return wvlt_ldrs


def calc_packet_size_delta_stats(flow):
    pckt_szs = np.array(flow.splt_ps)

    pckt_sz_dlts_mu, pckt_sz_dlts_std = 0.0, 0.0
    if len(pckt_szs) > 1:
        pckt_sz_dlts = np.abs(pckt_szs[1:] - pckt_szs[:-1])
        pckt_sz_dlts_mu, pckt_sz_dlts_std = pckt_sz_dlts.mean(), pckt_sz_dlts.std()

    return pckt_sz_dlts_mu, pckt_sz_dlts_std


def get_packet_size_direction(flow):
    pckt_direct_orig = np.array(flow.splt_direction)
    pckt_direct = deepcopy(pckt_direct_orig)

    # - In the original setting:
    #  0 - src2dst
    #  1 - dst2src
    # -1 - no packet
    # to make it more reasonable, we change the representation of the
    # indicators
    pckt_direct[np.argwhere(pckt_direct_orig == 0)] = 1
    pckt_direct[np.argwhere(pckt_direct_orig == 1)] = -1
    pckt_direct[np.argwhere(pckt_direct_orig == -1)] = 0

    pckt_szs_direct = np.array(flow.splt_ps) * pckt_direct

    return pckt_szs_direct


def calc_silence_windows(flow):
    tm_dlts = flow.splt_piat_ms[0]
    n_slnc_wndws = np.argwhere((TIME_DELTA_MIN < tm_dlts) & (tm_dlts < TIME_DELTA_MAX)).sum()
    return n_slnc_wndws


def calc_pps(flow):
    pps_fwrd = flow.src2dst_packets / (flow.src2dst_duration_ms + EPSILON)
    pps_bckwrd = flow.dst2src_packets / (flow.dst2src_duration_ms + EPSILON)

    return pps_fwrd, pps_bckwrd


def calc_inter_arrival_time_stats(pcap_file):
    pckts = rdpcap(pcap_file)
    pckt_times = np.array([pckt.time for pckt in pckts])
    inter_pckt_times = np.array(pckt_times[1:] - pckt_times[:-1], dtype=np.float32)
    inter_pckt_times_mu, inter_pckt_times_std = inter_pckt_times.mean(), inter_pckt_times.std()
    return inter_pckt_times_mu, inter_pckt_times_std


def main():
    data_df = pd.DataFrame()

    for root, dirs, files in os.walk(DATA_ROOT):
        dir_name = root.split('/')[-1]
        for file in tqdm(files):
            if file.endswith('pcap'):
                pcap_file = f"{root}/{file}"

                data_type = root.split('/')[-1].split('-')[0]
                file_name = file.split('.')[0]

                # - Packet arrival features
                inter_pckt_time_mu, inter_pckt_time_std = calc_inter_arrival_time_stats(
                    pcap_file=pcap_file
                )

                # - Statistical features
                streamer = NFStreamer(
                    source=pcap_file,
                    statistical_analysis=True,
                    splt_analysis=N_FIRST_PACKETS,
                    accounting_mode=ACCOUNTING_MODE,
                )

                for strm in streamer:
                    pps_frwrd, pps_bckwrd = calc_pps(flow=strm)
                    pckt_sz_deltas_mu, pckt_sz_deltas_std = calc_packet_size_delta_stats(flow=strm)
                    feats = dict(
                        data_type=data_type,
                        file_name=file_name,
                        n_first_packets=N_FIRST_PACKETS,
                        app_category=strm.application_category_name,
                        app_name=strm.application_name,
                        burst_start_time=strm.bidirectional_first_seen_ms,
                        burst_end_time=strm.bidirectional_last_seen_ms,
                        burst_duration=strm.bidirectional_duration_ms,
                        n_bytes=strm.bidirectional_bytes,

                        inter_packet_time_mean_all=inter_pckt_time_mu,
                        inter_packet_time_std_all=inter_pckt_time_std,

                        packet_size_min=strm.bidirectional_min_ps,
                        packet_size_max=strm.bidirectional_max_ps,
                        packet_size_mean=strm.bidirectional_mean_ps,
                        paket_size_std=strm.bidirectional_stddev_ps,

                        inter_packet_time_min=strm.bidirectional_min_piat_ms,
                        inter_packet_time_max=strm.bidirectional_max_piat_ms,
                        inter_packet_time_mean=strm.bidirectional_mean_piat_ms,
                        inter_packet_time_std=strm.bidirectional_stddev_piat_ms,

                        src2dst_packet_ia_time_min=strm.src2dst_min_piat_ms,
                        src2dst_packet_ia_time_max=strm.src2dst_max_piat_ms,
                        src2dst_packet_ia_time_mean=strm.src2dst_mean_piat_ms,
                        src2dst_packet_ia_time_std=strm.src2dst_stddev_piat_ms,

                        dst2src_packet_ia_time_min=strm.dst2src_min_piat_ms,
                        dst2src_packet_ia_time_max=strm.dst2src_max_piat_ms,
                        dst2src_packet_ia_time_mean=strm.dst2src_mean_piat_ms,
                        dst2src_packet_ia_time_std=strm.dst2src_stddev_piat_ms,

                        pps_forward=pps_frwrd,
                        pps_backward=pps_bckwrd,

                        packet_size_deltas_mean=pckt_sz_deltas_mu,
                        packet_size_deltas_std=pckt_sz_deltas_std,

                        n_silence_windows=calc_silence_windows(flow=strm),
                        wavelets=calc_wavelet(flow=strm),
                        packet_size_directions=get_packet_size_direction(flow=strm),

                    )
                    new_data_entry_df = pd.DataFrame([feats.values()], columns=list(feats.keys()))
                    data_df = pd.concat(
                        [
                            data_df.astype(new_data_entry_df.dtypes),
                            new_data_entry_df
                        ]
                    )
        if len(data_df) > 0:
            data_df = data_df.reset_index(drop=True)
            data_df.to_csv(OUTPUT_DIR / f'features_{dir_name}.csv', index=False)
            data_df = pd.DataFrame()
    print('Done!')


if __name__ == '__main__':
    main()

