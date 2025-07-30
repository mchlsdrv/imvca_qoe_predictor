import datetime
import pathlib
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor

from core.models import QoENet1D, EncResNet

# -----------
# - GENERAL -
# -----------
DEBUG = False
VERBOSE = True
IGNORE_WARNINGS = False
EPSILON = 1e-9
RANDOM_SEED = 0

DESCRIPTION = f'auto_encoder'
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------
# - DATA -
# --------
# -- Platform
# PLATFORM = 'linux'
# PLATFORM = 'mac'
# PLATFORM = 'wexac'
PLATFORM = 'windows'

PATHS = {
    'linux': {
        'train_data_path': pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds/train_test1/train_data.csv'),
        'test_data_path': pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds/train_test1/test_data.csv'),
        'data_root_dir': pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data'),
        'output_dir': pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/output/enc_resnet_atten'),
        'experiments_dir': pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/experiments'),
        'cv_root_dir': pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds')
    },
    'wexac': {
        'train_data_path': pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds/train_test1/train_data.csv'),
        'test_data_path': pathlib.Path('/home/projects/bagon/msidorov/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds/train_test1/test_data.csv'),
        'data_root_dir': pathlib.Path(f'/home/projects/bagon/msidorov/projects/phd/imvca_qoe_predictor/data'),
        'output_dir': pathlib.Path(f'/home/projects/bagon/msidorov/projects/phd/imvca_qoe_predictor/output'),
        'experiments_dir': pathlib.Path(f'/home/projects/bagon/msidorov/projects/phd/imvca_qoe_predictor/experiments'),
        'cv_root_dir': pathlib.Path('/home/projects/bagon/msidorov/projects/phd/imvca_qoe_predictor/data/pckt_sz_cv_10_folds')
    },
    'windows': {
        'train_data_path': pathlib.Path('/data/extracted/pckt_sz_cv_10_folds\\train_test1\\test_data.csv'),
        'test_data_path': pathlib.Path('/data/extracted/pckt_sz_cv_10_folds\\train_test1\\test_data.csv'),
        'data_root_dir': pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\phd\\imvca_qoe_predictor\\data'),
        'output_dir': pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\phd\\imvca_qoe_predictor\\output\\enc_attention_resnet'),
        'experiments_dir': pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\phd\\imvca_qoe_predictor\\experiments'),
        'cv_root_dir': pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\phd\\imvca_qoe_predictor\\data\\extracted\\pckt_sz_cv_10_folds'),
        # 'cv_root_dir': pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\phd\\imvca_qoe_predictor\\data\\extracted\\cv_10_folds'),
    },
    'mac': {
        'train_data_path': pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/train_data.csv'),
        'test_data_path': pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/test_data.csv'),
        'data_root_dir': pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data'),
        'output_dir': pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/output/enc_resnet'),
        'experiments_dir': pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/experiments'),
        'cv_root_dir': pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/pckt_sz_cv_10_folds')
    },
}
OUTPUT_DIR = PATHS.get(PLATFORM)['output_dir']
CV_ROOT_DIR = PATHS.get(PLATFORM)['cv_root_dir']
DATA_ROOT_DIR = PATHS.get(PLATFORM)['data_root_dir']
EXPERIMENTS_DIR = PATHS.get(PLATFORM)['experiments_dir']

OUTLIER_TH = 3

# ----------------
# - ARCHITECTURE -
# ----------------

# ------------
# - TRAINING -
# ------------
EPOCHS = 50
BATCH_SIZE = 64
VAL_PROP = 0.2
OPTIMIZER = torch.optim.Adam
LAYER_ACTIVATION = torch.nn.SiLU

# - LR Reduction
LR_INIT = 5e-4
LR_REDUCTION_FREQUENCY = 20
LR_REDUCTION_FACTOR = 0.8
LR_SCHEDULES = {
    0: {'mode': 'set', 'lr': 0.001},
    10: {'mode': 'set', 'lr': 0.0009},
    20: {'mode': 'set', 'lr': 0.0008},
    30: {'mode': 'set', 'lr': 0.0007},
    40: {'mode': 'set', 'lr': 0.0006},
    50: {'mode': 'set', 'lr': 0.0005},
    60: {'mode': 'set', 'lr': 0.0004},
    70: {'mode': 'set', 'lr': 0.0003},
    80: {'mode': 'set', 'lr': 0.0002},
    90: {'mode': 'set', 'lr': 0.0001},
    140: {'mode': 'reduce', 'factor': 0.9, 'min_lr': 0.00001}
}

# - Dropout
DROPOUT_EPOCH_START = 80
DROPOUT_DELTA = 25
DROPOUT_P_INIT = 0.05
DROPOUT_P_MAX = 0.3
DROP_SCHEDULE = {
    20: 0.1,
    50: 0.2,
    100: 0.3,
    140: 0.4,
}

MOMENTUM = 0.5
WEIGHT_DECAY = 1e-5

RAW_FEATURE_NAMES = {
    'frame.time_relative': 'relative_arrival_time',
    'frame.time_epoch': 'arrival_time',
    'ip.proto': 'ip_protocol',
    'ip.len': 'ip_packet_length',
    'ip.src': 'ip_source',
    'ip.dst': 'ip_destination',
    'udp.srcport': 'udp_source_port',
    'udp.dstport': 'udp_destination_port',
    'udp.length': 'udp_datagram_length',
}

PCKT_STAT_FEATURES = [
    'number_of_packet_sizes_in_time_window',
    'number_of_unique_packet_sizes_in_time_window',
    'min_packet_size',
    'max_packet_size',
    'mean_packet_size',
    'std_packet_size',
    'q1_packet_size',
    'q2_packet_size',
    'q3_packet_size',
]
PCKT_MICRO_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]

PIAT_STAT_FEATURES = [
    'number_of_piats_in_time_window',
    'number_of_unique_piats_in_time_window',
    'min_piat',
    'max_piat',
    'mean_piat',
    'std_piat',
    'q1_piat',
    'q2_piat',
    'q3_piat',
]
PIAT_MICRO_FEATURES = [f'piat_{i}' for i in range(1, 351)]



MODELS = {
    'EncResNet': EncResNet,
    'QoENet1D': QoENet1D,
    'RandomForestRegressor': RandomForestRegressor,
    'XGBoost': XGBRegressor,
    # 'CatBoost': CatBoostRegressor,
    'SVM': svm.SVR
}
