import os
import datetime
import pathlib
from configs.params import VAL_PROP, PACKET_SIZE_FEATURES, PIAT_FEATURES
from models import QoENet1D
from utils.train_utils import run_cv
import torch


DATA_TYPE = 'packet_size'
# DATA_TYPE = 'piat'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# CV_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\data\\packet_size_cv_10_folds')
# CV_ROOT_DIR = pathlib.Path('/home/projects/bagon/msidorov/projects/qoe/whatsapp/data/packet_size_cv_10_folds')
CV_ROOT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/packet_size_cv_10_folds')

# SAVE_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\output\\cv_{TS}')
# SAVE_DIR = pathlib.Path(f'/home/projects/bagon/msidorov/projects/qoe/whatsapp/output/cv_{TS}')

# LABELS = ['brisque']from torch
# LABELS = ['piqe']
LABELS = ['fps']
# LABELS = ['brisque', 'piqe', 'fps']
EPOCHS = 200
BATCH_SIZE = 254
INPUT_SIZE = 9
OUTPUT_SIZE = 1
N_LAYERS = 32
N_UNITS = 512
LOSS_FUNCTIONS = torch.nn.MSELoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3
MODEL = QoENet1D


def main():
    save_dir = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/cv_{TS}')
    os.makedirs(save_dir)
    feats = PACKET_SIZE_FEATURES if DATA_TYPE == 'packet_size' else PIAT_FEATURES
    run_cv(
        model=MODEL,
        model_params={
            'model_name': 'QoENet1d',
            'input_size': INPUT_SIZE,
            'output_size': OUTPUT_SIZE,
            'n_units': N_UNITS,
            'n_layers': N_LAYERS
        },
        n_folds=10,
        features=feats,
        labels=LABELS,
        cv_root_dir=CV_ROOT_DIR,
        save_dir=save_dir,
        nn_params={
            'batch_size': BATCH_SIZE,
            'val_prop': VAL_PROP,
            'epochs': EPOCHS,
            'loss_function': torch.nn.MSELoss,
            'learning_rate': LEARNING_RATE,
            'optimizer': torch.optim.Adam
        }
    )


if __name__ == '__main__':
    main()
