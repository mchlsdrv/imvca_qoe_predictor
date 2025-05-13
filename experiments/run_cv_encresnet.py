import os
import pathlib

import torchvision

from configs.params import MODELS, TS, MICRO_PCKT_SZ_FEATURES, LABELS, DEVICE
from enc_resnet import IMAGE_SIZE
from models import EncResNet
from utils.train_utils import run_cv

MODEL_NAME = 'EncResNet'

FEATURES = MICRO_PCKT_SZ_FEATURES

N_CV_FOLDS = 10

# - Windows paths
# DATA_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted')
# EXPERIMENTS_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\experiments')
# OUTPUT_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\output')

# - Mac paths
CV_ROOT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted')
SAVE_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/output/enc_resnet/outputs/10_cv_{TS}')

BB_MODEL = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

def main():
    print(f'> Running {N_CV_FOLDS}-fold CV ...')
    cv_root_dir = CV_ROOT_DIR / f'all_cv_10_folds'

    os.makedirs(SAVE_DIR, exist_ok=True)

    mdl = EncResNet(
        model=BB_MODEL,
        in_channels=len(FEATURES) // IMAGE_SIZE,
        out_size=IMAGE_SIZE * len(LABELS)
    )
    mdl.to(DEVICE)

    with (SAVE_DIR / 'log.txt').open(mode='a') as log_fl:
        run_cv(
            model=mdl,
            model_name=MODEL_NAME,
            model_params={},
            n_folds=N_CV_FOLDS,
            features=FEATURES,
            label=LABELS,
            cv_root_dir=cv_root_dir,
            save_dir=SAVE_DIR,
            nn_params={},
            log_file=log_fl
        )


if __name__ == '__main__':
    main()
