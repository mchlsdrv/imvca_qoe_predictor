import itertools
import os
import datetime

from configs.params import PACKET_SIZE_FEATURES, PIAT_FEATURES, MODELS, OUTPUT_DIR, EXPERIMENTS_DIR, LABELS
from utils.train_utils import run_cv

MODEL_NAME = 'XGBoost'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

FEATURES = [('all', [*PIAT_FEATURES, *PACKET_SIZE_FEATURES])]
# FEATURES = [('piat', PIAT_FEATURES), ('packet_size', PACKET_SIZE_FEATURES)]

N_CV_FOLDS = 10


def main():
    for (lbl, (feat_typ, feats)) in itertools.product(LABELS, FEATURES):
        print(f'> Running {N_CV_FOLDS}-fold CV for {feat_typ} feature type and {lbl} label ...')
        cv_root_dir = OUTPUT_DIR / f'{feat_typ}_cv_10_folds'

        save_dir = EXPERIMENTS_DIR / MODEL_NAME / f'cv_{N_CV_FOLDS}_folds_{TS}/{feat_typ.lower()}_features/{lbl}_prediction/'
        os.makedirs(save_dir, exist_ok=True)

        with (save_dir / 'log.txt').open(mode='a') as log_fl:
            run_cv(
                model=MODELS.get(MODEL_NAME),
                model_name=MODEL_NAME,
                model_params=None,
                n_folds=N_CV_FOLDS,
                features=feats,
                label=lbl,
                cv_root_dir=cv_root_dir,
                save_dir=save_dir,
                nn_params={},
                log_file=log_fl
            )


if __name__ == '__main__':
    main()
