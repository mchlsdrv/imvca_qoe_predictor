import pathlib
import pandas as pd
from utils.data_utils import build_test_datasets

# - Windows path
# DATA_ROOT_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted')

# - Mac path
DATA_ROOT_DIR = pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted')

DATA_NAMES = ['all', 'piat', 'packet_size']

N_FOLDS = 10
SHUFFLE = True

if __name__ == '__main__':

    # for (dt_nm, feats) in zip(DATA_NAMES, DATA_FEATURES):
    for dt_nm in DATA_NAMES:
        data_set_path = DATA_ROOT_DIR / f'{dt_nm}_features_labels.csv'
        if data_set_path.is_file():
            print(f'\n - Building {N_FOLDS}-fold CV for {dt_nm} data ...')
            save_dir = DATA_ROOT_DIR / f'{dt_nm}_cv_{N_FOLDS}_folds'
            data_df = pd.read_csv(data_set_path)

            # - Shuffle the rows of the dataset
            data_df = data_df.sample(frac=1.).reset_index(drop=True)

            # - Build the CV dataset
            build_test_datasets(
                data=data_df,
                n_folds=N_FOLDS,
                root_save_dir=save_dir
            )
            print(f'\n\t > {N_FOLDS}-fold CV for {dt_nm} data was built successfully and placed at {save_dir}')
            print('==========================================================================================')
