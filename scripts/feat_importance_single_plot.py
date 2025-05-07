import os
import pathlib
import time
import datetime
import itertools

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from regression_utils import normalize_columns

plt.style.use('ggplot')


def plot_feature_importance(
        data: list or np.ndarray,
        features: list,
        errors: list or np.ndarray,
        title: str,
        axes: plt.axes,
        log_file
):
    print(f'{title}:', file=log_file)
    print('\t> Summary:', file=log_file)
    for feat, scr in zip(features, data):
        print(f'\t\t{feat} score = {scr:.3f}', file=log_file)

    if isinstance(errors, list) or isinstance(errors, np.ndarray):
        axes.bar(features, data, yerr=errors)
    else:
        axes.bar(features, data)

    # - If the value is large - display it on a logarithmic scale
    if np.mean(data) > 100:
        axes.set_yscale('log')

    # - Set labels
    axes.tick_params(axis='both', which='major', labelsize=100)
    axes.set_xticks(ticks=np.arange(len(features)), labels=features, rotation=45, ha='right')
    axes.set_title(title, fontweight='bold', fontsize=120)


def calc_coefficient_importance(model, features, name: str, axes: plt.axes, log_file):
    try:
        coeffs = model.coef_.flatten()
    except AttributeError:
        coeffs = model.feature_importances_
    feats_coeffs = list(zip(features, coeffs))
    feats_coeffs = sorted(feats_coeffs, key=lambda x: np.abs(x[1]))[::-1]
    feats, coeffs = [x[0] for x in feats_coeffs], [x[1] for x in feats_coeffs]
    plot_feature_importance(
        data=coeffs,
        features=features,
        errors=None,
        title=f'{name}',
        axes=axes,
        log_file=log_file
    )


def calc_permutation_importance(model, x, y, features, scoring: str, name: str, axes: plt.axes, log_file):
    permut_imp = permutation_importance(model, x, y, scoring=scoring)
    imp_mu, imp_std = permut_imp.get('importances_mean'), permut_imp.get('importances_std')

    feats_mus_stds = list(zip(features, imp_mu, imp_std))
    feats_mus_stds = sorted(feats_mus_stds, key=lambda val: val[1])[::-1]
    feats, imp_mu, imp_std = [x[0] for x in feats_mus_stds], [x[1] for x in feats_mus_stds], [x[2] for x in feats_mus_stds]
    plot_feature_importance(
        data=imp_mu,
        features=feats,
        errors=imp_std,
        title=f'{name}',
        axes=axes,
        log_file=log_file
    )


def compute_feature_importance(model, x, y, features, name: str, scoring: str, axes: list, log_file):
    print('\t\t\t- Computing feature importance ...')
    t_strt = time.time()
    # - Coefficient importance
    calc_coefficient_importance(
        model=model,
        features=features,
        name=name,
        axes=axes[0],
        log_file=log_file
    )

    # - Permutation test
    calc_permutation_importance(
        model=model,
        x=x,
        y=y,
        features=features,
        scoring=scoring,
        name=name,
        axes=axes[1],
        log_file=log_file
    )
    print(f'\t\t\t\t- Done! Feature importance calculation took {datetime.timedelta(seconds=time.time() - t_strt)}')


def fit_model(model, x, y, model_name: str):
    print('\n\t\t\t============================================')
    print(f'\t\t\tTraining {model_name} model ...')
    t_strt = time.time()
    model.fit(x, y)
    print(f'\t\t\t\t- Done! Training took {datetime.timedelta(seconds=time.time() - t_strt)}')


def run_feature_importance_analysis(x, y, features: list, file_name_prefix: str, save_dir: str or pathlib.Path):
    save_dir = pathlib.Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    coeff_fig, coeff_ax = plt.subplots(2, 4, figsize=(200, 100), constrained_layout=True)
    permut_fig, permut_ax = plt.subplots(2, 4, figsize=(200, 100), constrained_layout=True)
    with (save_dir / 'log.txt').open(mode='w') as log_file:
        # - Linear regression
        # -- Coefficient test
        mdl = LinearRegression()
        fit_model(model=mdl, x=x, y=y, model_name='Linear Regression')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='Linear Regression',
            axes=[coeff_ax[0, 0], permut_ax[0, 0]],
            log_file=log_file
        )

        # - Ridge regression
        mdl = Ridge()
        fit_model(model=mdl, x=x, y=y, model_name='Ridge Regression')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='Ridge Regression',
            axes=[coeff_ax[0, 1], permut_ax[0, 1]],
            log_file=log_file
        )

        # - Elastic Net Regression
        mdl = ElasticNet()
        fit_model(model=mdl, x=x, y=y, model_name='Elastic Net Regression')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='Elastic Net Regression',
            axes=[coeff_ax[0, 2], permut_ax[0, 2]],
            log_file=log_file
        )

        # - Decision Trees
        mdl = DecisionTreeRegressor()
        fit_model(model=mdl, x=x, y=y, model_name='Decision Tree')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='Decision Tree',
            axes=[coeff_ax[0, 3], permut_ax[0, 3]],
            log_file=log_file
        )

        # - Random Forest
        mdl = RandomForestRegressor()
        fit_model(model=mdl, x=x, y=y.flatten(), model_name='Random Forest')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='Random Forest',
            axes=[coeff_ax[1, 0], permut_ax[1, 0]],
            log_file=log_file
        )

        # - SVM
        mdl = svm.SVR(kernel='linear')
        fit_model(model=mdl, x=x, y=y.flatten(), model_name='SVM')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='SVM',
            axes=[coeff_ax[1, 1], permut_ax[1, 1]],
            log_file=log_file
        )

        # - XGBoost
        mdl = XGBRegressor()
        fit_model(model=mdl, x=x, y=y.flatten(), model_name='XGBoost')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='XGBoost',
            axes=[coeff_ax[1, 2], permut_ax[1, 2]],
            log_file=log_file
        )

        # - AdaBoost
        mdl = AdaBoostRegressor()
        fit_model(model=mdl, x=x, y=y.flatten(), model_name='AdaBoost')
        compute_feature_importance(
            model=mdl,
            x=x,
            y=y,
            features=features,
            scoring='neg_mean_squared_error',
            name='AdaBoost',
            axes=[coeff_ax[1, 3], permut_ax[1, 3]],
            log_file=log_file
        )

        coeff_fig.savefig(save_dir / f'{file_name_prefix}_feature_importance_coeff_test.png')
        plt.close(coeff_fig)

        permut_fig.savefig(save_dir / f'{file_name_prefix}_feature_importance_permut_test.png')
        plt.close(permut_fig)


def main():
    params = list(itertools.product(DATA_TYPES, LIMITING_PARAMETERS, LABELS))

    for dt_key, lp, lbl in tqdm(params):
        save_dir_name = f'{dt_key.lower()}_{lp.lower()}_{lbl.lower()}'
        save_dir = OUTPUT_DIR / save_dir_name
        os.makedirs(save_dir, exist_ok=True)

        data_df = pd.read_csv(DATA_FILES.get(dt_key))
        data_df = data_df.dropna(axis=0)

        if lp != 'all':
            data_df = data_df.loc[data_df.loc[:, 'limiting_parameter'] == lp]

        feats_df = data_df.iloc[:, 1:len(COLUMN_NAMES)//2 + 1]
        feats_df = feats_df.rename(columns=COLUMN_NAMES)
        feat_names = feats_df.columns
        feats_df, _, _ = normalize_columns(data_df=feats_df, columns=feat_names)

        x = feats_df.values
        y = data_df.loc[:, [lbl]].values

        run_feature_importance_analysis(
            x=x,
            y=y,
            features=feat_names,
            file_name_prefix=save_dir_name,
            save_dir=save_dir
        )


OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/malware_detection/eda/feature_importance')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_FILES = {
    'packet_size': pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/malware_detection/output/packet_size_features_labels.csv'),
    'piat': pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/malware_detection/output/piat_features_labels.csv'),
}
DATA_TYPES = [
    'packet_size',
    'piat'
]
LIMITING_PARAMETERS = ['bandwidth', 'loss', 'falls', 'all']
LABELS = ['piqe', 'brisque', 'fps']
COLUMN_NAMES = {
    'number_of_packet_sizes_in_time_window': '# pckts',
    'number_of_unique_packet_sizes_in_time_window': '# pckts unq',
    'mean_packet_size': 'Mean',
    'std_packet_size': 'STD',
    'max_packet_size': 'Max',
    'min_packet_size': 'Min',
    'q1_packet_size': 'Q1',
    'q2_packet_size': 'Q2',
    'q3_packet_size': 'Q3',
    'number_of_piats_in_time_window': '# pckts',
    'number_of_unique_piats_in_time_window': '# piat unq',
    'mean_piat': 'Mean',
    'std_piat': 'STD',
    'max_piat': 'Max',
    'min_piat': 'Min',
    'q1_piat': 'Q1',
    'q2_piat': 'Q2',
    'q3_piat': 'Q3',
}

if __name__ == '__main__':
    main()
