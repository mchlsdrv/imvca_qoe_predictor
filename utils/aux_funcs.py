import _io
import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import sklearn
import torch
import matplotlib.pyplot as plt

from configs.params import (
    EPOCHS,
    LR_INIT,
    LR_REDUCTION_FREQUENCY,
    LR_REDUCTION_FACTOR,
    DROPOUT_EPOCH_START,
    BATCH_SIZE,
    VAL_PROP,
    OUTPUT_DIR,
    OUTLIER_TH,
    MOMENTUM,
    WEIGHT_DECAY,
    EPSILON,
    DESCRIPTION,
    DROPOUT_DELTA,
    DROPOUT_P_MAX, DROPOUT_P_INIT
)

def print_info(info: str, verbose: bool = True, logfile: _io.TextIOWrapper = None):
    if verbose:
        print(info)

    if isinstance(logfile, _io.TextIOWrapper):
        print(info, file=logfile)
    elif not verbose:
        print(f'WARNING: NO INFO IS BEING PRINTED! (BOTH verbose == {verbose} and type(logfile) == {type(logfile)} != _io.TextIOWrapper')

def get_files(dir_path: pathlib.Path):
    files = os.listdir(dir_path)
    if '.DS_Store' in files:
        os.remove(dir_path / '.DS_Store')
        files = os.listdir(dir_path)

    return files


def freeze_params(params: list):
    for param in params:
        param.requires_grad = False


def freeze_layers(model: torch.nn.Module, layers: list):
    for lyr in layers:
        print(f'- Freezing layer: {lyr}')
        eval(f'freeze_params(model.{lyr}.parameters())')


def get_p_drop(p_drop: float, epoch: int, drop_schedule: dict):
    drop_epchs = np.array(list(drop_schedule.keys()))
    drop_epch_idx = np.argwhere(drop_epchs <= epoch).flatten()

    if len(drop_epch_idx):
        max_drop_epch_idx = drop_epch_idx.max()
        p_drop = drop_schedule.get(drop_epchs[max_drop_epch_idx])

    return p_drop


def plot_losses(train_losses, val_losses, train_stds, val_stds, save_dir: pathlib.Path):
    # - Plot the train / val losses
    x = np.arange(1, len(train_losses) + 1)
    plt.errorbar(x=x, y=train_losses, yerr=train_stds ,label='train')
    plt.errorbar(x=x, y=val_losses, yerr=val_stds, label='validation')
    plt.xticks(ticks=x, labels=x)
    plt.locator_params(axis='x', nbins=20)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(save_dir / 'train_val_loss.png')
    plt.close()


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--lr', type=int, default=LR_INIT, help='Represents the initial learning rate of the optimizer')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--outlier_th', type=int, default=OUTLIER_TH, help='Represents the number of STDs from the mean to remove samples with')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')
    parser.add_argument('--lr_reduction_freq', type=int, default=LR_REDUCTION_FREQUENCY, help='Represents the number of epochs for the LR reduction')
    parser.add_argument('--lr_reduction_fctr', type=float, default=LR_REDUCTION_FACTOR, help='Represents the factor by which the LR reduced each LR_REDUCTION_FREQ epochs')
    parser.add_argument('--dropout_start', type=int, default=DROPOUT_EPOCH_START, help='The epoch when the dropout technique start being applied')
    parser.add_argument('--dropout_delta', type=int, default=DROPOUT_DELTA, help='The number of epochs in which the p_drop will be constant')
    parser.add_argument('--dropout_p', type=float, default=DROPOUT_P_INIT, help='The probability of the unit to be zeroed out')
    parser.add_argument('--dropout_p_max', type=float, default=DROPOUT_P_MAX, help='The maximal probability of the unit to be zeroed out')
    parser.add_argument('--desc', type=str, default=DESCRIPTION, help='The description of the current experiment')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='The momentum value to use in training')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='The weight decay value to use in training')

    return parser


def run_pca(dataset_df: pd.DataFrame):
    pca = sklearn.decomposition.PCA(n_components=dataset_df.shape[1])
    dataset_pca_df = pca.fit_transform(dataset_df)

    return dataset_pca_df, pca


def get_number_of_parameters(model: torch.nn.Module):

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_parameters = n_total_params - n_trainable_params

    return n_trainable_params, n_non_trainable_parameters


def get_errors(results: pd.DataFrame, columns: list):
    n_columns = len(columns)
    true = results.iloc[:, :n_columns].values
    pred = results.iloc[:, n_columns:].values

    columns = [column_name + '_errors(%)' for column_name in columns]

    errors = pd.DataFrame(np.abs(100 - (true + EPSILON) * 100 / (pred + EPSILON)), columns=columns)

    return errors
