import itertools
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from configs.params import TS, BATCH_SIZE, LR_REDUCTION_FCTR, LR_REDUCTION_FREQ
from models import EncResNet
from utils.data_utils import get_train_val_split, EncDS
from utils.regression_utils import calc_errors
from utils.train_utils import save_checkpoint, load_checkpoint, reduce_lr
from utils.aux_funcs import freeze_layers, plot_losses, get_p_drop

EXP_NAME = 'piat'
# EXP_NAME = 'piat_pckt_sz'
# - Windows paths
# TRAIN_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
# TEST_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
# OUTPUT_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\output\\enc_resnet')
# CV_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_cv_10_folds')

# - Mac paths
# TRAIN_DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/train_data.csv')
# TEST_DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/test_data.csv')
# OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/output/enc_resnet')
# CV_ROOT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds')

# - Mac paths
TRAIN_DATA_PATH = pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/train_data.csv')
TEST_DATA_PATH = pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/test_data.csv')
OUTPUT_DIR = pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor//output/enc_resnet')
CV_ROOT_DIR = pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds')


LABELS = ['brisque', 'piqe', 'fps']
MICRO_PIAT_FEATURES = [f'piat_{i}' for i in range(1, 351)]
MICRO_PCKT_SZ_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]
FEATURES = MICRO_PIAT_FEATURES
# FEATURES = [*MICRO_PCKT_SZ_FEATURES, *MICRO_PIAT_FEATURES]
IMAGE_SIZE = 35
N_LAYERS_TO_FREEZE = 4
PRED_THRESHOLD = 1
MODEL = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
LAYERS_TO_FREEZE = [
    'conv1',
    'bn1',
    *[f'layer{idx}' for idx in range(1, N_LAYERS_TO_FREEZE)]
]

LR = 1e-4
EPOCHS = 100
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(model: torch.nn.Module, epochs: int, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function, device: torch.device, save_dir: pathlib.Path):
    best_epch = 1
    best_loss = np.inf
    train_losses = np.array([])
    val_losses = np.array([])
    p_drop = 0.0
    for epch in tqdm(range(1, epochs + 1)):
        p_drop = get_p_drop(p_drop=p_drop, epoch=epch)
        if epch > 0 and epch % LR_REDUCTION_FREQ == 0:
            reduce_lr(
                optimizer=optimizer,
                lr_reduce_factor=LR_REDUCTION_FCTR
            )
        train_total_loss = 0.0
        for (X, Y) in train_data:
            X = X.to(device)
            Y = Y.to(device)

            outputs = model(X, p_drop)

            loss = loss_function(Y, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
        train_mean_btch_loss = train_total_loss / len(train_data)
        train_losses = np.append(train_losses, train_mean_btch_loss)
        print(f'\n> Mean train loss for epoch {epch}: {train_mean_btch_loss}')

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for (X, Y) in val_data:
                X = X.to(device)
                Y = Y.to(device)

                outputs = model(X, p_drop)

                loss = loss_function(Y, outputs)

                val_total_loss += loss.item()

        val_mean_btch_loss = val_total_loss / len(val_data)
        val_losses = np.append(val_losses, val_mean_btch_loss)

        plot_losses(train_losses=train_losses, val_losses=val_losses, save_dir=save_dir)

        if val_mean_btch_loss < best_loss:
            print(f'> Saving checkpoint for epoch {best_epch} with loss {val_mean_btch_loss:.5f} < {best_loss:.5f}')

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filename=save_dir / 'best_checkpoint.ckpt'
            )

            # - Update the best loss
            best_loss = val_mean_btch_loss
            best_epch = epch

    # - Load the checkpoint with the best loss
    print(f'> Loading best checkpoint from {best_epch} epoch with loss {best_loss:.4f}...')
    load_checkpoint(
        model=model,
        checkpoint_file=save_dir / 'best_checkpoint.ckpt'
    )


def run_test(model: torch.nn.Module, test_data: torch.utils.data.DataLoader, labels: list, device: torch.device, save_dir: pathlib.Path, sampler=None, pred_threshold: float = 1):
    p_drop = 0.0
    model.eval()
    errors_df = pd.DataFrame()
    metadata_df = pd.DataFrame()
    with torch.no_grad():
        for (X, Y) in test_data:
            X = X.to(device)

            # - Get the predictions
            outputs = model(X, p_drop)

            # - Replace the 0's with samples from the sampler model
            pred = outputs.cpu().numpy().flatten()
            low_vals_idxs = np.argwhere(pred < pred_threshold)
            if sampler is not None and low_vals_idxs.any():
                X = np.hstack(X.cpu().numpy()[0])
                Y_smpl = sampler.predict(X).T.flatten()
                pred[pred < pred_threshold] = Y_smpl[low_vals_idxs].flatten()

            # - Replace the 0's with mean value for each of the labels
            true = Y.numpy().reshape(len(labels), -1).T
            true_means = true.mean(axis=0)
            low_vals_idxs = np.argwhere(true < 1).reshape(-1, 2)
            if low_vals_idxs.any():
                for (lw_x, lw_y) in low_vals_idxs:
                    true[lw_x, lw_y] = true_means[lw_y]
            true = true.T.flatten()

            # - Calculate errors
            btch_errs = calc_errors(
                true=true,
                predicted=pred
            )

            # - As the outputs are in the form [BRISQE_ERRORS | PIQE_ERRORS | FPS_ERRORS] we need to reshape and transpose them
            btch_errs = btch_errs.reshape(3, -1).T

            # - Add the errors to history
            errors_df = pd.concat([errors_df, pd.DataFrame(columns=labels, data=btch_errs)])

            # - Add the metadata to history
            metadata_columns = [f'{typ}_{lbl}' for (typ, lbl) in itertools.product(['true', 'pred'], labels)]

            metadata_df = pd.concat([metadata_df, pd.DataFrame(columns=metadata_columns, data=np.hstack([true.reshape(3, -1).T, pred.reshape(3, -1).T]))])

    print(f'> Test stats:\n{errors_df.describe()}')

    # - Save the metadata
    metadata_df.reset_index(inplace=True, drop=True)
    metadata_df.to_csv(save_dir / 'metadata.csv', index=False)

    return errors_df.reset_index(drop=True), metadata_df


def train_test(train_data_file: pathlib.Path, test_data_file: pathlib.Path, features: list, labels: list, image_size: int, train_epochs: int, loss_function, optimizer, initial_learning_rate: float, batch_size: int, layers_to_freeze: list, device: torch.device, save_dir: pathlib.Path):
    # - Create the OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    # - Train
    # >  Load the train dataframe
    train_data_df = pd.read_csv(train_data_file)

    # >  Get only the features and the labels
    train_data_df = train_data_df.loc[:, [*features, *labels]]

    # >  Split the train data into train and validation datasets
    train_df, val_df = get_train_val_split(data=train_data_df, validation_proportion=0.2)

    # >  Train data
    train_ds = EncDS(
        data=train_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # >  Val data
    val_ds = EncDS(
        data=val_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size // 2,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # >  Get the model
    head_mdl = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    mdl = EncResNet(
        head_model=head_mdl,
        in_channels=len(features) // image_size,
        out_size=image_size * len(labels)
    )
    # >  If this parameter is supplied - freeze the corresponding layers
    if isinstance(layers_to_freeze, list):
        freeze_layers(model=mdl.mdl, layers=layers_to_freeze)

    mdl.to(device)

    # >  Train the model
    run_train(
        model=mdl,
        epochs=train_epochs,
        train_data=train_dl,
        val_data=val_dl,
        optimizer=optimizer(filter(lambda p: p.requires_grad, mdl.parameters()), lr=initial_learning_rate),
        loss_function=loss_function,
        device=device,
        save_dir=save_dir
    )

    # - Test
    # >  Load the train dataframe
    test_data_df = pd.read_csv(test_data_file)

    # >  Get only the features and the labels
    test_data_df = test_data_df.loc[:, [*features, *labels]]

    # >  Train data
    test_ds = EncDS(
        data=test_data_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1
    )

    # >  Test the model
    errs_df, _ = run_test(
        model=mdl,
        test_data=test_dl,
        sampler=test_ds.sampler,
        pred_threshold=PRED_THRESHOLD,
        labels=labels,
        device=device,
        save_dir=save_dir
    )
    return errs_df


def run_cv(cv_root_dir: pathlib.Path):
    cv_fld = 1
    cv_dirs = os.listdir(cv_root_dir)
    errors_df = pd.DataFrame()
    save_dir = OUTPUT_DIR / f'{EXP_NAME}_{len(cv_dirs)}_cv_{TS}'
    for cv_dir in tqdm(cv_dirs):
        train_data_fl = cv_root_dir / cv_dir / 'train_data.csv'
        test_data_fl = cv_root_dir / cv_dir / 'test_data.csv'
        if train_data_fl.is_file() and test_data_fl.is_file():
            cv_errs_df = train_test(
                train_data_file=train_data_fl,
                test_data_file=test_data_fl,
                features=FEATURES,
                labels=LABELS,
                image_size=IMAGE_SIZE,
                train_epochs=EPOCHS,
                loss_function=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                initial_learning_rate=LR,
                batch_size=BATCH_SIZE,
                layers_to_freeze=LAYERS_TO_FREEZE,
                device=DEVICE,
                save_dir=save_dir / f'cv{cv_fld}'
            )

            errors_df = pd.concat([errors_df, cv_errs_df])
            cv_fld += 1

    errors_df.reset_index(drop=True)
    errors_df.to_csv(save_dir / 'final_errors.csv', index=False)


if __name__ == '__main__':
    run_cv(cv_root_dir=CV_ROOT_DIR)
