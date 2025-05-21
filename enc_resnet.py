import itertools
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from configs.params import (
    TS,
    BATCH_SIZE,
    LR_SCHEDULES,
    MICRO_PCKT_SZ_FEATURES,
    MICRO_PIAT_FEATURES,
    DEVICE,
    CV_ROOT_DIR,
    OUTPUT_DIR,
    LABELS,
    DROP_SCHEDULE
)
from models import EncResNet
from utils.data_utils import get_train_val_split, EncDS, EncRowDS
from utils.regression_utils import calc_errors
from utils.train_utils import save_checkpoint, load_checkpoint, reduce_lr, MAPELoss
from utils.aux_funcs import freeze_layers, plot_losses, get_p_drop


# - LOCAL HYPERPARAMETERS
# -- Features
# EXP_NAME = 'piat_mape_loss'
EXP_NAME = 'pckt_sz_mape_loss_no_weights'
# EXP_NAME = 'pckt_sz_mape_loss_no_samp'
if EXP_NAME.startswith('pckt_sz'):
    FEATURES = MICRO_PCKT_SZ_FEATURES
elif EXP_NAME.startswith('piat'):
    FEATURES = MICRO_PIAT_FEATURES
elif EXP_NAME.startswith('all'):
    FEATURES = [*MICRO_PCKT_SZ_FEATURES, *MICRO_PIAT_FEATURES]

# -- Head model parameters
MODEL = torchvision.models.resnet34
# WEIGHTS = torchvision.models.ResNet34_Weights.IMAGENET1K_V1

# MODEL = torchvision.models.resnet18
# WEIGHTS = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
WEIGHTS = None

IMAGE_SIZE = 5
# IMAGE_SIZE = 35

LAYERS_TO_FREEZE = []
# N_LAYERS_TO_FREEZE = 4
# LAYERS_TO_FREEZE = [
#     'conv1',
#     'bn1',
#     *[f'layer{idx}' for idx in range(1, N_LAYERS_TO_FREEZE)]
# ]

# -- Train parameters
EPOCHS = 150
INITIAL_LEARNING_RATE = 1e-3
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = MAPELoss()
AUG_P_DATA_SAMPLE = 0.1
# - CHECKS
# LOSS_FUNCTION(torch.as_tensor([50, 45, 17], dtype=torch.float32), torch.as_tensor([78, 35, 14], dtype=torch.float32))

# for epch in [3, 20, 51, 81, 82, 83, 90]:
#     reduce_lr(optimizer=OPTIMIZER, epoch=epch, schedules=LR_SCHEDULES)


def replace_zeros_with_mean(y, n_labels: int, flatten: bool = False, to_tensor: bool = False):
    # - Replace the 0's with mean value for each of the labels
    y_shape = y.shape
    no_zero_y = y.numpy().reshape(n_labels, -1).T
    true_means = no_zero_y.mean(axis=0)
    low_vals_idxs = np.argwhere(no_zero_y < 1).reshape(-1, 2)
    if low_vals_idxs.any():
        for (lw_x, lw_y) in low_vals_idxs:
            no_zero_y[lw_x, lw_y] = true_means[lw_y]
    if flatten:
        no_zero_y = no_zero_y.T.flatten()
    else:
        no_zero_y = no_zero_y.reshape(y_shape)

    if to_tensor:
        no_zero_y = torch.as_tensor(no_zero_y, dtype=torch.float32)

    return no_zero_y


def run_train(model: torch.nn.Module, epochs: int, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function, device: torch.device, save_dir: pathlib.Path):
    best_epch = 1
    best_loss = np.inf

    # - History
    train_loss_means, val_loss_means = np.array([]), np.array([])
    train_loss_stds, val_loss_stds = np.array([]), np.array([])

    p_drop = 0.0

    for epch in tqdm(range(1, epochs + 1)):
        p_drop = get_p_drop(
            p_drop=p_drop,
            epoch=epch,
            drop_schedule=DROP_SCHEDULE
        )
        # if epch > 0 and epch % LR_REDUCTION_FREQ == 0:
        reduce_lr(
            optimizer=optimizer,
            epoch=epch,
            schedules=LR_SCHEDULES
        )
        train_btch_losses = np.array([])
        for (X, Y) in train_data:
            X = X.to(device)
            Y = replace_zeros_with_mean(
                y=Y,
                n_labels=train_data.dataset.lbls.shape[-1],
                flatten=False,
                to_tensor=True
            )
            Y = Y.to(device)

            # - Zero gradients for each batch
            optimizer.zero_grad()

            # - Compute outputs
            outputs = model(X, p_drop)

            # - Calculate the loss
            loss = loss_function(Y, outputs)
            loss.backward()

            # - Adjust the weights
            optimizer.step()

            train_btch_losses = np.append(train_btch_losses, loss.item())

        train_btch_loss_mean, train_btch_loss_std = train_btch_losses.mean(), train_btch_losses.std()

        train_loss_means = np.append(train_loss_means, train_btch_loss_mean)
        train_loss_stds = np.append(train_loss_stds, train_btch_loss_std)

        print(f'\n> Mean train loss for epoch {epch}: {train_btch_loss_mean:.5f}+/-{train_btch_loss_std:.6}')

        model.eval()
        val_btch_losses = np.array([])
        with torch.no_grad():
            for (X, Y) in val_data:
                X = X.to(device)
                Y = replace_zeros_with_mean(
                    y=Y,
                    n_labels=val_data.dataset.lbls.shape[-1],
                    flatten=False,
                    to_tensor=True
                )
                Y = Y.to(device)

                outputs = model(X, p_drop)

                loss = loss_function(Y, outputs)

                val_btch_losses = np.append(val_btch_losses, loss.item())

        val_btch_loss_mean, val_btch_loss_std = val_btch_losses.mean(), val_btch_losses.std()

        val_loss_means = np.append(val_loss_means, val_btch_loss_mean)
        val_loss_stds = np.append(val_loss_stds, val_btch_loss_std)

        print(f'\n> Mean val loss for epoch {epch}: {val_btch_loss_mean:.5f}+/-{val_btch_loss_std:.6}')

        plot_losses(
            train_losses=train_loss_means,
            val_losses=val_loss_means,
            train_stds=train_loss_stds,
            val_stds=val_loss_stds,
            save_dir=save_dir
        )

        if val_btch_loss_mean < best_loss:
            print(f'> Saving checkpoint for epoch {best_epch} with loss {val_btch_loss_mean:.5f} < {best_loss:.5f}')

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filename=save_dir / 'best_checkpoint.ckpt'
            )

            # - Update the best loss
            best_loss = val_btch_loss_mean
            best_epch = epch

    # - Load the checkpoint with the best loss
    print(f'> Loading best checkpoint from {best_epch} epoch with loss {best_loss:.4f}...')
    load_checkpoint(
        model=model,
        checkpoint_file=save_dir / 'best_checkpoint.ckpt'
    )


def run_test(model: torch.nn.Module, test_data: torch.utils.data.DataLoader, labels: list, device: torch.device, save_dir: pathlib.Path):
    p_drop = 0.0
    model.eval()
    errors_df = pd.DataFrame()
    metadata_df = pd.DataFrame()
    with torch.no_grad():
        for (X, Y) in test_data:
            X = X.to(device)

            # - Get the predictions
            outputs = model(X, p_drop)

            # - Get the preds
            pred = outputs.cpu().numpy().flatten()

            # - Replace the 0's with mean value for each of the labels
            true = replace_zeros_with_mean(
                y=Y,
                n_labels=len(labels),
                flatten=True,
                to_tensor=False
            )

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


def train_test(head_model, train_data_file: pathlib.Path, test_data_file: pathlib.Path, features: list, labels: list, image_size: int, train_epochs: int, loss_function, optimizer, initial_learning_rate: float,
               batch_size: int, weights, layers_to_freeze: list, device: torch.device, save_dir: pathlib.Path):
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
    # train_ds = EncDS(
    train_ds=EncRowDS(
        data=train_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True,
        p_sample=AUG_P_DATA_SAMPLE
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # >  Val data
    val_ds=EncRowDS(
    # val_ds = EncDS(
        data=val_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True,
        p_sample=AUG_P_DATA_SAMPLE
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size // 2,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # >  Get the model
    head_mdl = head_model(weights=weights)
    # head_mdl = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    mdl = EncResNet(
        head_model=head_mdl,
        in_channels=len(features) // image_size ** 2,
        # in_channels=len(features) // image_size,
        out_size=len(labels)
        # out_size = image_size * len(labels)
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
    test_ds=EncRowDS(
    # test_ds = EncDS(
        data=test_data_df,
        features=features,
        labels=labels,
        image_size=image_size,
        chanel_mode=True,
        p_sample=AUG_P_DATA_SAMPLE
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
                head_model=MODEL,
                train_data_file=train_data_fl,
                test_data_file=test_data_fl,
                features=FEATURES,
                labels=LABELS,
                image_size=IMAGE_SIZE,
                train_epochs=EPOCHS,
                loss_function=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                initial_learning_rate=INITIAL_LEARNING_RATE,
                batch_size=BATCH_SIZE,
                weights=WEIGHTS,
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
