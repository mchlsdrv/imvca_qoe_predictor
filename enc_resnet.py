import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from configs.params import TS, BATCH_SIZE
from models import EncResNet
from utils.data_utils import get_train_val_split, EncDS
from utils.regression_utils import calc_errors
from utils.train_utils import save_checkpoint, load_checkpoint
from utils.aux_funcs import freeze_layers, plot_losses

# - Windows paths
# TRAIN_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
# TEST_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
# OUTPUT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\experiments\\results\\convnet')

# - Mac paths
TRAIN_DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/train_data.csv')
TEST_DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/data/extracted/all_cv_10_folds/train_test1/test_data.csv')
OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/imvca_qoe_predictor/output/enc_resnet/train_{TS}')


LABELS = ['brisque', 'piqe', 'fps']
MICRO_PIAT_FEATURES = [f'piat_{i}' for i in range(1, 351)]
MICRO_PCKT_SZ_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]
FEATURES = MICRO_PCKT_SZ_FEATURES
IMAGE_SIZE = 35
N_LAYERS_TO_FREEZE = 5
MODEL = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
LAYERS_TO_FREEZE = [
    'conv1',
    'bn1',
    *[f'layer{idx}' for idx in range(1, N_LAYERS_TO_FREEZE)]
]

LR = 1e-4
EPOCHS = 1
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(model: torch.nn.Module, epochs: int, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function, device: torch.device, save_dir: pathlib.Path):
    best_epch = 1
    best_loss = np.inf
    train_losses = np.array([])
    val_losses = np.array([])
    for epch in tqdm(range(1, epochs + 1)):
        train_total_loss = 0.0
        for (X, Y) in train_data:
            X = X.to(device)
            Y = Y.to(device)

            outputs = model(X)

            loss = loss_function(Y, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
        train_mean_btch_loss = train_total_loss / len(train_data)
        train_losses = np.append(train_losses, train_mean_btch_loss)
        print(f'> Mean train loss for epoch {epch}: {train_mean_btch_loss}')

        model.eval()
        val_total_loss = 0.0
        for (X, Y) in val_data:
            X = X.to(device)
            Y = Y.to(device)

            outputs = model(X)

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


def run_test(model: torch.nn.Module, test_data: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    errors = []
    with torch.no_grad():
        for (X, Y) in test_data:
            X = X.to(device)

            outputs = model(X)

            btch_errs = calc_errors(true=Y.flatten(), predicted=outputs.cpu().numpy().flatten())

            errors.extend(btch_errs)

    errors = np.array(errors)

    print(f'> Mean test errors: {errors.mean():.3f}+/-{errors.std():.4f}')

    return errors


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
        image_size=image_size
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
        image_size=image_size
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size // 2,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # >  Get the model
    mdl = EncResNet(
        model=MODEL,
        in_channels=len(features) // image_size,
        out_size=image_size * len(labels)
    )
    mdl.to(device)

    # >  If this parameter is supplied - freeze the corresponding layers
    if isinstance(layers_to_freeze, list):
        freeze_layers(model=mdl.mdl, layers=layers_to_freeze)

    # >  Train the model
    run_train(
        model=mdl,
        epochs=train_epochs,
        train_data=train_dl,
        val_data=val_dl,
        optimizer=optimizer(filter(lambda p: p.requires_grad, mdl.parameters()), lr=learning_rate),
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
        image_size=image_size
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1
    )

    # >  Test the model
    errs = run_test(
        model=mdl,
        test_data=test_dl,
        device=device
    )
    return errs

def run_cv(cv_root_dir: pathlib.Path):
    for root, dirs, files in os.walk(cv_root_dir):
        root_dir = pathlib.Path(root)
        for fl in files:
            train_test(
                train_data_file=root_dir / 'train_data.csv',
                test_data_file=root_dir / 'test_data.csv',
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
                save_dir=SAVED
            )

        pass


if __name__ == '__main__':
    run_cv()
