import datetime
import os
import pathlib
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from tqdm import tqdm

from configs.params import EPSILON
from utils.data_utils import get_train_val_split
from utils.regression_utils import calc_errors
from utils.train_utils import save_checkpoint, load_checkpoint


class EncDS(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, features: list, labels: list, image_size: int, p_sample: float = 0.1):
        super().__init__()
        
        self.data = data
        self.feat_cols = features
        self.lbl_cols = labels
        self.img_sz = image_size
        self.sampler = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=10)
        self.p_smpl = p_sample
        
        self.feats = None
        self.lbls = None
        self.n_chnls = None
        
        self.make_dataset()

    def __len__(self):
        return len(self.data) - self.img_sz
        
    def __getitem__(self, index):
        i_strt = index
        i_end = i_strt + self.img_sz
        
        feats = self.feats.iloc[i_strt:i_end, :].T.values
        lbls = self.lbls.iloc[i_strt:i_end, :].T.values

        # - If we add noise to the data or not
        if np.random.random() < self.p_smpl:
            feats += np.random.randn(*feats.shape)
            lbls = self.sampler.predict(feats.T)

        try:
            feats = feats.reshape(self.n_chnls, self.img_sz, self.img_sz)
            feats = np.array(list(map(lambda x: x.T, feats)))  # change the order of the features to represent joint events
        except Exception as err:
            print(err)

        lbls = lbls.flatten()

        X = torch.as_tensor(feats, dtype=torch.float32)
        Y = torch.as_tensor(lbls, dtype=torch.float32)

        return X, Y

    def make_dataset(self):
        # - Drop N/As
        self.data = self.data.dropna()
        
        # - Split the data into features and labels
        self.feats = self.data.loc[:, self.feat_cols]
        self.lbls = self.data.loc[:, self.lbl_cols]
        
        # - Calculate the number of channels
        self.n_chnls = self.feats.shape[-1] // self.img_sz
        
        # - Normalize the features
        self.feats = (self.feats - self.feats.mean()) / (self.feats.std() + EPSILON)
        
        # - Train the sampler on the train data to use in the course of training fo augmentation
        t_strt = time.time()
        print(f'- Fitting sampler ...')
        self.sampler.fit(self.feats.values, self.lbls.values)
        print(f'- Sampler training took {datetime.timedelta(seconds=time.time() - t_strt)}')


class EncResNet(torch.nn.Module):
    def __init__(self, model, in_channels: int,  out_size: int):
        super().__init__()

        self.mdl = model
        self.in_chnls = in_channels
        self.out_sz = out_size

        self.make_model()

    def make_model(self):
        fst_lyr = self.mdl.conv1
        self.mdl.conv1 = torch.nn.Conv2d(
            self.in_chnls,
            fst_lyr.out_channels,
            kernel_size=(fst_lyr.kernel_size[0], fst_lyr.kernel_size[1]),
            stride=(fst_lyr.stride[0], fst_lyr.stride[1]),
            padding=fst_lyr.padding,
            bias=False if fst_lyr.bias is None else fst_lyr.bias
        )

        lst_lyr = self.mdl.fc
        self.mdl.fc = torch.nn.Linear(
            lst_lyr.in_features,
            self.out_sz
        )

    def forward(self, x):
        x = self.mdl(x)
        return x


def freeze_params(params: list):
    for param in params:
        param.requires_grad = False


# - Windows paths
TRAIN_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
TEST_DATA_PATH = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\data\\extracted\\all_features_labels.csv')
OUTPUT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\imvca_qoe_predictor\\experiments\\results\\convnet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['brisque', 'piqe', 'fps']
PIAT_FEATURES = [f'piat_{i}' for i in range(1, 351)]
PCKT_SZ_FEATURES = [f'packet_size_{i}' for i in range(1, 351)]
FEATURES = PCKT_SZ_FEATURES
IMAGE_SIZE = 35
N_FREEZE = 5
MODEL = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
LAYERS_TO_FREEZE = [
    'conv1',
    'bn1',
    *[f'layer{idx}' for idx in range(1, N_FREEZE)]
]

LR = 1e-4
EPOCHS = 1
OPTIMIZER = torch.optim.Adam
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(model: torch.nn.Module, epochs: int, train_data: torch.utils.data.DataLoader, val_data: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function, device: torch.device, output_dir: pathlib.Path):
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

        if val_mean_btch_loss < best_loss:
            print(f'> Saving checkpoint for epoch {best_epch} with loss {val_mean_btch_loss:.5f} < {best_loss:.5f}')

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filename=output_dir / 'best_checkpoint.ckpt'
            )

            # - Update the best loss
            best_loss = val_mean_btch_loss
            best_epch = epch

    # - Load the checkpoint with the best loss
    print(f'> Loading best checkpoint from {best_epch} epoch with loss {best_loss:.4f}...')
    load_checkpoint(
        model=model,
        checkpoint_file=output_dir / 'best_checkpoint.ckpt'
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


def main():
    # - Train
    # >  Load the train dataframe
    train_data_df = pd.read_csv(TRAIN_DATA_PATH)

    # >  Get only the features and the labels
    train_data_df = train_data_df.loc[:, [*FEATURES, *LABELS]]

    # >  Split the train data into train and validation datasets
    train_df, val_df = get_train_val_split(data=train_data_df, validation_proportion=0.2)

    # >  Train data
    train_ds = EncDS(
        data=train_df,
        features=PCKT_SZ_FEATURES,
        labels=LABELS,
        image_size=IMAGE_SIZE
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # >  Val data
    val_ds = EncDS(
        data=val_df,
        features=PCKT_SZ_FEATURES,
        labels=LABELS,
        image_size=IMAGE_SIZE
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        drop_last=True,
        num_workers=4
    )

    # >  Get the model
    mdl = EncResNet(
        model=MODEL,
        in_channels=len(PCKT_SZ_FEATURES) // IMAGE_SIZE,
        out_size=IMAGE_SIZE * len(LABELS)
    )
    mdl.to(DEVICE)

    # >  If this parameter is supplied - freeze the corresponding layers
    if isinstance(LAYERS_TO_FREEZE, list):
        for lyr in LAYERS_TO_FREEZE:
            eval(f'freeze_params(mdl.mdl.{lyr}.parameters())')

    # >  Train the model
    run_train(
        model=mdl,
        epochs=EPOCHS,
        train_data=train_dl,
        val_data=val_dl,
        optimizer=OPTIMIZER(filter(lambda p: p.requires_grad, mdl.parameters()), lr=LR),
        loss_function=LOSS_FUNCTION,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )

    # - Test
    # >  Load the train dataframe
    test_data_df = pd.read_csv(TEST_DATA_PATH)

    # >  Get only the features and the labels
    test_data_df = test_data_df.loc[:, [*FEATURES, *LABELS]]

    # >  Train data
    test_ds = EncDS(
        data=test_data_df,
        features=FEATURES,
        labels=LABELS,
        image_size=IMAGE_SIZE
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
        device=DEVICE
    )
    print(f'Final errors: {errs}')


if __name__ == '__main__':
    main()
