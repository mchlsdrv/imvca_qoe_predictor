import os
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import sklearn
from tqdm import tqdm

from configs.params import LR_REDUCTION_FREQ, LR_REDUCTION_FCTR, DROPOUT_START, DROPOUT_P, OUTLIER_TH
from utils.regression_utils import calc_errors
from utils.aux_funcs import plot_losses
from utils.data_utils import calc_data_reduction, normalize_columns, get_data, get_input_data, remove_outliers


def save_checkpoint(model, optimizer, filename: str or pathlib.Path):
    filename = pathlib.Path(filename)
    print(f'=> Saving checkpoint to \'{filename}\' ...')
    state = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    # os.makedirs(filename.parent, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_file: str or pathlib.Path):
    print('=> Loading checkpoint ...')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])


def get_train_val_losses(
        model: torch.nn.Module, epochs: int,
        train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
        loss_function: torch.nn, optimizer: torch.optim,
        lr_reduce_frequency: int, lr_reduce_factor: float = 1.0,
        dropout_epoch_start: int = 20, dropout_epoch_delta: int = 20, p_dropout_init: float = 0.1,
        p_dropout_max: float = 0.5,
        checkpoint_file: pathlib.Path = None,
        checkpoint_save_frequency: int = 10,
        device: torch.device = torch.device('cpu'),
        tokenize: bool = False,
        save_dir: str or pathlib.Path = pathlib.Path('./outputs')
):
    # - Load the checkpoint to continue training
    if isinstance(checkpoint_file, pathlib.Path):
        load_checkpoint(model=model, checkpoint_file=checkpoint_file)

    # - Make sure the save_dir exists and of type pathlib.Path
    assert isinstance(save_dir, str) or isinstance(save_dir,
                                                   pathlib.Path), f'save_dir must be of type str or pathlib.Path, but is of type {type(save_dir)}!'
    # os.makedirs(save_dir, exist_ok=True)
    save_dir = pathlib.Path(save_dir)

    # - Initialize the p_drop to 0.0
    p_drop = 0.0

    # - Initialize the train / val loss arrays
    train_losses = np.array([])
    val_losses = np.array([])

    for epch in range(epochs):
        model.train(True)
        if epch > dropout_epoch_start and p_drop < p_dropout_max:
            p_drop = p_dropout_init * ((epch - dropout_epoch_start + dropout_epoch_delta) // dropout_epoch_delta)
        print(f'\nEpoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
        print(f'\t ** INFO ** p_drop = {p_drop:.4f}')
        btch_train_losses = np.array([])
        btch_pbar = tqdm(train_data_loader)
        for data in btch_pbar:

            input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

            results = model(*input_data)
            loss = loss_function(results, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            btch_train_losses = np.append(btch_train_losses, loss.item())

        # - Reduce learning rate every
        if epch > 0 and epch % lr_reduce_frequency == 0:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * lr_reduce_factor
            optimizer.param_groups[0]['lr'] = new_lr
            print(f'\t ** INFO ** The learning rate was changed from {old_lr} -> {new_lr}')

        train_losses = np.append(train_losses, btch_train_losses.mean())

        btch_val_losses = np.array([])
        model.eval()
        with torch.no_grad():
            for data in val_data_loader:

                input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

                results = model(*input_data)
                loss = loss_function(results, Y)
                btch_val_losses = np.append(btch_val_losses, loss.item())

        val_losses = np.append(val_losses, btch_val_losses.mean())

        plot_losses(train_losses=train_losses, val_losses=val_losses, save_dir=save_dir)

        # - Save the checkpoint
        if epch > 0 and epch % checkpoint_save_frequency == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                filename=save_dir / f'checkpoints/epoch_{epch}_checkpoint.ckpt'
            )

        print(f'''
==========================================
================= STATS ==================
==========================================
    Epoch {epch + 1} loss:
        > train = {train_losses.mean():.4f}
        > validation = {val_losses.mean():.4f}
==========================================
==========================================
        ''')

    return train_losses, val_losses


def train_model(model, epochs, train_data_loader, validation_data_loader, loss_function, optimizer, learning_rate, save_dir, tokenize: bool =False):
    # - Train
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # - Train the model
    train_losses, val_losses = get_train_val_losses(
        model=model,
        epochs=epochs,
        train_data_loader=train_data_loader,
        val_data_loader=validation_data_loader,
        loss_function=loss_function(),
        optimizer=optimizer(model.parameters(), lr=learning_rate),
        lr_reduce_frequency=LR_REDUCTION_FREQ,
        lr_reduce_factor=LR_REDUCTION_FCTR,
        dropout_epoch_start=DROPOUT_START,
        p_dropout_init=DROPOUT_P,
        tokenize=tokenize,
        device=device,
        save_dir=save_dir
    )

    # - Save the train / val loss metadata
    np.save(save_dir / 'train_losses.npy', train_losses)
    np.save(save_dir / 'val_losses.npy', val_losses)


def test_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, tokenize: bool = False, device: torch.device = torch.device('cpu')):
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            input_data, Y = get_input_data(data=data, tokenize=tokenize, device=device)

            # - Calculate predictions
            preds = model(*input_data)

            # - Append to the output arrays
            y_true = np.append(y_true, Y.cpu().numpy())
            y_pred = np.append(y_pred, preds.cpu().numpy())

    return y_true, y_pred


def run_cv(model, model_name: str,  model_params: dict or None, cv_root_dir: pathlib.Path or str, n_folds: int, features: list, label: str, save_dir: pathlib.Path or None, nn_params: dict, log_file):
    print(f'> Running {n_folds}-CV ...', file=log_file)
    test_objective = ''
    tokenize = True if model_name == 'bert-base-uncased' else False

    train_data_reductions = np.array([])
    test_data_reductions = np.array([])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = pd.DataFrame()
    fld_idx = 0
    for fold_dir in tqdm(os.listdir(cv_root_dir)):
        print(f'=========================================================================', file=log_file)
        print(f'> CV Fold: {fld_idx + 1}', file=log_file)
        if fold_dir[0] != '.':
            cv_save_dir = save_dir / f'cv{fld_idx + 1}'

            cv_train_dir = cv_save_dir / 'train'
            os.makedirs(cv_train_dir)

            cv_test_dir = cv_save_dir / 'test'
            os.makedirs(cv_test_dir)

            feat_lbls_names = [*features, label]

            # - Train data
            train_df = pd.read_csv(cv_root_dir / fold_dir / 'train_data.csv')
            train_df = train_df.loc[:, feat_lbls_names]
            train_df, _, _ = normalize_columns(data_df=train_df, columns=[*features])
            train_df, train_rdct_pct = remove_outliers(
                dataset=train_df,
                columns=features,
                std_th=OUTLIER_TH,
                log_file=log_file
            )
            train_df.dropna(axis=0, inplace=True)
            train_data_reductions = np.append(train_data_reductions, train_rdct_pct)

            # - Test data
            test_df = pd.read_csv(cv_root_dir / fold_dir / 'test_data.csv')
            test_df = test_df.loc[:, feat_lbls_names]
            test_data_len_orig = len(test_df)
            test_df, _, _ = normalize_columns(data_df=test_df, columns=[*features])
            test_df.dropna(axis=0, inplace=True)

            # - Make sure that the train and the test data are both made for the same task
            train_objective = 'classification' if train_df.loc[:, label].dtype == object else 'regression'
            test_objective = 'classification' if test_df.loc[:, label].dtype == object else 'regression'
            assert train_objective == test_objective, f'ERROR: The train objective ({train_objective}) does not match the test objective ({test_objective})!'

            # -- Remove the test data rows where the label is 0
            if test_objective == 'regression':
                test_data_len_reduced = len(test_df)
                test_df = test_df.loc[test_df.loc[:, label] > 0]
                test_rdct_pct = calc_data_reduction(original_size=test_data_len_orig, reduced_size=test_data_len_reduced)
                test_data_reductions = np.append(test_data_reductions, test_rdct_pct)
            else:
                # - Convert the str labels to ints
                feat_codes = model_params.get('feature_codes')
                inv_feat_codes = {v: k for k, v in feat_codes.items()}
                train_df.loc[:, label] = train_df.loc[:, label].apply(lambda x: feat_codes.get(x))
                test_df.loc[:, label] = test_df.loc[:, label].apply(lambda x: feat_codes.get(x))

            train_data, val_data, test_data, test_ds = get_data(
                train_df=train_df,
                test_df=test_df,
                features=features,
                labels=[label],
                batch_size=nn_params.get('batch_size'),
                val_prop=nn_params.get('val_prop'),
                tokenize=tokenize
            )

            # - For NN-based models
            if isinstance(val_data, torch.utils.data.DataLoader):

                # - Build the model
                mdl = model(**model_params)

                train_model(
                    model=mdl,
                    epochs=nn_params.get('epochs'),
                    train_data_loader=train_data,
                    validation_data_loader=val_data,
                    loss_function=nn_params.get('loss_function'),
                    optimizer=nn_params.get('optimizer'),
                    learning_rate=nn_params.get('learning_rate'),
                    save_dir=cv_train_dir,
                    tokenize=tokenize
                )

                # - Test the model
                y_true, y_pred = test_model(
                    model=mdl,
                    data_loader=test_data,
                    tokenize=tokenize,
                    device=device
                )
            else:
                # - For ML-based models
                X_train, y_train = train_data[0], train_data[1]
                if isinstance(model_params, dict):
                    mdl = model(**model_params)
                else:
                    mdl = model()
                mdl.fit(
                    X_train.astype(np.float32),
                    y_train.flatten().astype(np.int16)
                )

                X_test = test_data[0]

                # - Predict the y_test labels based on X_test features
                y_pred = mdl.predict(X_test)

            y_true = test_data[1]
            if test_objective == 'regression':
                errs = calc_errors(true=y_true, predicted=y_pred)

                y_true_mean = y_true.mean()
                y_pred_mean = y_pred.mean()
                pred_vs_true_err_pct = y_pred.mean()

                print(f'''
                Mean Test Errors - {fld_idx + 1} CV Fold:
                    > Mean true values        : {y_true_mean:.2f}+/-{y_true.std():.3f}
                    > Mean predicted values   : {y_pred_mean:.2f}+/-{y_pred.std():.3f}
                    > Mean true vs pred error : {pred_vs_true_err_pct} %
                    > Mean errors             : {errs.mean():.2f}+/-{errs.std():.3f}
    ''', file=log_file)
                results = pd.concat([
                    results,
                    pd.DataFrame(
                        {
                            'true': y_true.flatten(),
                            'predicted': y_pred.flatten(),
                            'error (%)': errs.flatten()
                        }
                    )
                ], ignore_index=True)
            else:
                # - Get the unique classes
                pred_cls = np.unique(y_true)

                # - Get the metrics
                precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(y_true.flatten().astype(np.int16), y_pred.flatten().astype(np.int16))

                # - Get back the original feature names
                metrics = list(zip([inv_feat_codes.get(cls) for cls in pred_cls], precision, recall, f1_score, support))

                print(f'=========================================================================', file=log_file)
                print(f'- Results for {fld_idx + 1} CV fold', file=log_file)
                for (cls_nm, prec, rec, f1, sup) in metrics:
                    print(f'''
                            Test Scores for class {cls_nm} - {fld_idx + 1} CV Fold:
                                > Precision : {prec:.3f}
                                > Recall    : {rec:.3f}
                                > F1 Score  : {f1:.3f}
                                > Support   : {sup}
                    ''', file=log_file)

                    results = pd.concat([
                        results,
                        pd.DataFrame(
                            {
                                'class': cls_nm,
                                'precision': prec,
                                'recall': rec,
                                'f1': f1,
                                'support': sup,
                                'cv_fold': fld_idx + 1
                            },
                            index=pd.Index([0])
                        )
                    ], ignore_index=True)

            results.reset_index(drop=True, inplace=True)
            results.to_csv(cv_test_dir / f'{fld_idx}_fold_final_results.csv')

        fld_idx += 1

    results.to_csv(save_dir / f'{n_folds}_folds_cv_results.csv')


    if test_objective == 'regression':
        mean_error = results.loc[:, "error (%)"].mean()
        std_error = results.loc[:, "error (%)"].std()
        print(f'''
=========================================================================       
=========================================================================       
=    Mean Stats on {n_folds} CV for {label}:
=        Mean Errors (%)
=        ---------------
=        {mean_error:.2f}+/-{std_error:.3f}
=========================================================================       
=========================================================================       
        ''')
    else:
        results_mean_gb = results.groupby('class').agg('mean')
        results_std_gb = results.groupby('class').agg('std')

        res_means = results_mean_gb.reset_index()
        res_stds = results_std_gb.reset_index()
        res_total_mean = res_means.loc[:, ['precision', 'recall', 'f1']].mean()
        res_total_std = res_means.loc[:, ['precision', 'recall', 'f1']].std()
        print(f'''
=========================================================================       
=========================================================================       
        ''')
        print(f'> Final results for configuration: {model_params.get("feature_type")} | {label}')
        print(f'''
        - Precision: {res_total_mean.iloc[0]:.2f} +/- {res_total_std.iloc[0]:.3f} 
        - Recall: {res_total_mean.iloc[1]:.2f} +/- {res_total_std.iloc[1]:.3f}
        - F1-Score: {res_total_mean.iloc[2]:.2f} +/- {res_total_std.iloc[2]:.3f}
        ''')
        print('Per-class Performance:')
        for cls in res_means.loc[:, 'class']:
            print('---')
            print(f'Class: {cls}')
            print(f"\t- Precision: {res_means.loc[res_means.loc[:, 'class'] == cls, 'precision'].values[0]:.2f} +/- {res_stds.loc[res_stds.loc[:, 'class'] == cls, 'precision'].values[0]:.3f}")
            print(f"\t- Recall: {res_means.loc[res_means.loc[:, 'class'] == cls, 'recall'].values[0]:.2f} +/- {res_stds.loc[res_stds.loc[:, 'class'] == cls, 'recall'].values[0]:.3f}")
            print(f"\t- F1-Score: {res_means.loc[res_means.loc[:, 'class'] == cls, 'f1'].values[0]:.2f} +/- {res_stds.loc[res_stds.loc[:, 'class'] == cls, 'f1'].values[0]:.3f}")
        print(f'''
=========================================================================       
=========================================================================       
        ''')
    print(f'''
    Mean reduced data (%)
        - Train: {0.0 if not len(train_data_reductions) else train_data_reductions.mean():.2f}+/-{0.0 if not len(train_data_reductions) else train_data_reductions.std():.3f}
        - Test: {0.0 if not len(test_data_reductions) else test_data_reductions.mean():.2f}+/-{0.0 if not len(test_data_reductions) else test_data_reductions.std():.3f}
    ''')
    print(f'''
    Mean reduced data (%)
        - Train: {0.0 if not len(train_data_reductions) else train_data_reductions.mean():.2f}+/-{0.0 if not len(train_data_reductions) else train_data_reductions.std():.3f}
        - Test: {0.0 if not len(test_data_reductions) else test_data_reductions.mean():.2f}+/-{0.0 if not len(test_data_reductions) else test_data_reductions.std():.3f}
    ''', file=log_file)
    return results
