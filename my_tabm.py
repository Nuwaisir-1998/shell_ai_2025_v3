import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import optuna
import hashlib
import json
from glob import glob


def hash_dataframe(df: pd.DataFrame) -> str:
    """Stable hash for a DataFrame (content + index)"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def make_hash(hparams: dict, train_fold: pd.DataFrame, val_fold: pd.DataFrame, df_test_pred: pd.DataFrame, feature_cols: list, target_col: str, n_splits='None', seed='None') -> str:
    """Generate a unique hash string for given inputs"""
    key_data = {
        "hparams": hparams,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_hash": hash_dataframe(train_fold),
        "val_hash": hash_dataframe(val_fold),
        "test_hash": hash_dataframe(df_test_pred),
        "n_split": str(n_splits),
        "seed": str(seed),
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)  # default=str handles numpy types
    return hashlib.md5(key_str.encode()).hexdigest()


def apply_tabm(hparams, df_train, df_val, df_test_pred, feature_cols, target_col_name):
    all_run_dirs = glob(f'./runs/tabm/run*')
    hash = make_hash(hparams, df_train, df_val, df_test_pred, feature_cols, target_col_name)
    with open("./runs/tabm/map_run_key_to_num.json", "r") as f:
        map_hash_run = json.load(f)
        if hash in map_hash_run:
            print('Found hash!')
            run_dir = map_hash_run[hash]
            score_df = pd.read_csv(f'./runs/tabm/{run_dir}/score.csv')
            val_preds = np.load(f'./runs/tabm/{run_dir}/val_preds.npy')
            test_preds = np.load(f'./runs/tabm/{run_dir}/test_preds.npy')
            return score_df['Score'].values[0], val_preds, test_preds
        else:
            latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
            # map_hash_run[hash] = f'run_{latest_run + 1}'
    
    with open("./runs/tabm/map_run_key_to_num.json", "w") as f:
        f.write(json.dumps(map_hash_run, indent=4))  # indent=4 makes it human-readable
            
    

    # target_col: [55, 64]
    
    task_type = 'regression'
    # target_col_name = df_train.columns[target_col]
    
    X_train = df_train[feature_cols].values.astype(np.float32)
    Y_train = df_train[target_col_name].values.astype(np.float32)
    
    X_val = df_val[feature_cols].values.astype(np.float32)
    Y_val = df_val[target_col_name].values.astype(np.float32)
    
    X_test = df_test_pred[feature_cols].values.astype(np.float32)
    Y_test = df_test_pred[target_col_name].values.astype(np.float32)
    
    n_num_features = len(feature_cols)

    data_numpy = {
        'train': {'x_num': X_train, 'y': Y_train},
        'val': {'x_num': X_val, 'y': Y_val},
        'test': {'x_num': X_test, 'y': Y_test},
    }


    for part, part_data in data_numpy.items():
        for key, value in part_data.items():
            # print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')
            del key, value
        del part, part_data
        


    # Data Processing



    # Feature preprocessing.
    # NOTE
    # The choice between preprocessing strategies depends on a task and a model.

    # Simple preprocessing strategy.
    # preprocessing = sklearn.preprocessing.StandardScaler().fit(
    #     data_numpy['train']['x_num']
    # )

    # Advanced preprocessing strategy.
    # The noise is added to improve the output of QuantileTransformer in some cases.
    x_num_train_numpy = data_numpy['train']['x_num']
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, x_num_train_numpy.shape)
        .astype(x_num_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(df_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(x_num_train_numpy + noise)
    del x_num_train_numpy

    # Apply the preprocessing.
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
        
        # np.save(f'./dataset/{part}_qt.npy', data_numpy[part]['x_num'])


    # Label preprocessing.
    class RegressionLabelStats(NamedTuple):
        mean: float
        std: float


    Y_train = data_numpy['train']['y'].copy()
    if task_type == 'regression':
        # For regression tasks, it is highly recommended to standardize the training labels.
        regression_label_stats = RegressionLabelStats(
            Y_train.mean().item(), Y_train.std().item()
        )
        Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    else:
        regression_label_stats = None
        
        

    # Pytorch settings



    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train = Y_train.float()

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True can speed up training
    # of large enough models on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = False

    # fmt: off
    print(f'Device:        {device.type.upper()}')
    # print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
    # print(f'torch.compile: {compile_model}')
    # fmt: on





    # Model and optimizer

    # The best performance is usually achieved with `num_embeddings`
    # from the `rtdl_num_embeddings` package. Typically, `PiecewiseLinearEmbeddings`
    # and `PeriodicEmbeddings` perform best.

    embedding_type = hparams['embedding_type']
    # embedding_type = trial.suggest_categorical('embedding_type', ['PeriodicEmbeddings', 'PiecewiseLinearEmbeddings'])

    # Periodic embeddings.
    num_embeddings_periodic = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)


    n_bins = int(hparams['n_bins'])
    d_embedding = int(hparams['d_embedding'])
    
    # n_bins = trial.suggest_int('n_bins', 2, 128) # prev 48
    # d_embedding = trial.suggest_int('d_embedding', 8, 32, step=4) # prev 16
    
    # Piecewise-linear embeddings.
    num_embeddings_piecewise = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=n_bins),
        d_embedding=d_embedding,
        activation=False,
        version='B',
    )
    
    if embedding_type == 'PeriodicEmbeddings':
        num_embeddings = num_embeddings_periodic
    else:
        num_embeddings = num_embeddings_piecewise
        
    
    n_blocks = int(hparams['n_blocks'])
    d_block = int(hparams['d_block'])
    arch_type = hparams['arch_type']
    # n_blocks = trial.suggest_int("n_blocks", 1, 5)
    # d_block = trial.suggest_int("d_block", 64, 1296, step=16)
    # arch_type = trial.suggest_categorical('arch_type', ['tabm', 'tabm-mini'])
    
    n_classes = None
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=1 if n_classes is None else n_classes,
        num_embeddings=num_embeddings,
        n_blocks=n_blocks,
        d_block=d_block,
        k=int(hparams['k']),
        arch_type=arch_type,
    ).to(device)
    
    lr = float(hparams['lr'])
    weight_decay = float(hparams['weight_decay'])
    # lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gradient_clipping_norm: Optional[float] = 1.0

    if compile_model:
        # NOTE
        # `torch.compile(model, mode="reduce-overhead")` caused issues during training,
        # so the `mode` argument is not used.
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode
        



    # A quick reminder: TabM represents an ensemble of k MLPs.
    #
    # The option below determines if the MLPs are trained
    # on the same batches (share_training_batches=True) or
    # on different batches. Technically, this option determines:
    # - How the loss function is implemented.
    # - How the training batches are constructed.
    #
    # `True` is recommended by default because of better training efficiency.
    # On some tasks, `False` may provide better performance.
    
    
    share_training_batches = hparams['share_training_batches']
    # share_training_batches = trial.suggest_categorical('share_training_batches', ['T', 'F'])
    
    if share_training_batches == 'T':
        share_training_batches = True
    else:
        share_training_batches = False


    task_is_regression = task_type == 'regression'

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_num'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )


    base_loss_fn = (
        nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
    )


    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions. Each of them must be trained separately.

        # Regression:     (batch_size, k)            -> (batch_size * k,)
        # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
        y_pred = y_pred.flatten(0, 1)

        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            y_true = y_true.repeat_interleave(model.backbone.k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)

        return base_loss_fn(y_pred, y_true)


    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 32
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        if task_type == 'regression':
            # Transform the predictions back to the original label space.
            assert regression_label_stats is not None
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

        # Compute the mean of the k predictions.
        if not task_is_regression:
            # For classification, the mean must be computed in the probability space.
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)

        y_true = data[part]['y'].cpu().numpy()
        score = (
            (sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred))
            if task_type == 'regression'
            else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        )
        # score = (
        #     -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        #     if task_type == 'regression'
        #     else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        # )
        return float(100 - 90 * score / 2.72), y_pred  # The higher -- the better.


    # print(f'Test score before training: {evaluate("test")[0]:.4f}')








    n_epochs = 1_000_000_000
    train_size = df_train.shape[0]
    batch_size = 32
    epoch_size = math.ceil(train_size / batch_size)

    epoch = -1
    metrics = {'val': -math.inf, 'test': -math.inf}


    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }
        )


    best_checkpoint = make_checkpoint()

    # Early stopping: the training stops if the validation score
    # does not improve for more than `patience` consecutive epochs.
    patience = 64
    remaining_patience = patience

    for epoch in range(n_epochs):
        batches = (
            # Create one standard batch sequence.
            torch.randperm(train_size, device=device).split(batch_size)
            if share_training_batches
            # Create k independent batch sequences.
            else (
                torch.rand((train_size, model.backbone.k), device=device)
                .argsort(dim=0)
                .split(batch_size, dim=0)
            )
        )
        for batch_idx in batches:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()

        metrics = {part: evaluate(part)[0] for part in ['val', 'test']}
        val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

        # print(
        #     f'{"*" if val_score_improved else " "}'
        #     f' [epoch] {epoch:<3}'
        #     f' [val] {metrics["val"]:.3f}'
        #     f' [test] {metrics["test"]:.3f}'
        # )

        if val_score_improved:
            best_checkpoint = make_checkpoint()
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

    # To make final predictions, load the best checkpoint.
    model.load_state_dict(best_checkpoint['model'])

    val_score = best_checkpoint["metrics"]["val"]
    
    print('\n[Summary]')
    print(f'best epoch:  {best_checkpoint["epoch"]}')
    print(f'val score:  {best_checkpoint["metrics"]["val"]}')
    # print(f'test score: {best_checkpoint["metrics"]["test"]}')

    # df_submit = pd.read_csv('./from_BRACU_HPC/submission_autogluon_time_fraction_experimental_quality.csv')
    val_preds = evaluate('val')[1]
    test_preds = evaluate('test')[1]
    # df_submit[f'BlendProperty{target_col - 55 + 1}'] = test_preds
    # df_submit.to_csv(f'submission_tabm_test{target_col - 55 + 1}.csv', index=False)
    
    
    # trial.report(val_score, epoch)

    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()
    
    latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
    save_path = f'./runs/tabm/run_{latest_run + 1}'
    os.makedirs(save_path)
    pd.DataFrame([best_checkpoint["metrics"]["val"]], columns=['Score']).to_csv(f"{save_path}/score.csv")
    np.save(f'{save_path}/val_preds.npy', val_preds)
    np.save(f'{save_path}/test_preds.npy', test_preds)
    hparams['score'] = best_checkpoint["metrics"]["val"]
    hparams['target_col_name'] = target_col_name
    map_hash_run[hash] = f'run_{latest_run + 1}'

    with open(f'{save_path}/params.json', "w") as f:
        f.write(json.dumps(hparams, indent=4))  # indent=4 makes it human-readable
    
    return best_checkpoint["metrics"]["val"], val_preds, test_preds

def apply_tabm_cv(hparams, df_train, df_test_pred, feature_cols, col_name, seed=42, n_splits=5, callback=None):
    
    all_run_dirs = glob(f'./runs/tabm_cv/cv_run*')
    hash = make_hash(hparams, df_train, df_train, df_test_pred, feature_cols, col_name, seed=seed, n_splits=n_splits)
    with open("./runs/tabm_cv/map_run_key_to_num.json", "r") as f:
        map_hash_run = json.load(f)
        if hash in map_hash_run:
            print('Found hash!')
            run_dir = map_hash_run[hash]
            score_df = pd.read_csv(f'./runs/tabm_cv/{run_dir}/score.csv')
            df_oof_preds = np.load(f'./runs/tabm_cv/{run_dir}/df_oof_preds.npy')
            test_preds_avg = np.load(f'./runs/tabm_cv/{run_dir}/test_preds_avg.npy')
            return score_df['Score'].values[0], df_oof_preds, test_preds_avg
        else:
            latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
            # map_hash_run[hash] = f'cv_run_{latest_run + 1}'
    
    with open("./runs/tabm_cv/map_run_key_to_num.json", "w") as f:
        f.write(json.dumps(map_hash_run, indent=4))  # indent=4 makes it human-readable
        
        
        
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # col_name = f'BlendProperty{target_col - 55 + 1}'
    df_oof_preds = df_train[[col_name]].copy()
    
    test_preds_avg = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]
        
        score, val_preds, test_preds = apply_tabm(hparams, train_fold, val_fold, df_test_pred, feature_cols, col_name)
        df_oof_preds.loc[df_train.index[val_idx], col_name] = val_preds
        
        if test_preds_avg is None:
            test_preds_avg = test_preds
        else:
            test_preds_avg += test_preds
        if callback:
            callback(fold + 1, n_splits)
    
    test_preds_avg = test_preds_avg / n_splits
    
        # results_dir = f'./TabM/CV_b{target_col - 55 + 1}'
        # os.makedirs(results_dir, exist_ok=True)
        # np.save(f'{results_dir}/test_pred_fold_{fold}.npy', test_preds)
    
    mape = mean_absolute_percentage_error(df_train[col_name], df_oof_preds[col_name])
    score = 100 - 90 * mape / 2.72
    col_num = col_name[-1]
    if col_num == '0':
        col_num = 10
    os.makedirs(f'./Ensemble_tabm/{col_name}', exist_ok=True)
    np.save(f'./Ensemble_tabm/{col_name}/oof_b{col_num}_seed_{seed}_splits_{n_splits}_score_{score}.npy', df_oof_preds[col_name].values)
    
    latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
    save_path = f'./runs/tabm_cv/cv_run_{latest_run + 1}'
    os.makedirs(save_path)
    pd.DataFrame([score], columns=['Score']).to_csv(f"{save_path}/score.csv")
    np.save(f'{save_path}/df_oof_preds.npy', df_oof_preds[col_name].values)
    np.save(f'{save_path}/test_preds.npy', test_preds)
    
    hparams['score'] = score
    hparams['target_col_name'] = col_name
    hparams['seed'] = seed
    hparams['n_splits'] = n_splits
    map_hash_run[hash] = f'cv_run_{latest_run + 1}'
    
    with open(f'{save_path}/params.json', "w") as f:
        f.write(json.dumps(hparams, indent=4))  # indent=4 makes it human-readable
    
    return score, df_oof_preds, test_preds_avg


def apply_tabm_cv_tune(trial, df_train, df_test_pred, feature_cols, target_col, seed=42, n_splits=5, hparam_ranges={}):
        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    col_name = target_col
    df_oof_preds = df_train[[col_name]].copy()
    
    hparams = {}
    hparams['embedding_type'] = trial.suggest_categorical('embedding_type', hparam_ranges['embedding_type'])
    hparams['n_bins'] = trial.suggest_int('n_bins', hparam_ranges['n_bins'][0], hparam_ranges['n_bins'][1]) # prev 48
    hparams['d_embedding'] = trial.suggest_int('d_embedding', hparam_ranges['d_embedding'][0], hparam_ranges['d_embedding'][1], step=hparam_ranges['d_embedding'][2]) # prev 16
    hparams['n_blocks'] = trial.suggest_int("n_blocks", hparam_ranges['n_blocks'][0], hparam_ranges['n_blocks'][0])
    hparams['d_block'] = trial.suggest_int("d_block", hparam_ranges['d_block'][0], hparam_ranges['d_block'][1], step=hparam_ranges['d_block'][2])
    hparams['arch_type'] = trial.suggest_categorical('arch_type', hparam_ranges['arch_type'])
    hparams['lr'] = trial.suggest_float("lr", hparam_ranges['lr'][0], hparam_ranges['lr'][1], log=True)
    hparams['weight_decay'] = trial.suggest_float("weight_decay", hparam_ranges['weight_decay'][0], hparam_ranges['weight_decay'][1], log=True)
    hparams['share_training_batches'] = trial.suggest_categorical('share_training_batches', hparam_ranges['share_training_batches'])
    hparams['k'] = 32
    
    # hparams['embedding_type'] = trial.suggest_categorical('embedding_type', ['PeriodicEmbeddings', 'PiecewiseLinearEmbeddings'])
    # hparams['n_bins'] = trial.suggest_int('n_bins', 2, 128) # prev 48
    # hparams['d_embedding'] = trial.suggest_int('d_embedding', 8, 32, step=4) # prev 16
    # hparams['n_blocks'] = trial.suggest_int("n_blocks", 1, 4)
    # hparams['d_block'] = trial.suggest_int("d_block", 64, 1024, step=16)
    # hparams['arch_type'] = trial.suggest_categorical('arch_type', ['tabm', 'tabm-mini'])
    # hparams['lr'] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    # hparams['weight_decay'] = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    # hparams['share_training_batches'] = trial.suggest_categorical('share_training_batches', ['T', 'F'])
    # hparams['k'] = 32
    
    
    all_run_dirs = glob(f'./runs/tabm_cv/cv_run*')
    hash = make_hash(hparams, df_train, df_train, df_test_pred, feature_cols, target_col, seed=seed, n_splits=n_splits)
    with open("./runs/tabm_cv/map_run_key_to_num.json", "r") as f:
        map_hash_run = json.load(f)
        if hash in map_hash_run:
            print('Found hash!')
            run_dir = map_hash_run[hash]
            score_df = pd.read_csv(f'./runs/tabm_cv/{run_dir}/score.csv')
            df_oof_preds = np.load(f'./runs/tabm_cv/{run_dir}/df_oof_preds.npy')
            # test_preds_avg = np.load(f'./runs/tabm_cv/{run_dir}/test_preds_avg.npy')
            return score_df['Score'].values[0]
        else:
            latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
            map_hash_run[hash] = f'cv_run_{latest_run + 1}'
    
    with open("./runs/tabm_cv/map_run_key_to_num.json", "w") as f:
        f.write(json.dumps(map_hash_run, indent=4))  # indent=4 makes it human-readable
    
    

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]
        
        
        
        score, val_preds, test_preds = apply_tabm(hparams, train_fold, val_fold, df_test_pred, feature_cols, target_col)
        df_oof_preds.loc[df_train.index[val_idx], col_name] = val_preds
    
        # results_dir = f'./TabM/CV_b{target_col - 55 + 1}'
        # os.makedirs(results_dir, exist_ok=True)
        # np.save(f'{results_dir}/test_pred_fold_{fold}.npy', test_preds)
    
    mape = mean_absolute_percentage_error(df_train[col_name], df_oof_preds[col_name])
    score = 100 - 90 * mape / 2.72
    os.makedirs(f'./Ensemble_tabm/target_col/oof', exist_ok=True)
    np.save(f'./Ensemble_tabm/target_col/oof/target_col_trial_{trial.number}_{score}.npy', df_oof_preds[col_name].values)
    
    latest_run = max([int(run_dir.split('_')[-1]) for run_dir in all_run_dirs])
    save_path = f'./runs/tabm_cv/cv_run_{latest_run + 1}'
    os.makedirs(save_path)
    pd.DataFrame([score], columns=['Score']).to_csv(f"{save_path}/score.csv")
    np.save(f'{save_path}/df_oof_preds.npy', df_oof_preds[col_name].values)
    np.save(f'{save_path}/test_preds.npy', test_preds)
    
    hparams['score'] = score
    hparams['target_col_name'] = col_name
    hparams['seed'] = seed
    hparams['n_splits'] = n_splits
    
    with open(f'{save_path}/params.json', "w") as f:
        f.write(json.dumps(hparams, indent=4))  # indent=4 makes it human-readable
    
    return score
        
def apply_tabm_tune(trial, df_train, df_val, df_test_pred, feature_cols, target_col):

    # target_col: [55, 64]
    
    task_type = 'regression'
    target_col_name = target_col
    
    X_train = df_train[feature_cols].values.astype(np.float32)
    Y_train = df_train[target_col_name].values.astype(np.float32)
    
    X_val = df_val[feature_cols].values.astype(np.float32)
    Y_val = df_val[target_col_name].values.astype(np.float32)
    
    X_test = df_test_pred[feature_cols].values.astype(np.float32)
    Y_test = df_test_pred[target_col_name].values.astype(np.float32)
    
    n_num_features = len(feature_cols)

    data_numpy = {
        'train': {'x_num': X_train, 'y': Y_train},
        'val': {'x_num': X_val, 'y': Y_val},
        'test': {'x_num': X_test, 'y': Y_test},
    }


    for part, part_data in data_numpy.items():
        for key, value in part_data.items():
            # print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')
            del key, value
        del part, part_data
        


    # Data Processing



    # Feature preprocessing.
    # NOTE
    # The choice between preprocessing strategies depends on a task and a model.

    # Simple preprocessing strategy.
    # preprocessing = sklearn.preprocessing.StandardScaler().fit(
    #     data_numpy['train']['x_num']
    # )
    
    # Advanced preprocessing strategy.
    # The noise is added to improve the output of QuantileTransformer in some cases.
    x_num_train_numpy = data_numpy['train']['x_num']
    noise = (
        np.random.default_rng(0)
        .normal(0.0, 1e-5, x_num_train_numpy.shape)
        .astype(x_num_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(df_train.shape[0] // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
    ).fit(x_num_train_numpy + noise)
    del x_num_train_numpy

    # Apply the preprocessing.
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])


    # Label preprocessing.
    class RegressionLabelStats(NamedTuple):
        mean: float
        std: float


    Y_train = data_numpy['train']['y'].copy()
    if task_type == 'regression':
        # For regression tasks, it is highly recommended to standardize the training labels.
        regression_label_stats = RegressionLabelStats(
            Y_train.mean().item(), Y_train.std().item()
        )
        Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    else:
        regression_label_stats = None
        
        

    # Pytorch settings



    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train = Y_train.float()

    # Automatic mixed precision (AMP)
    # torch.float16 is implemented for completeness,
    # but it was not tested in the project,
    # so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True can speed up training
    # of large enough models on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = False

    # fmt: off
    print(f'Device:        {device.type.upper()}')
    # print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
    # print(f'torch.compile: {compile_model}')
    # fmt: on





    # Model and optimizer

    # The best performance is usually achieved with `num_embeddings`
    # from the `rtdl_num_embeddings` package. Typically, `PiecewiseLinearEmbeddings`
    # and `PeriodicEmbeddings` perform best.
    # d_block, n_block, lr, d_embedding, n_bins, weight_decay
    embedding_type = trial.suggest_categorical('embedding_type', ['PeriodicEmbeddings', 'PiecewiseLinearEmbeddings'])
    # embedding_type = 'PiecewiseLinearEmbeddings'
    # Periodic embeddings.
    num_embeddings_periodic = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)


    n_bins = trial.suggest_int('n_bins', 2, 128) # prev 48
    d_embedding = trial.suggest_int('d_embedding', 8, 32, step=4) # prev 16

    # Piecewise-linear embeddings.
    num_embeddings_piecewise = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=n_bins),
        d_embedding=d_embedding,
        activation=False,
        version='B',
    )
    
    if embedding_type == 'PeriodicEmbeddings':
        num_embeddings = num_embeddings_periodic
    else:
        num_embeddings = num_embeddings_piecewise
        
    #  d_block, n_block, lr, d_embedding, n_bins, weight_decay
    n_blocks = trial.suggest_int("n_blocks", 1, 4)
    d_block = trial.suggest_int("d_block", 64, 1024, step=16)
    arch_type = trial.suggest_categorical('arch_type', ['tabm', 'tabm-mini'])
    # arch_type = 'tabm'
    
    n_classes = None
    
    model = tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[],
        d_out=1 if n_classes is None else n_classes,
        num_embeddings=num_embeddings,
        n_blocks=n_blocks,
        d_block=d_block,
        arch_type=arch_type,
    ).to(device)
    
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gradient_clipping_norm: Optional[float] = 1.0

    if compile_model:
        # NOTE
        # `torch.compile(model, mode="reduce-overhead")` caused issues during training,
        # so the `mode` argument is not used.
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode
        



    # A quick reminder: TabM represents an ensemble of k MLPs.
    #
    # The option below determines if the MLPs are trained
    # on the same batches (share_training_batches=True) or
    # on different batches. Technically, this option determines:
    # - How the loss function is implemented.
    # - How the training batches are constructed.
    #
    # `True` is recommended by default because of better training efficiency.
    # On some tasks, `False` may provide better performance.
    
    
    share_training_batches_var = trial.suggest_categorical('share_training_batches', ['T', 'F'])
    # share_training_batches_var = 'T'
    if share_training_batches_var == 'T':
        share_training_batches = True
    else:
        share_training_batches = False


    task_is_regression = task_type == 'regression'

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_num'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )


    base_loss_fn = (
        nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
    )


    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions. Each of them must be trained separately.

        # Regression:     (batch_size, k)            -> (batch_size * k,)
        # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
        y_pred = y_pred.flatten(0, 1)

        if share_training_batches:
            # (batch_size,) -> (batch_size * k,)
            y_true = y_true.repeat_interleave(model.backbone.k)
        else:
            # (batch_size, k) -> (batch_size * k,)
            y_true = y_true.flatten(0, 1)

        return base_loss_fn(y_pred, y_true)


    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 32
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )
        if task_type == 'regression':
            # Transform the predictions back to the original label space.
            assert regression_label_stats is not None
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

        # Compute the mean of the k predictions.
        if not task_is_regression:
            # For classification, the mean must be computed in the probability space.
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)

        y_true = data[part]['y'].cpu().numpy()
        score = (
            (sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred))
            if task_type == 'regression'
            else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        )
        # score = (
        #     -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        #     if task_type == 'regression'
        #     else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        # )
        return float(100 - 90 * score / 2.72), y_pred  # The higher -- the better.


    # print(f'Test score before training: {evaluate("test")[0]:.4f}')








    n_epochs = 1_000_000_000
    train_size = df_train.shape[0]
    batch_size = 32
    epoch_size = math.ceil(train_size / batch_size)

    epoch = -1
    metrics = {'val': -math.inf, 'test': -math.inf}


    def make_checkpoint() -> dict[str, Any]:
        return deepcopy(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
            }
        )


    best_checkpoint = make_checkpoint()

    # Early stopping: the training stops if the validation score
    # does not improve for more than `patience` consecutive epochs.
    patience = 100
    remaining_patience = patience

    for epoch in range(n_epochs):
        batches = (
            # Create one standard batch sequence.
            torch.randperm(train_size, device=device).split(batch_size)
            if share_training_batches
            # Create k independent batch sequences.
            else (
                torch.rand((train_size, model.backbone.k), device=device)
                .argsort(dim=0)
                .split(batch_size, dim=0)
            )
        )
        for batch_idx in batches:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()

        metrics = {part: evaluate(part)[0] for part in ['val', 'test']}
        val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

        # print(
        #     f'{"*" if val_score_improved else " "}'
        #     f' [epoch] {epoch:<3}'
        #     f' [val] {metrics["val"]:.3f}'
        #     f' [test] {metrics["test"]:.3f}'
        # )

        if val_score_improved:
            best_checkpoint = make_checkpoint()
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

    # To make final predictions, load the best checkpoint.
    model.load_state_dict(best_checkpoint['model'])

    val_score = best_checkpoint["metrics"]["val"]
    
    print('\n[Summary]')
    print(f'best epoch:  {best_checkpoint["epoch"]}')
    print(f'val score:  {best_checkpoint["metrics"]["val"]}')
    # print(f'test score: {best_checkpoint["metrics"]["test"]}')

    # df_submit = pd.read_csv('./from_BRACU_HPC/submission_autogluon_time_fraction_experimental_quality.csv')
    val_preds = evaluate('val')[1]
    test_preds = evaluate('test')[1]
    # df_submit[f'BlendProperty{target_col - 55 + 1}'] = test_preds
    # df_submit.to_csv(f'submission_tabm_test{target_col - 55 + 1}.csv', index=False)
    
    
    # if val_score >= 94:
    os.makedirs(f'./Ensemble_tabm/target_col/Test', exist_ok=True)
    os.makedirs(f'./Ensemble_tabm/target_col/Validation', exist_ok=True)
    np.save(f'./Ensemble_tabm/target_col/Test/test_{target_col}_trial_{trial.number}_{val_score}.npy', test_preds)
    np.save(f'./Ensemble_tabm/target_col/Validation/val_{target_col}_trial_{trial.number}_{val_score}.npy', val_preds)
        
    
    trial.report(val_score, epoch)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return best_checkpoint["metrics"]["val"], val_preds, test_preds











