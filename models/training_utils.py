import os
import sys
import json
import yaml
import copy
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.mlp import MLP
from utils.decorr_utils import distance_corr, distance_corr_multi
from models import mlp_plotter


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, sample_weights, no_absolute_weights=None, disco_var=None):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.no_absolute_weights = no_absolute_weights
        self.disco_var = disco_var # 1D tensor aligned to X (e.g. mjj)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.disco_var is not None and self.no_absolute_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.no_absolute_weights[idx], self.disco_var[idx]
        elif self.disco_var is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.disco_var[idx]
        elif self.no_absolute_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.no_absolute_weights[idx]
        else:
            return self.X[idx], self.y[idx], self.sample_weights[idx]


def apply_disco(
        loss_nominal: torch.Tensor,
        y_pred: torch.Tensor,
        weights_batch: torch.Tensor,
        disco_var_batch: torch.Tensor,
        decorr_lambda: float,
        disco_signal_class_idx: Union[int, list[int], None],
        disco_reduce: str='mean'
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the DisCo (Decorrelation) loss to the nominal loss.
    """
    USE_MULTIDIM_DISCO = True # Experimental. Baseline only supports 1-dim decorrelation.

    norm_w = weights_batch / weights_batch.mean() # Normalize weights to mean 1
    # print(f"[DEBUG] weights_batch shape: {weights_batch.shape}")
    # print(f"[DEBUG] norm_w shape: {norm_w.shape}")
    # print(f"[DEBUG] weights_batch sum: {weights_batch.sum().item()}") # BUG: ~0.0014 for train, ~0.003 for val
    # print(f"[DEBUG] sum of norm_w: {norm_w.sum().item()}") # ~1024 for both train and val
    # print(f"[DEBUG] norm_w mean: {norm_w.mean().item()}, std: {norm_w.std().item()}")
    # print(f"[DEBUG] weights_batch mean: {weights_batch.mean().item()}, std: {weights_batch.std().item()}")
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        probs = F.softmax(y_pred, dim=1) # [N,C] # QUESTION: Can we get more effective DisCo without softmax?
        if USE_MULTIDIM_DISCO:
            if disco_signal_class_idx is None:
                raise ValueError("disco_signal_class_idx must be provided when use_disco=True.")
            if isinstance(disco_signal_class_idx, int):
                class_indices = [disco_signal_class_idx]
            elif isinstance(disco_signal_class_idx, list):
                class_indices = disco_signal_class_idx
            else:
                raise ValueError("disco_signal_class_idx must be int or list of int when y_pred is multi-dimensional.")
            d_corr = distance_corr_multi(disco_var_batch.reshape(-1), probs[:, class_indices], norm_w, reduce=disco_reduce)
        else:
            if isinstance(disco_signal_class_idx, int):
                class_indices = [disco_signal_class_idx]
            elif isinstance(disco_signal_class_idx, list):
                raise NotImplementedError("Multi-class DisCo with list of indices not implemented without USE_MULTIDIM_DISCO=True.")
            else:
                raise ValueError("disco_signal_class_idx must be int or list of int when y_pred is multi-dimensional.")
            d_corr = distance_corr_multi(disco_var_batch.reshape(-1), probs[:, class_indices], norm_w, reduce=disco_reduce)
    else:
        # Single output (binary classification)
        probs = torch.sigmoid(y_pred).reshape(-1) # [N]
        d_corr = distance_corr(disco_var_batch.reshape(-1), probs, norm_w.reshape(-1))
    total_weighted_loss = loss_nominal + decorr_lambda * d_corr

    return total_weighted_loss, d_corr


# Training and evaluation functions
def train_one_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        epoch: int,
        use_disco=False,
        decorr_lambda=0.1,
        disco_signal_class_idx: Union[int, list[int], None]=0,
        disco_reduce: str='mean'
) -> tuple[float, float, float, float, float, Union[float, None], Union[float, None]]:
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The neural network model to train.
        optimizer (optim.Optimizer): The optimizer for updating model parameters.
        data_loader (DataLoader): DataLoader providing training data batches.
        loss_fn (nn.Module): Loss function to compute the loss, like BCE. Don't include DisCo in loss_fn itself.
        device (torch.device): Device to run the training on (CPU or GPU).
        epoch (int): Current epoch number (for logging).
        use_disco (bool): Whether to include DisCo distance correlation in the loss.
        decorr_lambda (float): Weighting factor for the DisCo term in the loss. Typically between 0 and 1.
        disco_signal_class_idx (int | list[int] | None): Index/indices of class(es) in model output to use for DisCo decorrelation.
            If None, don't use DisCo even if use_disco=True.
            If int, use that one class index. (n.b. zero indexed)
            If list[int], use those class indices (e.g. for multi-class classification).
        disco_reduce (str): 'mean'|'sum'|'max'|'quadrature'|'none' to aggregate distance correlations across multiple classes.

    Returns:
        tuple: Average across batches for the epoch
            - Avg. training loss
            - Avg. accuracy
            - Avg. loss without absolute weights
            - Avg. loss without distance correlation
            - Avg. distance correlation
            - Avg. distance correlation times lambda
    """
    model.train()
    batch_losses = []
    batch_accs = []
    batch_losses_no_abs = []

    # Special for DisCo edits
    batch_losses_no_dist_corr = []
    batch_dist_corr = []
    batch_losses_no_abs_no_dist_corr = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    for batch in progress_bar:
        # Unpack depending on options
        if use_disco:
            # Expect dataset to include disco_var in batch
            if len(batch) == 5:
                X_batch, y_batch, weights_batch, weights_batch_no, disco_var_batch = batch
            else:
                raise ValueError("use_disco=True but dataset does not include disco_var. Check how the dataset was created.")
        else:
            # Not using DisCo
            if len(batch) == 4:
                X_batch, y_batch, weights_batch, weights_batch_no = batch
            else:
                # Not using weights without absolute value (not using no_absolute_weights)
                X_batch, y_batch, weights_batch = batch
                weights_batch_no = weights_batch # dummy assignment to avoid errors

        # Print first parts
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        weights_batch = weights_batch.to(device)
        weights_batch_no = weights_batch_no.to(device)
        if use_disco:
            disco_var_batch = disco_var_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch) # [N]
        wsum = weights_batch.sum()
        weighted_loss = (loss * weights_batch).sum() / wsum

        # Add DisCo term
        if use_disco:
            total_loss, d_corr = apply_disco(
                weighted_loss,
                y_pred,
                weights_batch,
                disco_var_batch,
                decorr_lambda,
                disco_signal_class_idx,
                disco_reduce
            )
            assert d_corr is not None, "d_corr should not be None when using DisCo. Something is wrong inside the apply_disco function."
        else:
            total_loss = weighted_loss
            d_corr = None # Dummy
        
        total_loss.backward()
        optimizer.step()

        weighted_loss_no_abs = (loss * weights_batch_no).sum() / weights_batch_no.sum()

        # compute weighted accuracy
        correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
        weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()

        batch_losses.append(            weighted_loss.item())
        batch_accs.append(              weighted_acc.item())
        batch_losses_no_abs.append(     weighted_loss_no_abs.item())
        batch_losses_no_dist_corr.append(weighted_loss.item() if not use_disco else (weighted_loss.item() - decorr_lambda * d_corr.item()))
        batch_dist_corr.append(         d_corr.item() if d_corr is not None else None)
        batch_losses_no_abs_no_dist_corr.append(weighted_loss_no_abs.item() if not use_disco else (weighted_loss_no_abs.item() - decorr_lambda * d_corr.item()))

        # Update progress bar
        if use_disco:
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item() - decorr_lambda * d_corr.item():.4f}+{decorr_lambda}*{d_corr.item():.4f}',
                'Acc': f'{weighted_acc.item():.4f}',
                'Loss_no_abs': f'{weighted_loss_no_abs.item():.4f}'
            })
        else:
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}',
                'Acc': f'{weighted_acc.item():.4f}',
                'Loss_no_abs': f'{weighted_loss_no_abs.item():.4f}'
            })

    mean_batch_losses                       = float(np.mean(batch_losses))
    mean_accs                               = float(np.mean(batch_accs))
    mean_batch_losses_no_abs                = float(np.mean(batch_losses_no_abs))
    mean_batch_losses_no_dist_corr          = float(np.mean(batch_losses_no_dist_corr))
    mean_batch_losses_no_abs_no_dist_corr   = float(np.mean(batch_losses_no_abs_no_dist_corr))
    mean_batch_dist_corr                    = float(np.mean([dc for dc in batch_dist_corr if dc is not None])) if use_disco else None
    mean_batch_dist_corr_times_lambda       = float(np.mean([dc * decorr_lambda for dc in batch_dist_corr if dc is not None])) if use_disco else None

    return (
        mean_batch_losses, 
        mean_accs,
        mean_batch_losses_no_abs,
        mean_batch_losses_no_dist_corr,
        mean_batch_losses_no_abs_no_dist_corr,
        mean_batch_dist_corr,
        mean_batch_dist_corr_times_lambda,
    )


def evaluate(
        model,
        data_loader,
        loss_fn,
        device,
        epoch,
        use_disco=False,
        decorr_lambda=0.1,
        disco_signal_class_idx:
        Union[int, list[int], None]=0, 
        disco_reduce: str='mean'
    ) -> tuple[float, float, float, Union[float, None], Union[float, None]]:
    model.eval()
    val_losses = []
    val_accs = []
    val_losses_no_dist_corr = []
    val_dist_corr = []
    val_dist_corr_times_lambda = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if use_disco:
                # Expect dataset to include disco_var in batch
                if len(batch) == 5:
                    X_batch, y_batch, weights_batch, weights_batch_no, disco_var_batch = batch
                elif len(batch) == 4:
                    X_batch, y_batch, weights_batch, disco_var_batch = batch
                else:
                    raise ValueError("use_disco=True but dataset does not include disco_var. Check how the dataset was created.")
            else:
                # Not using DisCo
                if len(batch) == 4:
                    X_batch, y_batch, weights_batch, weights_batch_no = batch
                else:
                    # Not using weights without absolute value (not using no_absolute_weights)
                    X_batch, y_batch, weights_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            weights_batch = weights_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()

            # Add DisCo term
            if use_disco:
                disco_var_batch = disco_var_batch.to(device)
                total_loss, d_corr = apply_disco(
                    weighted_loss,
                    y_pred,
                    weights_batch,
                    disco_var_batch,
                    decorr_lambda,
                    disco_signal_class_idx,
                    disco_reduce
                )
            else:
                total_loss = weighted_loss
                d_corr = None # Dummy

            # compute weighted accuracy
            correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()


            val_losses.append(                  total_loss.item())
            val_accs.append(                    weighted_acc.item())
            val_losses_no_dist_corr.append(     weighted_loss.item() if not use_disco else (weighted_loss.item() - decorr_lambda * d_corr.item()))
            val_dist_corr.append(               d_corr.item() if d_corr is not None else None)
            val_dist_corr_times_lambda.append(  d_corr.item() * decorr_lambda if d_corr is not None else None)

            # Update progress bar
            if use_disco:
                progress_bar.set_postfix({
                    'Loss': f'{weighted_loss.item() - decorr_lambda * d_corr.item():.4f}+{decorr_lambda}*{d_corr.item():.4f}',
                    'Acc': f'{weighted_acc.item():.4f}',
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{weighted_loss.item():.4f}',
                    'Acc': f'{weighted_acc.item():.4f}',
                })

        mean_losses                 = float(np.mean(val_losses))
        mean_accs                   = float(np.mean(val_accs))
        mean_losses_no_dist_corr    = float(np.mean(val_losses_no_dist_corr))
        mean_dist_corr              = float(np.mean(val_dist_corr)) if d_corr is not None else None
        mean_dist_corr_times_lambda = float(np.mean(val_dist_corr_times_lambda)) if d_corr is not None else None

    return (
        mean_losses, # Includes distance correlation if use_disco=True
        mean_accs,
        mean_losses_no_dist_corr,
        mean_dist_corr,
        mean_dist_corr_times_lambda
    )


# Save the best model
def save_checkpoint(**kwargs):
    """Save checkpoint to disk.

    Kwargs:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        scheduler (torch.optim.lr_scheduler): Scheduler state.
        lr_hist (list[float]): Learning rate history.
        disco_in_loss (bool): Whether DisCo was used in the loss function.
        file_path (str): Path to save the checkpoint.
        
        train_loss_hist (list[float]): Training loss history.
        train_loss_hist_no_absolute_weights (list[float]): Training loss history without absolute weights.
        train_loss_hist_no_dist_corr (list[float]): Training loss history without distance correlation.
        train_loss_hist_no_absolute_weights_no_dist_corr (list[float]): Training loss history without absolute weights and without distance correlation.
        train_dist_corr_hist (list[float]): Training distance correlation history.
         
        val_loss_hist (list[float]): Validation loss history.
        train_acc_hist (list[float]): Training accuracy history.
        val_acc_hist (list[float]): Validation accuracy history.

        best_weights (dict): Best model weights.
        best_loss (float): Best loss value.
        best_dist_corr (float): Best distance correlation value.

    Note: 
        Unless specified otherwise, losses include distance correlation if it was used during training.
        If distance correlation was not used, all losses are without it.
    
    """
    _epoch = kwargs.get('epoch')
    _model: torch.nn.Module | None              = kwargs.get('model', None)
    _optimizer: torch.optim.Optimizer | None    = kwargs.get('optimizer', None)
    _scheduler: ReduceLROnPlateau | None        = kwargs.get('scheduler', None)
    _lr_hist: list[float] | None                = kwargs.get('lr_hist', None)
    _disco_in_loss: bool | None                 = kwargs.get('disco_in_loss', None)
    _file_path: str | None                      = kwargs.get('file_path', None)

    assert _epoch is not None, "Epoch number must be provided"
    assert _model is not None, "Model must be provided"
    assert _optimizer is not None, "Optimizer must be provided"
    assert _scheduler is not None, "Scheduler must be provided"
    assert _lr_hist is not None, "Learning rate history must be provided"
    assert _disco_in_loss is not None, "Whether DisCo was used in loss must be provided as bool, not None"
    assert _file_path is not None, "File path for saving checkpoint must be provided"

    os.makedirs(os.path.dirname(_file_path), exist_ok=True)


    # TRAIN LOSS
    _train_loss_hist                        = kwargs.get('train_loss_hist', None) # loss used in training, with absolute weights
    _train_loss_hist_no_absolute_weights    = kwargs.get('train_loss_hist_no_absolute_weights', None) # loss used in training, without absolute weights
    _train_loss_hist_no_dist_corr           = kwargs.get('train_loss_hist_no_dist_corr', None) # no distcorr, regardless of DisCo usage
    _train_loss_hist_no_absolute_weights_no_dist_corr = kwargs.get('train_loss_hist_no_absolute_weights_no_dist_corr', None) # no distcorr, regardless of DisCo usage
    _train_dist_corr_hist                   = kwargs.get('train_dist_corr_hist', None) # distance correlation history during training

    assert _train_loss_hist is not None, "Training loss history must be provided"
    assert _train_loss_hist_no_absolute_weights is not None, "Training loss history without absolute weights must be provided"
    assert _train_loss_hist_no_dist_corr is not None, "Training loss history without distance correlation must be provided"
    assert _train_loss_hist_no_absolute_weights_no_dist_corr is not None, "Training loss history without absolute weights and without distance correlation must be provided"
    # _train_dist_corr_hist can be None if not using DisCo


    # VAL LOSS
    _val_loss_hist                  = kwargs.get('val_loss_hist', None) # used validation loss
    _val_loss_hist_no_dist_corr     = kwargs.get('val_loss_hist_no_dist_corr', None) # no distcorr, regardless of DisCo usage
    _val_dist_corr_hist             = kwargs.get('val_dist_corr_hist', None) # distance correlation history during validation

    assert _val_loss_hist is not None, "Validation loss history must be provided"
    assert _val_loss_hist_no_dist_corr is not None, "Validation loss history without distance correlation must be provided"
    # _val_dist_corr_hist can be None if not using DisCo


    # BEST
    _best_loss          = kwargs.get('best_loss', None) # includes/excludes distcorr based on training, nominally val loss
    _best_weights       = kwargs.get('best_weights', None)
    _best_dist_corr     = kwargs.get('best_dist_corr', None) # n.b. lowest = best

    assert _best_loss is not None, "Best loss must be provided"
    assert _best_weights is not None, "Best model weights must be provided"
    # _best_dist_corr can be None if not using DisCo


    # ACCURACY
    _train_acc_hist = kwargs.get('train_acc_hist', None)
    _val_acc_hist = kwargs.get('val_acc_hist', None)

    assert _train_acc_hist is not None, "Training accuracy history must be provided"
    assert _val_acc_hist is not None, "Validation accuracy history must be provided"

    # If using DisCo, check all related values are not None
    if _disco_in_loss:
        assert _train_dist_corr_hist is not None, "Training distance correlation history must be provided when using DisCo"
        assert _val_dist_corr_hist is not None, "Validation distance correlation history must be provided when using DisCo"
        assert _best_dist_corr is not None, "Best distance correlation must be provided when using DisCo"


    checkpoint = {
        'epoch':                    _epoch,
        'model_state_dict':         _model.state_dict(),
        'optimizer_state_dict':     _optimizer.state_dict(),
        'scheduler_state_dict':     _scheduler.state_dict(),
        'lr_hist':                  _lr_hist,
        'disco_in_loss':            _disco_in_loss,

        'train_loss_hist':                          _train_loss_hist,
        'train_loss_hist_no_absolute_weights':      _train_loss_hist_no_absolute_weights,
        'train_loss_hist_no_dist_corr':             _train_loss_hist_no_dist_corr,
        'train_loss_hist_no_absolute_weights_no_dist_corr': _train_loss_hist_no_absolute_weights_no_dist_corr,
        'train_dist_corr_hist':                      _train_dist_corr_hist,

        'val_loss_hist':                _val_loss_hist,
        'val_loss_hist_no_dist_corr':   _val_loss_hist_no_dist_corr,
        'val_dist_corr_hist':           _val_dist_corr_hist,

        'best_weights':                 _best_weights,
        'best_loss':                    _best_loss,
        'best_dist_corr':               _best_dist_corr,

        'train_acc_hist':               _train_acc_hist,
        'val_acc_hist':                 _val_acc_hist,
    }
    torch.save(checkpoint, _file_path)
    print(f'Checkpoint saved to {_file_path}')


def _validate_training_config(training_config: dict):
    required_keys = [
        "random_seed",
        "weight_scheme",
        "cuda_device"
    ]
    for key in required_keys:
        if key not in training_config:
            raise ValueError(f"Missing required key '{key}' in training configuration.")
    
    # Multi-dependent checks
    if training_config.get("use_DisCo", False):
        if training_config.get("decorr_lambda") is None:
            raise ValueError("Missing key in training_config.yaml. decorr_lambda must be specified when use_DisCo is True.")
        if training_config.get("disco_reduce_method") is None:
            raise ValueError("Missing key in training_config.yaml. disco_reduce_method must be specified when use_DisCo is True.")
        if training_config.get("disco_decorr_class_idx") is None:
            raise ValueError("Missing key in training_config.yaml. disco_decorr_class_idx must be specified when use_DisCo is True.")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--input_path', type=str, help='Path to the input files')
    parser.add_argument('--training_config_path', type=str, default=10, help='Training configuration path')
    parser.add_argument('--job_config_path', type=str, default="", help='Job configuration path')
    #parser.add_argument('', type=str, help='Path to the best parameters')
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)
    with open(f"{args.job_config_path}", 'r') as f:
        job_config = yaml.safe_load(f)

    _validate_training_config(training_config)

    # TODO: Add config file archiver

    seed:                   int             = training_config["random_seed"]
    weight_scheme:          str             = training_config["weight_scheme"]
    max_epoch:              int             = training_config.get("max_epoch", 500)
    use_disco:              bool            = training_config.get("use_DisCo", False)
    decorr_lambda:          float           = training_config.get("decorr_lambda", 0.1)
    disco_reduce_method:    str             = training_config.get("decorr_reduce_method", "mean")
    disco_decorr_class_idx: list[int] | int = training_config.get("disco_decorr_class_idx", 0) # 0 for signal, [0,1,2,3] for all classes, etc.

    # --- REPROD SETUP ---
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
    print('\n', 'INFO: Used device is', device, '\n')

    input_path = args.input_path
    if not os.path.exists(f'{input_path}/random_search_1/best_params.json'):
        print("INFO: No random search done for performing training")
        print("Using predefined parameters which are saved in the folder")

        os.makedirs(f'{input_path}/random_search_1', exist_ok=True)
        # best_params = {"num_layers": 3, "num_nodes": 100, "act_fn_name": "ELU", "lr": 2.027496582741043e-05, "weight_decay": 5.159904717896079e-05, "dropout_prob": 0.05, "n_trials": 0}
        best_params = {"num_layers": 5, "num_nodes": 1024, "act_fn_name": "ELU", "lr": 2.027496582741043e-05, "weight_decay": 5.159904717896079e-05, "dropout_prob": 0.25, "n_trials": 0}
        with open(f'{input_path}/random_search_1/best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params, f)

    best_params_path = f'{input_path}/random_search_1/best_params.json'
    path_to_checkpoint = f'{input_path}/after_random_search_best1'
    os.makedirs(path_to_checkpoint, exist_ok=True)

    # Load data
    X_train = np.load(f'{input_path}/X_train.npy')
    X_val = np.load(f'{input_path}/X_val.npy')
    y_train = np.load(f'{input_path}/y_train.npy')
    y_val = np.load(f'{input_path}/y_val.npy')

    # set weight scheme for training
    if weight_scheme == "weighted_abs":
        class_weights_for_training = np.load(f'{input_path}/class_weights_for_training_abs.npy')
        class_weights_for_val = np.load(f'{input_path}/class_weights_for_val_abs.npy') # WITH absolute value

    elif weight_scheme == "weighted_only_positive":
        class_weights_for_training = np.load(f'{input_path}/class_weights_only_positive.npy')
        class_weights_for_val = np.load(f'{input_path}/class_weights_for_val_only_positive.npy')

    elif weight_scheme == "weighted_CRUW_abs":
        # to be completed
        pass

    elif weight_scheme == "weighted_CRUW_only_positive":
        # to be completed
        pass

    # class_weights_for_{training,val} should be directly comparable

    class_weights_for_train_no_absolute = np.load(f'{input_path}/true_class_weights.npy')
    class_weights_for_val_no_absolute = np.load(f'{input_path}/class_weights_for_val.npy') # WITHOUT absolute value

    # Convert targets to class indices (if one-hot encoded)
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32)
    class_weights_for_train_no_absolute = torch.tensor(class_weights_for_train_no_absolute, dtype=torch.float32)
    class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32)
    class_weights_for_val_no_absolute = torch.tensor(class_weights_for_val_no_absolute, dtype=torch.float32)

    # Load input features (optional)
    with open(f'{input_path}/input_vars.txt', 'r') as f:
        input_vars = json.load(f)
    print('INFO: Input features are', input_vars, '\n')

    # Load best parameters
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    print("Parameters: ", best_params, '\n')

    # Save the parameters in the training folder
    with open(f"{path_to_checkpoint}/params.json", 'w') as f:
        json.dump(best_params, f)

    # Model parameters
    best_num_layers = best_params['num_layers']
    best_num_nodes = best_params['num_nodes']
    best_act_fn_name = best_params['act_fn_name']
    best_act_fn = getattr(nn, best_act_fn_name)
    best_lr = best_params['lr']
    best_weight_decay = best_params['weight_decay']
    best_dropout_prob = best_params['dropout_prob']
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))  # Number of classes

    # Load DisCo variable if needed
    if use_disco:
        z_train = np.load(f'{input_path}/z_train.npy')  # e.g., mjj for decorrelation
        z_val = np.load(f'{input_path}/z_val.npy')
        disco_var_train = torch.tensor(z_train, dtype=torch.float32)
        disco_var_val = torch.tensor(z_val, dtype=torch.float32)
    else:
        disco_var_train, disco_var_val = None, None

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, class_weights_for_training, class_weights_for_train_no_absolute, disco_var=disco_var_train)
    val_dataset = CustomDataset(X_val, y_val, class_weights_for_val, class_weights_for_val_no_absolute, disco_var=disco_var_val)

    # Create data loaders
    g = torch.Generator().manual_seed(seed)
    batch_size = 1024 #16384 # 8192 # 32768 # 1024 #16384
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Debugging sum of weights per batch issue
    all_weights_batch_train = np.array([])
    all_weights_batch_train_no_abs = np.array([])
    all_weights_batch_val = np.array([])
    all_weights_batch_val_no_abs = np.array([])
    for idx, (X_batch, y_batch, weights_batch, weights_batch_no_abs, disco_var_batch) in enumerate(train_loader):
        # print(idx)
        # print(f"[DEBUG] weights_batch sum (from dataset): {weights_batch.sum().item()}")
        # print(f"type(weights_batch): {type(weights_batch)}")
        # print(f"type(weights_batch.numpy()): {type(weights_batch.numpy())}")
        # print(f"weights_batch.shape: {weights_batch.shape}")
        # print(f"weights_batch.numpy().shape: {weights_batch.numpy().shape}")
        # print(f"weights_batch.numpy(): {weights_batch.numpy()}")
        # print(f"weights_batch.sum().item(): {weights_batch.sum().item()}")
        all_weights_batch_train = np.append(all_weights_batch_train, weights_batch.numpy())
        all_weights_batch_train_no_abs = np.append(all_weights_batch_train_no_abs, weights_batch_no_abs.numpy())
        if idx >= 1000:
            print("break after 1000 batches")
            break
    for idx, (X_batch, y_batch, weights_batch, weights_batch_no_abs, disco_var_batch) in enumerate(val_loader):
        # print(idx)
        # print(f"[DEBUG] weights_batch sum (from dataset - val): {weights_batch.sum().item()}")
        all_weights_batch_val = np.append(all_weights_batch_val, weights_batch.numpy())
        all_weights_batch_val_no_abs = np.append(all_weights_batch_val_no_abs, weights_batch_no_abs.numpy())
        if idx >= 1000:
            print("break after 1000 batches")
            break

    # delta = 0.0005
    # for item in all_weights_batch_train:
    #     if item 

    print(f"Total number of weights in train batches (first 10 batches): {len(all_weights_batch_train)}")

    # Manually bin
    nbins = 100
    min_weight = -0.000100
    max_weight =  0.000100
    edges = np.linspace(min_weight, max_weight, nbins + 1)
    hist_train, _ = np.histogram(all_weights_batch_train, bins=edges)
    hist_val, _ = np.histogram(all_weights_batch_val, bins=edges)
    hist_train_no_abs, _ = np.histogram(all_weights_batch_train_no_abs, bins=edges)
    hist_val_no_abs, _ = np.histogram(all_weights_batch_val_no_abs, bins=edges)

    centers = (edges[:-1] + edges[1:]) / 2.0

    # Statistical (Poisson) uncertainty: sigma = sqrt(N)
    eps = 1e-1  # small floor to allow plotting on log scale
    y_train = hist_train.astype(float) + eps
    y_val = hist_val.astype(float) + eps
    y_train_no_abs = hist_train_no_abs.astype(float) + eps
    y_val_no_abs = hist_val_no_abs.astype(float) + eps

    err_train = np.sqrt(hist_train).astype(float)
    err_val = np.sqrt(hist_val).astype(float)
    err_train_no_abs = np.sqrt(hist_train_no_abs).astype(float)
    err_val_no_abs = np.sqrt(hist_val_no_abs).astype(float)

    # Ensure a minimum error for zero-count bins so they are visible on log scale
    err_train[err_train == 0] = eps
    err_val[err_val == 0] = eps
    err_train_no_abs[err_train_no_abs == 0] = eps
    err_val_no_abs[err_val_no_abs == 0] = eps

    plt.figure(figsize=(10,6))
    plt.step(centers, y_train, where='mid', label='Train', color='blue', alpha=0.7)
    plt.errorbar(centers, y_train, yerr=err_train, fmt='none', ecolor='blue', alpha=0.6, capsize=2)

    plt.step(centers, y_val, where='mid', label='Validation', color='orange', alpha=0.7)
    plt.errorbar(centers, y_val, yerr=err_val, fmt='none', ecolor='orange', alpha=0.6, capsize=2)

    plt.step(centers, y_train_no_abs, where='mid', label='Train (No Abs)', color='green', alpha=0.7)
    plt.errorbar(centers, y_train_no_abs, yerr=err_train_no_abs, fmt='none', ecolor='green', alpha=0.6, capsize=2)

    plt.step(centers, y_val_no_abs, where='mid', label='Validation (No Abs)', color='red', alpha=0.7)
    plt.errorbar(centers, y_val_no_abs, yerr=err_val_no_abs, fmt='none', ecolor='red', alpha=0.6, capsize=2)

    plt.xlabel('Weight value', loc="center")
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights per Sample')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{path_to_checkpoint}/weights_per_sample_distribution.png')
    plt.close()


    # Define model, loss function, optimizer, and scheduler
    best_model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    best_scheduler = ReduceLROnPlateau(best_optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    # Training loop parameters
    print(f"INFO: Training for {max_epoch} epochs", '\n')
    best_loss = np.inf
    best_dist_corr = np.inf # n.b. should use raw dist corr, not weighted by lambda
    best_weights = None
    patience = 50
    counter = 0

    train_loss_hist,                    val_loss_hist                   = [], []
    train_loss_hist_no_absolute_weights                                 = []
    train_acc_hist,                     val_acc_hist                    = [], []
    lr_hist                                                             = []
    train_loss_hist_no_dist_corr,       val_loss_hist_no_dist_corr      = [], []
    train_dist_corr_hist,               val_dist_corr_hist              = [], []
    train_dist_corr_times_lambda_hist,  val_dist_corr_times_lambda_hist = [], []

    # Training loop
    for epoch in range(max_epoch):
        # Training
        packed_metrics = train_one_epoch(
            best_model,
            best_optimizer,
            train_loader,
            loss_fn,
            device,
            epoch,
            use_disco=use_disco,
            decorr_lambda=decorr_lambda,
            disco_signal_class_idx=disco_decorr_class_idx,
            disco_reduce=disco_reduce_method
        )

        train_loss, train_acc, train_loss_no_absolute, train_loss_no_dist_corr, train_loss_no_abs_no_dist_corr, train_dist_corr, train_dist_corr_times_lambda = packed_metrics

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        train_loss_hist_no_absolute_weights.append(train_loss_no_absolute)
        train_loss_hist_no_dist_corr.append(train_loss_no_dist_corr)
        assert train_loss_no_dist_corr is not None, "train_loss_no_dist_corr should not be None, even if not using DisCo"
        if use_disco:
            train_dist_corr_hist.append(train_dist_corr)
            train_dist_corr_times_lambda_hist.append(train_dist_corr_times_lambda)
            assert train_dist_corr is not None, "train_dist_corr should not be None when using DisCo"
            assert decorr_lambda is not None, "decorr_lambda should not be None when using DisCo"


        # Validation
        val_loss, val_acc, val_loss_no_dist_corr, val_dist_corr, val_dist_corr_times_lambda = evaluate(
            best_model,
            val_loader,
            loss_fn,
            device,
            epoch,
            use_disco=use_disco,
            decorr_lambda=decorr_lambda,
            disco_signal_class_idx=disco_decorr_class_idx,
            disco_reduce=disco_reduce_method
        )
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_loss_hist_no_dist_corr.append(val_loss_no_dist_corr)
        assert val_loss_no_dist_corr is not None, "val_loss_no_dist_corr should not be None, even if not using DisCo"
        if use_disco:
            val_dist_corr_hist.append(val_dist_corr)
            val_dist_corr_times_lambda_hist.append(val_dist_corr_times_lambda)
            assert val_dist_corr is not None, "val_dist_corr should not be None when using DisCo"

        # Scheduler step
        best_scheduler.step(val_loss)
        current_lr = best_optimizer.param_groups[0]['lr']
        lr_hist.append(current_lr)
        print('-'*80)
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")

        if val_dist_corr is not None and val_dist_corr < best_dist_corr:
            best_dist_corr = val_dist_corr

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(best_model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"Counter: {counter}")
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch {epoch} - Train Loss no abs: {train_loss_no_absolute:.4f}")
        if use_disco:
            print(f"Epoch {epoch} - Train Dist Corr: {train_dist_corr:.4f}, Val Dist Corr: {val_dist_corr:.4f}")
            print(f"Epoch {epoch} - (no dist corr) Train Loss: {train_loss_no_dist_corr:.4f}, Val Loss: {val_loss_no_dist_corr:.4f}")
            print(f"Epoch {epoch} - Train Loss = {train_loss_no_dist_corr:.4f} + {decorr_lambda} * {train_dist_corr:.4f}")
        
        # Save checkpoint every ___ epochs
        if epoch % 10 == 0:
            # WARNING: Be mindful of disk space when saving frequent checkpoints
            # that include the full suite of y prediction arrays for ROC plots.
            # 300 epochs x ~100 MB = ~30 GB   <-- That's not insignificant!
            # 300 epochs x ~100 MB x 10% of epochs = ~3 GB   <-- More reasonable.
            print(f"Saving checkpoint at epoch {epoch}")
            save_checkpoint(
                epoch=epoch,
                model=best_model,
                optimizer=best_optimizer,
                scheduler=best_scheduler,
                file_path=f"{path_to_checkpoint}/checkpoints/epoch{epoch}/mlp.pth",
                lr_hist=lr_hist,
                disco_in_loss=use_disco,

                train_loss_hist=train_loss_hist,
                train_loss_hist_no_absolute_weights=train_loss_hist_no_absolute_weights,
                train_loss_hist_no_absolute_weights_no_dist_corr=train_loss_no_abs_no_dist_corr,
                train_acc_hist=train_acc_hist,
                train_loss_hist_no_dist_corr=train_loss_hist_no_dist_corr,
                train_dist_corr_hist=train_dist_corr_hist,

                val_loss_hist=val_loss_hist,
                val_acc_hist=val_acc_hist,
                val_loss_hist_no_dist_corr=val_loss_hist_no_dist_corr,
                val_dist_corr_hist=val_dist_corr_hist,

                best_weights=best_weights,
                best_loss=best_loss,
                best_dist_corr=best_dist_corr,
            )
            # TODO: Numpy save y_pred_train.npy, y_train.npy, y_pred_val.npy, and y_val.npy for checkpoint ROC plots
            mlp_plotter.run_condor_job(
                input_path=input_path,                                                    # base path to input files
                condor_dir=f"{path_to_checkpoint}/condor/mlp_plotter/",                   # condor directory
                plot_dir=f"{path_to_checkpoint}/checkpoints/epoch{epoch}/plots/",         # output directory for plots
                checkpoint_file=f"{path_to_checkpoint}/checkpoints/epoch{epoch}/mlp.pth", # checkpoint path
                job_config=job_config,
                dry_run=False,
                epoch=epoch,
            )
        print()




    save_checkpoint(
        epoch=epoch,
        model=best_model,
        optimizer=best_optimizer,
        scheduler=best_scheduler,
        file_path=f"{path_to_checkpoint}/mlp.pth",
        lr_hist=lr_hist,
        disco_in_loss=use_disco,

        train_loss_hist=train_loss_hist,
        train_loss_hist_no_absolute_weights=train_loss_hist_no_absolute_weights,
        train_loss_hist_no_absolute_weights_no_dist_corr=train_loss_no_abs_no_dist_corr,
        train_acc_hist=train_acc_hist,
        train_loss_hist_no_dist_corr=train_loss_hist_no_dist_corr,
        train_dist_corr_hist=train_dist_corr_hist,

        val_loss_hist=val_loss_hist,
        val_acc_hist=val_acc_hist,
        val_loss_hist_no_dist_corr=val_loss_hist_no_dist_corr,
        val_dist_corr_hist=val_dist_corr_hist,

        best_weights=best_weights,
        best_loss=best_loss,
        best_dist_corr=best_dist_corr,
        )

    # Load the best state of the model
    best_model.load_state_dict(best_weights)

    # Save predictions (optional)
    best_model.eval()
    batch_size = 1024
    y_pred_train_probs = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = best_model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_pred_train_probs.append(y_batch.cpu().numpy())
    y_pred_train_probs = np.concatenate(y_pred_train_probs, axis=0)

    y_pred_val_probs = []
    for i in range(0, len(X_val), batch_size):
        X_batch = X_val[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = best_model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_pred_val_probs.append(y_batch.cpu().numpy())
    y_pred_val_probs = np.concatenate(y_pred_val_probs, axis=0)

    # best_model.eval()
    # with torch.no_grad():
    #     y_pred_train = best_model(X_train.to(device))
    #     y_pred_val = best_model(X_val.to(device))

    # y_pred_train_probs = F.softmax(y_pred_train, dim=1)
    #y_pred_train_np = y_pred_train_probs.cpu().detach().numpy()
    y_pred_train_np = y_pred_train_probs
    y_train_np = y_train.cpu().numpy()

    # y_pred_val_probs = F.softmax(y_pred_val, dim=1)
    #y_pred_val_np = y_pred_val_probs.cpu().detach().numpy()
    y_pred_val_np = y_pred_val_probs
    y_val_np = y_val.cpu().numpy()

    # Save predictions
    np.save(f"{path_to_checkpoint}/y_pred_train.npy", y_pred_train_np)
    np.save(f"{path_to_checkpoint}/y_train.npy", y_train_np)
    np.save(f"{path_to_checkpoint}/y_pred_val.npy", y_pred_val_np)
    np.save(f"{path_to_checkpoint}/y_val.npy", y_val_np)