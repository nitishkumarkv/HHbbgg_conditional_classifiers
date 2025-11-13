import copy
import json
import os
import sys
from typing import Union, Optional, Literal
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models import mlp_plotter
from models.mlp import MLP
from utils.decorr_utils import distance_corr, reduce_disco_scores
from utils.predictions import save_predictions


def sigmoid(x: Tensor, x0: Tensor, k: Tensor) -> Tensor:
    """Elementwise generalized sigmoid with per-class midpoints/steepness.

    Args:
        x (Tensor): [N,C] values to transform (typically class probabilities).
        x0 (Tensor): [C,] midpoint for each class where sigmoid = 0.5.
        k (Tensor): [C,] steepness for each class (sign controls monotonicity).

    Returns:
        Tensor: [N,C] values in (0,1).
    """

    return torch.sigmoid(k * (x - x0))


def generalized_sigmoid(x: Tensor, x0: Tensor, k: Tensor, y_min: float, y_max: float) -> Tensor:
    """Generalized sigmoid mapping to [y_min, y_max] for each class."""
    return y_min + (y_max - y_min) * sigmoid(x, x0, k)

def generalized_5090_sigmoid(
        x: Tensor,
        x0: Tensor,
        x90: Tensor,
        y_min: float,
        y_max: float,
    ) -> Tensor:
    """Sigmoid per class hitting 50% at x0 and 90% at x90."""
    if torch.any((x90 - x0).abs() < 1e-6).item():
        raise ValueError("score_midpoint and score_90_percent must differ for each class.")

    log_nine = torch.log(torch.tensor(9.0, device=x.device, dtype=x.dtype))
    k: Tensor = log_nine / (x90 - x0)
    return generalized_sigmoid(x, x0, k, y_min, y_max)


def sigmoid_upweight(
        y_pred: Tensor,
        x0: Tensor,
        x90: Tensor,
        y_min: float,
        y_max: float,
    ) -> Tensor:
    """SR upweighting based on class scores.

    Produces a per-event multiplier in [y_min, y_max] that approaches y_max
    only when all classes satisfy their SR criteria: signal scores high (x > x0)
    and background scores low (x < x0).
    """
    probs = F.softmax(y_pred, dim=1, dtype=torch.float32)
    per_class_focus = generalized_5090_sigmoid(
        probs,
        x0=x0.to(device=y_pred.device, dtype=probs.dtype),
        x90=x90.to(device=y_pred.device, dtype=probs.dtype),
        y_min=0.0,
        y_max=1.0,
    )  # [N,C]
    sr_focus = torch.prod(per_class_focus, dim=1)  # [N]
    return y_min + (y_max - y_min) * sr_focus


def _extract_ordered_values(obj) -> list[float]:
    """Return a list preserving the original ordering for dicts/Series."""
    values_attr = getattr(obj, "values", None)
    if callable(values_attr):
        iterable = values_attr()
    elif values_attr is not None:
        iterable = values_attr
    else:
        iterable = obj
    return list(iterable)




# @dataclass
# class Weights():
#     """Highly efficient, flexible weight storage class to wrap around different weight types
#     while minimizing memory usage and data transfers. This class allows convenient storage and access
#     of various modifications to event weights used during training and evaluation, like upweight factors.

#     weights = Weights(weights_for_training)

#     """
#     base: torch.Tensor
#     modifiers: dict[str, torch.Tensor] = field(default_factory=dict)
#     enabled: dict[str, bool] = field(default_factory=dict)

#     # unfinished


    



# def _validate_upweight_params(config: dict):
#     """Validate upweight params once instead of inside the SR_upweight function every call."""
#     if "upweight_params" not in config:
#         raise ValueError("'upweight_params' not found in training_config. This field is necessary for SR upweighting.")
    
#     params = config["upweight_params"]
#     required_keys = ["max_upweight", ""]

#     pass

# def SR_upweight(y_pred: torch.Tensor, max_upweight: float, min_upweight: float, ) -> torch.Tensor:
#     """Apply signal region upweighting to model predictions.

#     Args:
#         y_pred (torch.Tensor): Model predictions (logits or probabilities).
#         training_config (dict): Training configuration dictionary.

#     Returns:
#         torch.Tensor: Upweighted model predictions.
#     """
#     params = training_config["upweight_params"]
#     max_upweight = params["max_upweight"]
#     min_upweight = params.get("min_upweight", 1.0)
#     sr_def = {}




# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, sample_weights, no_absolute_weights=None, disco_vars=None):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.no_absolute_weights = no_absolute_weights
        self.disco_vars = disco_vars # 2D tensor aligned to X (e.g. mjj, mgg)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.disco_vars is not None and self.no_absolute_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.no_absolute_weights[idx], self.disco_vars[idx]
        elif self.disco_vars is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.disco_vars[idx]
        elif self.no_absolute_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.no_absolute_weights[idx]
        else:
            return self.X[idx], self.y[idx], self.sample_weights[idx]


def _plot_norm_weights_for_disco(weights: Tensor, title="Distribution of Normalized Weights used in DisCo Calculation",fname="norm_weights_disco_0.png"):
    """This is a diagnostic plot. Do not call it every batch during training apart from debugging efforts."""
    # fname = "norm_weights_disco_0.png"
    n = 0
    while os.path.exists(fname):
        n += 1
        fname = f"norm_weights_disco_{n}.png"

    plt.figure(figsize=(8,6))
    plt.hist(weights.cpu().numpy(), bins=100, histtype='stepfilled', alpha=0.7)
    plt.xlabel("Normalized Weights for DisCo", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.savefig(fname)
    plt.close()


def _get_bkg_mask(y_true: Tensor) -> Tensor:
    """Get boolean mask for background events only, assuming y_true is class indices.
    0 -> nonRes_score   (bkg)
    1 -> ttH_score      (bkg)
    2 -> singleH_score  (bkg)
    3 -> HH_score       (sig)
    """
    return (y_true == 0) | (y_true == 1) | (y_true == 2)


def apply_disco(
        loss_nominal: Tensor,
        y_pred: Tensor,
        weights_batch: Tensor,
        disco_vars_batch: Tensor,
        decorr_lambda: float,
        disco_signal_class_idx: Union[int, list[int], None],
        y_true: Union[Tensor, None]=None,
        disco_reduce: str='mean'
    ) -> tuple[Tensor, Tensor]:
    """
    Apply the DisCo (Decorrelation) loss to the nominal loss.

    N -> Number of events per batch
    C -> Number of classes (nominally 4)
    """
    USE_BKG_MASK = True # Only bkg MC events contribute to DisCo score

    if USE_BKG_MASK and y_true is None:
        raise ValueError(
            "Must provide target truths (y_true) when using DisCo background-only mask (USE_BKG_MASK = True). "
            + "DisCo loss component only counts background MC events when background-only mask is enabled."
        )

    if USE_BKG_MASK and y_true is not None:
        bkg_mask = _get_bkg_mask(y_true) # [N,] boolean tensor
    else:
        bkg_mask = torch.ones_like(disco_vars_batch[:, 0], dtype=torch.bool) # [N,] all True

    disco_vars_batch = disco_vars_batch[bkg_mask]
    weights_batch = weights_batch[bkg_mask]

    if disco_vars_batch.dim() == 1:
        raise ValueError("disco_vars_batch must have at least two columns to compute distance correlation.")

    n_vars = disco_vars_batch.shape[1]
    if n_vars < 2:
        raise ValueError("At least two decorrelation variables are required to compute DisCo.")

    norm_w = weights_batch / (weights_batch.mean() + 1e-12) # Normalize weights to mean 1

    pairwise_corrs: list[Tensor] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pairwise_corrs.append(
                distance_corr(
                    disco_vars_batch[:, i],
                    disco_vars_batch[:, j],
                    norm_w.reshape(-1),
                )
            )

    if not pairwise_corrs:
        raise ValueError("DisCo requires at least one pair of variables to decorrelate.")

    d_corr = reduce_disco_scores(torch.stack(pairwise_corrs), disco_reduce)

    # print(f"d_corr = {d_corr.item():.6f}")
    # print(f"d_corr.shape = {d_corr.shape}")
    # print(f"type(d_corr) = {type(d_corr)}")
    # print(f"d_corr_vec = {d_corr_vec.detach().cpu().numpy()}")
    # print(f"d_corr_vec.shape = {d_corr_vec.shape}")
    # print(f"loss_nominal = {loss_nominal.item():.6f}")
    # print(f"loss_nominal.shape = {loss_nominal.shape}")
    # print(f"type(loss_nominal) = {type(loss_nominal)}")

    total_weighted_loss = loss_nominal + decorr_lambda * d_corr

    # print(f"total_weighted_loss = {total_weighted_loss.item():.6f}")
    # print(f"total_weighted_loss.shape = {total_weighted_loss.shape}")
    # print(f"type(total_weighted_loss) = {type(total_weighted_loss)}")

    # print(f"[DEBUG] Multidim DisCo scores: {d_corr_vec.item()}") # COMMENT OUT WHEN TRAINING NORMALLY

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
        disco_reduce: str='mean',
        progress_update_interval: int = 50,
        upweight_params: Union[dict, None] = None,
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
        progress_update_interval (int): Interval (in batches) to update the progress bar.
        upweight_params (dict): Parameters for upweighting. None means no upweighting.

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

    weights_batch_multiplier_epoch = torch.Tensor() # [N*B,]
    y_pred_epoch = torch.Tensor() # [N*B, C]
    weights_batch_epoch = torch.Tensor() # [N*B,]
    mjj_epoch = torch.Tensor() # [N*B,]
    mgg_epoch = torch.Tensor() # [N*B,]

    # planning to make these plots:
    # - weight multiplier vs HH score
    # - weight multiplier vs nonres score
    # - weight multiplier vs ttH score
    # - weight multiplier vs singleH score
    # - mgg/mjj correlation vs weight multiplier (coarse bins of weight multiplier) (should see positive trend if upweighting is selecting mgg-mjj correlated events)


    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    # Enable mixed precision on CUDA for speed; safe no-op on CPU
    use_cuda = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_cuda)
    for batch_idx, batch in enumerate(progress_bar):
        # Unpack depending on options
        if use_disco:
            # Expect dataset to include disco_var in batch
            if len(batch) == 5:
                X_batch, y_batch, weights_batch, weights_batch_no, disco_vars_batch = batch
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
        # Use non_blocking transfers to overlap HtoD when pin_memory=True
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        # weights_batch = weights_batch.to(device, non_blocking=True)
        # weights_batch_no = weights_batch_no.to(device, non_blocking=True)
        # if use_disco:
        #     disco_vars_batch = disco_vars_batch.to(device, non_blocking=True)

    
        use_sr_upweight = bool(upweight_params and upweight_params.get("do_upweight", False))
        upweight_score_midpoint = None
        upweight_score_90_percent = None
        upweight_max: float = 1.0
        upweight_min: float = 1.0
        if use_sr_upweight:
            upweight_score_midpoint = torch.tensor(
                _extract_ordered_values(upweight_params["sr_definition"]["score_midpoint"]),
                device=device,
                dtype=torch.float32,
            )
            upweight_score_90_percent = torch.tensor(
                _extract_ordered_values(upweight_params["sr_definition"]["score_90_percent"]),
                device=device,
                dtype=torch.float32,
            )
            upweight_max = float(upweight_params["max_upweight"])
            upweight_min = float(upweight_params["min_upweight"])


        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_cuda):
            y_pred = model(X_batch)
            weights_batch = weights_batch.to(device, non_blocking=True)
            weights_batch_no = weights_batch_no.to(device, non_blocking=True)
            if use_disco:
                disco_vars_batch = disco_vars_batch.to(device, non_blocking=True)
            if use_sr_upweight:
                weights_batch_multiplier = sigmoid_upweight(
                    y_pred,
                    x0=upweight_score_midpoint,
                    x90=upweight_score_90_percent,
                    y_min=upweight_min,
                    y_max=upweight_max,
                )
            else:
                weights_batch_multiplier = torch.ones_like(weights_batch, device=device)

            loss = loss_fn(y_pred, y_batch) # [N]
            # print(f"loss = {loss.detach().cpu().numpy()}")
            # print(f"loss.shape: {loss.shape}")
            # print(f"type(loss) = {type(loss)}")
            wsum: Tensor = weights_batch.sum() # weights_batch is absolute-valued
            weighted_loss: Tensor = (loss * weights_batch).sum() / wsum # scalar
            # print(f"weighted_loss = {weighted_loss.item():.6f}")
            # print(f"weighted_loss.shape: {weighted_loss.shape}")
            # print(f"type(weighted_loss) = {type(weighted_loss)}")

            # Add DisCo term
            if use_disco:
                total_weighted_loss, d_corr = apply_disco(
                    weighted_loss,
                    y_pred,
                    weights_batch * weights_batch_multiplier,
                    disco_vars_batch,
                    decorr_lambda,
                    disco_signal_class_idx,
                    y_true=y_batch,
                    disco_reduce=disco_reduce
                )
                assert d_corr is not None, "d_corr should not be None when using DisCo. Something is wrong inside the apply_disco function."
            else:
                total_weighted_loss: Tensor = weighted_loss # scalar
                d_corr: Tensor = Tensor(0.0, device=device) # Dummy

        # Backward with gradient scaling (no-op on CPU)
        scaler.scale(total_weighted_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_loss_no_abs = (loss * weights_batch_no).sum() / weights_batch_no.sum()

        # compute weighted accuracy
        correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
        weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()

        # Convert once to Python floats to avoid repeated device syncs
        total_weighted_loss_f = float(total_weighted_loss.detach().item())
        weighted_acc_f = float(weighted_acc.detach().item())
        weighted_loss_no_abs_f = float(weighted_loss_no_abs.detach().item())
        weighted_loss_f = float(weighted_loss.detach().item())
        d_corr_f = float(d_corr.detach().item())

        batch_losses.append(            total_weighted_loss_f)
        batch_accs.append(              weighted_acc_f)
        batch_losses_no_abs.append(     weighted_loss_no_abs_f)
        # Avoid recomputation: "no dist corr" is just the nominal weighted loss
        batch_losses_no_dist_corr.append(weighted_loss_f)
        batch_dist_corr.append(         d_corr_f)
        batch_losses_no_abs_no_dist_corr.append(weighted_loss_no_abs_f)

        # Update progress bar
        if progress_update_interval and (batch_idx % progress_update_interval == 0 or batch_idx == len(progress_bar) - 1):
            if use_disco:
                progress_bar.set_postfix({
                    'Loss': f'{weighted_loss_f:.4f}+{decorr_lambda}*{(d_corr_f):.4f}',
                    'Acc': f'{weighted_acc_f:.4f}',
                    'Loss_no_abs': f'{weighted_loss_no_abs_f:.4f}'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{total_weighted_loss_f:.4f}',
                    'Acc': f'{weighted_acc_f:.4f}',
                    'Loss_no_abs': f'{weighted_loss_no_abs_f:.4f}'
                })

    mean_batch_losses                       = float(np.mean(batch_losses))
    mean_accs                               = float(np.mean(batch_accs))
    mean_batch_losses_no_abs                = float(np.mean(batch_losses_no_abs))
    mean_batch_losses_no_dist_corr          = float(np.mean(batch_losses_no_dist_corr))
    mean_batch_losses_no_abs_no_dist_corr   = float(np.mean(batch_losses_no_abs_no_dist_corr))
    mean_batch_dist_corr                    = float(np.mean([dc for dc in batch_dist_corr])) if use_disco else None
    mean_batch_dist_corr_times_lambda       = float(np.mean([dc * decorr_lambda for dc in batch_dist_corr])) if use_disco else None

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
        disco_signal_class_idx: Union[int, list[int], None]=0,
    disco_reduce: str='mean',
    progress_update_interval: int = 50,
    ) -> tuple[float, float, float, Union[float, None], Union[float, None]]:
    model.eval()
    val_losses = []
    val_accs = []
    val_losses_no_dist_corr = []
    val_dist_corr = []
    val_dist_corr_times_lambda = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
    use_cuda = (device.type == 'cuda')
    # inference_mode is slightly faster than no_grad for eval
    with torch.inference_mode():
        for batch_idx, batch in enumerate(progress_bar):
            if use_disco:
                # Expect dataset to include disco_var in batch
                if len(batch) == 5:
                    X_batch, y_batch, weights_batch, weights_batch_no_abs, disco_vars_batch = batch
                elif len(batch) == 4:
                    X_batch, y_batch, weights_batch, disco_vars_batch = batch
                else:
                    raise ValueError("use_disco=True but dataset does not include disco_var. Check how the dataset was created.")
            else:
                # Not using DisCo
                if len(batch) == 4:
                    X_batch, y_batch, weights_batch, weights_batch_no_abs = batch
                else:
                    # Not using weights without absolute value (not using no_absolute_weights)
                    X_batch, y_batch, weights_batch = batch
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            weights_batch = weights_batch.to(device, non_blocking=True)
            # print(f"[DEBUG] weights_batch sum (inside val loop): {weights_batch.sum().item()}") 

            with autocast(enabled=use_cuda):
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()

            # Add DisCo term
            if use_disco:
                disco_vars_batch = disco_vars_batch.to(device, non_blocking=True)
                total_weighted_loss, d_corr = apply_disco(
                    weighted_loss,
                    y_pred,
                    weights_batch,
                    disco_vars_batch,
                    decorr_lambda,
                    disco_signal_class_idx,
                    y_true=y_batch,
                    disco_reduce=disco_reduce
                )
            else:
                total_weighted_loss = weighted_loss
                d_corr = None # Dummy

            # compute weighted accuracy
            correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()

            total_weighted_loss_f = float(total_weighted_loss.detach().item())
            weighted_acc_f = float(weighted_acc.detach().item())
            weighted_loss_f = float(weighted_loss.detach().item())
            d_corr_f = float(d_corr.detach().item()) if d_corr is not None else None

            val_losses.append(                  total_weighted_loss_f)
            val_accs.append(                    weighted_acc_f)
            # For display we separate the dist-corr part, but for storage reuse weighted_loss directly
            val_losses_no_dist_corr.append(     weighted_loss_f)
            val_dist_corr.append(               d_corr_f)
            val_dist_corr_times_lambda.append(  (d_corr_f * decorr_lambda) if d_corr_f is not None else None)

            # Update progress bar
            if progress_update_interval and (batch_idx % progress_update_interval == 0 or batch_idx == len(progress_bar) - 1):
                if use_disco:
                    progress_bar.set_postfix({
                        'Loss': f'{weighted_loss_f:.4f}+{decorr_lambda}*{(d_corr_f if d_corr_f is not None else 0.0):.4f}',
                        'Acc': f'{weighted_acc_f:.4f}',
                    })
                else:
                    progress_bar.set_postfix({
                        'Loss': f'{weighted_loss_f:.4f}',
                        'Acc': f'{weighted_acc_f:.4f}',
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
    _training_config: dict | None               = kwargs.get('training_config', None)

    assert _epoch is not None, "Epoch number must be provided"
    assert _model is not None, "Model must be provided"
    assert _optimizer is not None, "Optimizer must be provided"
    assert _scheduler is not None, "Scheduler must be provided"
    assert _lr_hist is not None, "Learning rate history must be provided"
    assert _disco_in_loss is not None, "Whether DisCo was used in loss must be provided as bool, not None"
    assert _file_path is not None, "File path for saving checkpoint must be provided"
    assert _training_config is not None, "Training configuration must be provided"

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

        'training_config':              _training_config
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
    parser.add_argument('--model_folder', type=str, default="after_random_search_best1", help='Folder to save the trained model')
    #parser.add_argument('', type=str, help='Path to the best parameters')
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)
    with open(f"{args.job_config_path}", 'r') as f:
        job_config = yaml.safe_load(f)

    _validate_training_config(training_config)

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
        # best_params = {"num_layers": 5, "num_nodes": 1024, "act_fn_name": "ELU", "lr": 2.027496582741043e-05, "weight_decay": 5.159904717896079e-05, "dropout_prob": 0.25, "n_trials": 0}
        best_params = {
            "num_layers": 5,
            "num_nodes": 1024,
            "act_fn_name": "ELU",
            "lr": 2.027496582741043e-05,
            "weight_decay": 5.159904717896079e-05,
            "dropout_prob": 0.25,
            "n_trials": 0
        }
        with open(f'{input_path}/random_search_1/best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params, f)

    best_params_path = f'{input_path}/random_search_1/best_params.json'
    path_to_checkpoint = f'{input_path}/{args.model_folder}'
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
        z_train = np.load(f'{input_path}/z_train.npy')  # e.g., mjj and mgg for decorrelation
        z_val = np.load(f'{input_path}/z_val.npy')
        disco_var_train = torch.tensor(z_train, dtype=torch.float32)
        disco_var_val = torch.tensor(z_val, dtype=torch.float32)
    else:
        disco_var_train, disco_var_val = None, None

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, class_weights_for_training, class_weights_for_train_no_absolute, disco_vars=disco_var_train)
    val_dataset = CustomDataset(X_val, y_val, class_weights_for_val, class_weights_for_val_no_absolute, disco_vars=disco_var_val)

    # Create data loaders
    g = torch.Generator().manual_seed(seed)
    batch_size = 1024 #16384 # 8192 # 32768 # 1024 #16384
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Debugging sum of weights per batch issue
    PLOT_WEIGHT_COMPARISON = False
    if PLOT_WEIGHT_COMPARISON:
        all_weights_batch_train = np.array([])
        all_weights_batch_train_no_abs = np.array([])
        all_weights_batch_val = np.array([])
        all_weights_batch_val_no_abs = np.array([])
        for idx, (X_batch, y_batch, weights_batch, weights_batch_no_abs, disco_vars_batch) in enumerate(train_loader):
            all_weights_batch_train = np.append(all_weights_batch_train, weights_batch.numpy())
            all_weights_batch_train_no_abs = np.append(all_weights_batch_train_no_abs, weights_batch_no_abs.numpy())
            if idx >= 1000:
                print("break after 1000 batches")
                break
        for idx, (X_batch, y_batch, weights_batch, weights_batch_no_abs, disco_vars_batch) in enumerate(val_loader):
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

                training_config=training_config
            )
            # TODO: Numpy save y_pred_train.npy, y_train.npy, y_pred_val.npy, and y_val.npy for checkpoint ROC plots
            mlp_plotter.run_condor_job(
                input_path=input_path,                                                    # base path to input files
                condor_dir=f"{path_to_checkpoint}/condor/mlp_plotter/",                   # condor directory
                plot_dir=f"{path_to_checkpoint}/checkpoints/epoch{epoch}/plots/",         # output directory for plots
                checkpoint_file=f"{path_to_checkpoint}/checkpoints/epoch{epoch}/mlp.pth", # checkpoint path
                job_config=job_config,
                model_folder=args.model_folder,
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

        training_config=training_config
        )

    save_predictions(best_model.state_dict(), path_to_checkpoint, training_config, input_path, batch_size=1024)

    # # Load the best state of the model
    # best_model.load_state_dict(best_weights)

    # # Save predictions (optional)
    # best_model.eval()
    # batch_size = 1024
    # y_pred_train_probs = []
    # for i in range(0, len(X_train), batch_size):
    #     X_batch = X_train[i:i + batch_size].to(device)
    #     with torch.no_grad():
    #         y_batch = best_model(X_batch)
    #         y_batch = F.softmax(y_batch, dim=1)
    #         y_pred_train_probs.append(y_batch.cpu().numpy())
    # y_pred_train_probs = np.concatenate(y_pred_train_probs, axis=0)

    # y_pred_val_probs = []
    # for i in range(0, len(X_val), batch_size):
    #     X_batch = X_val[i:i + batch_size].to(device)
    #     with torch.no_grad():
    #         y_batch = best_model(X_batch)
    #         y_batch = F.softmax(y_batch, dim=1)
    #         y_pred_val_probs.append(y_batch.cpu().numpy())
    # y_pred_val_probs = np.concatenate(y_pred_val_probs, axis=0)

    # # best_model.eval()
    # # with torch.no_grad():
    # #     y_pred_train = best_model(X_train.to(device))
    # #     y_pred_val = best_model(X_val.to(device))

    # # y_pred_train_probs = F.softmax(y_pred_train, dim=1)
    # #y_pred_train_np = y_pred_train_probs.cpu().detach().numpy()
    # y_pred_train_np = y_pred_train_probs
    # y_train_np = y_train.cpu().numpy()

    # # y_pred_val_probs = F.softmax(y_pred_val, dim=1)
    # #y_pred_val_np = y_pred_val_probs.cpu().detach().numpy()
    # y_pred_val_np = y_pred_val_probs
    # y_val_np = y_val.cpu().numpy()

    # # Save predictions
    # np.save(f"{path_to_checkpoint}/y_pred_train.npy", y_pred_train_np)
    # np.save(f"{path_to_checkpoint}/y_train.npy", y_train_np)
    # np.save(f"{path_to_checkpoint}/y_pred_val.npy", y_pred_val_np)
    # np.save(f"{path_to_checkpoint}/y_val.npy", y_val_np)
