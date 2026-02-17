"""
Example: 
python3 -m utils.predictions --input_dir Version_20250524_MVAID_forPreApp --checkpoint_path Version_20250524_MVAID_forPreApp/after_random_search_best1/mlp.pth --training_config_path config/Version_20250524_MVAID_forPreApp/training_config.yaml

Useful if training aborted unexpectedly and you want to run predictions (and later categorization)
using a periodically saved checkpoint.
"""
import os
import sys
import yaml
import json
from typing import (
    Optional,
    Tuple,
)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mlp import MLP
from utils.device import get_torch_device

def _load_X_train_val(input_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X_train and X_val using memory mapping for chunked access.

    This avoids loading the full arrays into RAM; downstream code should slice
    these arrays in batches and convert each slice to a torch.Tensor on the fly.

    Args:
        input_dir (str): Directory containing the input data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Memory-mapped numpy arrays for X_train and X_val.
    """
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(input_dir, 'X_val.npy'), mmap_mode='r')

    return X_train, X_val


def _load_y_train_val(input_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load y_train and y_val, convert from one-hot to class indices, return as Tensors.

    Args:
        input_dir (str): Directory containing the input data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: y_train, y_val tensors.
    """
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(input_dir, 'y_val.npy'))
    
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    y_train = torch.from_numpy(y_train.astype(np.int64))
    y_val = torch.from_numpy(y_val.astype(np.int64))
    return y_train, y_val


def save_predictions_from_checkpoint(
        input_dir: str, # e.g. Version_20250524_MVAID_forPreApp
        checkpoint_path: str, # /path/to/mlp.pth
        training_config_path: str, # /path/to/training_config.yaml
        batch_size: int = 1024,
):
    """
    Load a model from a checkpoint and save its predictions on the training data.
    """
    # Load checkpoint to the appropriate device (safe on CPU as well)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model_state_dict']
    checkpoint_dir = os.path.dirname(checkpoint_path)

    with open(training_config_path, 'r', encoding='utf-8') as f:
        training_config = yaml.safe_load(f)

    save_predictions(
        model_state_dict=model, # loaded pytorch model state dict
        checkpoint_dir=checkpoint_dir, # dir holding checkpoint
        training_config=training_config, # training config dict
        input_dir = input_dir, # dir holding X/y data, e.g. Version_20250524_MVAID_forPreApp
        batch_size=batch_size, # batch size for predictions
        )


def save_predictions(
        model_state_dict: dict,
        checkpoint_dir: str,
        training_config: dict,
        input_dir: str,
        batch_size: int = 1024,
    ):
    """Run model inference over train and val sets, then save to .npy files.

    Args:
        model (dict): The model to use for inference.
        checkpoint_dir (str): Directory to save the output .npy files.
        training_config (dict): Training configuration parameters from training_config.yaml.
        input_dir (str): Directory containing the input data, e.g. Version_20250524_MVAID_forPreApp.
        batch_size (int, optional): Batch size for predictions. Defaults to 1024.
    """
    device = get_torch_device(training_config.get("cuda_device"))

    # Load data
    X_train, X_val = _load_X_train_val(input_dir)
    y_train, y_val = _load_y_train_val(input_dir)

    # Initialize model
    with open(f"{checkpoint_dir}/params.json", 'r', encoding='utf-8') as f:
        model_params = json.load(f)
    input_size = X_train.shape[1]
    num_layers = model_params['num_layers']
    num_nodes = model_params['num_nodes']
    output_size = len(np.unique(y_train.numpy()))
    act_fn = getattr(nn, model_params['act_fn_name'])
    dropout_prob = model_params['dropout_prob']
    model = MLP(input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob).to(device)
    model.load_state_dict(model_state_dict)

    model.eval()
    y_pred_train_probs = []
    for i in range(0, X_train.shape[0], batch_size):
        X_np = X_train[i:i + batch_size]
        X_batch = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(device)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_pred_train_probs.append(y_batch.cpu().numpy())
    y_pred_train_probs = np.concatenate(y_pred_train_probs, axis=0)

    y_pred_val_probs = []
    for i in range(0, X_val.shape[0], batch_size):
        X_np = X_val[i:i + batch_size]
        X_batch = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(device)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_pred_val_probs.append(y_batch.cpu().numpy())
    y_pred_val_probs = np.concatenate(y_pred_val_probs, axis=0)

    # model.eval()
    # with torch.no_grad():
    #     y_pred_train = model(X_train.to(device))
    #     y_pred_val = model(X_val.to(device))

    # y_pred_train_probs = F.softmax(y_pred_train, dim=1)
    #y_pred_train_np = y_pred_train_probs.cpu().detach().numpy()
    y_pred_train_np = y_pred_train_probs
    y_train_np = y_train.cpu().numpy()

    # y_pred_val_probs = F.softmax(y_pred_val, dim=1)
    #y_pred_val_np = y_pred_val_probs.cpu().detach().numpy()
    y_pred_val_np = y_pred_val_probs
    y_val_np = y_val.cpu().numpy()

    # Save predictions
    np.save(f"{checkpoint_dir}/y_pred_train.npy", y_pred_train_np)
    np.save(f"{checkpoint_dir}/y_train.npy", y_train_np)
    np.save(f"{checkpoint_dir}/y_pred_val.npy", y_pred_val_np)
    np.save(f"{checkpoint_dir}/y_val.npy", y_val_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model predictions from checkpoint.")
    save_pred_group = parser.add_argument_group("Save Predictions Arguments")
    save_pred_group.add_argument("--input_dir", type=str, required=True, help="Directory containing the input data.")
    save_pred_group.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    save_pred_group.add_argument("--training_config_path", type=str, required=True, help="Path to the training configuration YAML file.")
    save_pred_group.add_argument("--batch_size", type=int, default=1024, help="Batch size for predictions.")

    args = parser.parse_args()
    
    save_predictions_from_checkpoint(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint_path,
        training_config_path=args.training_config_path,
        batch_size=args.batch_size,
    ) 
