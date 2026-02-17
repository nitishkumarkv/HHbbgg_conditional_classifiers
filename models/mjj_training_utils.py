import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import copy
import torch.nn.functional as F
import yaml
from typing import Union

from mjj_predictor_mlp import MJJPredictorMLP
from training_utils import CustomDataset, save_checkpoint
from utils.device import get_torch_device


def _loss_no_weights(loss_fn, y_pred, y_true, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(y_pred, y_true)
    loss = loss.mean()  # Reduce to scalar for backward pass
    loss.backward()
    optimizer.step()
    return loss.item()



# Regression-specific training and evaluation functions
def train_one_epoch_regression(model, optimizer, data_loader, loss_fn, device, epoch):
    DO_WEIGHTS = True
    model.train()
    batch_losses = []
    batch_losses_no_abs = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    for batch_data in progress_bar:
        if len(batch_data) == 4:  # Training data with both weight types
            X_batch, y_batch, weights_batch, weights_batch_no = batch_data
            weights_batch_no = weights_batch_no.to(device)
        else:  # Validation data with single weight type
            X_batch, y_batch, weights_batch = batch_data
            weights_batch_no = weights_batch  # Use same weights

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().squeeze()  # Ensure float for regression and match dimensions
        weights_batch = weights_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()  # Remove extra dimension for regression
        if not DO_WEIGHTS:
            _loss = _loss_no_weights(loss_fn, y_pred, y_batch, optimizer)
            batch_losses.append(_loss)
            progress_bar.set_postfix({
                'Loss': f'{_loss:.4f}'
            })
            # Add gradient monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        else:
            loss = loss_fn(y_pred, y_batch)
            
            # Guard against division by zero if weights are pathological
            weighted_loss = (loss * weights_batch).sum() / (weights_batch.sum() + 1e-12)
            weighted_loss.backward()
            
            # Add gradient monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Clip gradients if too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            if len(batch_data) == 4:
                weighted_loss_no_abs = (loss * weights_batch_no).sum() / (weights_batch_no.sum() + 1e-12)
                batch_losses_no_abs.append(weighted_loss_no_abs.item())

            batch_losses.append(weighted_loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}'
            })

    if batch_losses_no_abs:
        return np.mean(batch_losses), 0.0, np.mean(batch_losses_no_abs), total_norm  # No accuracy for regression
    else:
        return np.mean(batch_losses), 0.0, 0.0, total_norm


def evaluate_regression(model, data_loader, loss_fn, device, epoch):
    model.eval()
    val_losses = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
    with torch.no_grad():
        for X_batch, y_batch, weights_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().squeeze()  # Ensure float for regression and match dimensions
            weights_batch = weights_batch.to(device)

            y_pred = model(X_batch).squeeze()  # Remove extra dimension for regression
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss * weights_batch).sum() / (weights_batch.sum() + 1e-12)

            val_losses.append(weighted_loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}'
            })

    return np.mean(val_losses), 0.0  # No accuracy for regression


def save_y_pred_hist(y_pred, path, bins=50) -> None:
    """Save histogram of y_pred values to a specified path.

    Args:
        y_pred (np.ndarray): Array of predicted values.
        path (str): Path to save the histogram.
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
    """
    plt.hist(y_pred, bins=bins)
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Values')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def save_y_true_hist(y_true, path, bins=50) -> None:
    """Save histogram of y_true values to a specified path.

    Args:
        y_true (np.ndarray): Array of true values.
        path (str): Path to save the histogram.
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
    """
    plt.hist(y_true, bins=bins)
    plt.xlabel('True Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of True Values')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def save_pred_vs_true_2d_hist(y_pred, y_true, path, bins=50) -> None:
    """Save 2D histogram of y_pred vs y_true values to a specified path.

    Args:
        y_pred (np.ndarray): Array of predicted values.
        y_true (np.ndarray): Array of true values.
        path (str): Path to save the 2D histogram.
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
    """
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    plt.hist2d(y_true, y_pred, bins=bins, cmap='Blues')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('2D Histogram of Predicted vs True Values')
    plt.colorbar(label='Counts')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


from torch.utils.data import Subset

def get_predictions(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, indices: Union[list, None]=None) -> np.ndarray:
    """Get predictions from the model, optionally for specific indices, using a Subset loader."""
    model.eval()
    preds = []
    with torch.no_grad():
        if indices is not None:
            subset = Subset(data_loader.dataset, indices)
            sub_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=False)
            for batch in sub_loader:
                X_batch = batch[0].to(device)
                y_hat = model(X_batch).squeeze()
                preds.append(y_hat.cpu())
        else:
            for batch in data_loader:
                X_batch = batch[0].to(device)
                y_hat = model(X_batch).squeeze()
                preds.append(y_hat.cpu())
    return torch.cat(preds, dim=0).numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Mjj Predictor MLP for sculpting studies")
    parser.add_argument('--input_path', type=str, help='Path to the input files')
    parser.add_argument('--training_config_path', type=str, default=10, help='Training configuration path')
    parser.add_argument('--sculpting_study_config_path', type=str, help='Sculpting study configuration path')
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r', encoding='utf-8') as f:
        training_config = yaml.safe_load(f)

    # Load sculpting study configuration
    with open(f"{args.sculpting_study_config_path}", 'r', encoding='utf-8') as f:
        sculpting_study_config = yaml.safe_load(f)

    seed = training_config["random_seed"]
    weight_scheme = training_config["weight_scheme"]
    max_epoch = sculpting_study_config.get("max_epoch", 500)

    # --- REPROD SETUP ---
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True

    # Set device
    device = get_torch_device(training_config.get("cuda_device"))
    print('\n', 'INFO: Used device is', device, '\n')
    input_path = args.input_path
    os.makedirs(f'{input_path}/sculpting_study', exist_ok=True)
    if not os.path.exists(f'{input_path}/sculpting_study/best_params.json'):
        print('INFO: Creating best_params.json file with default best params')

    best_params = {"num_layers": 3, "num_nodes": 100, "act_fn_name": "ELU", "lr": 2.027496582741043e-05, "weight_decay": 5.159904717896079e-05, "dropout_prob": 0.15, "n_trials": 0}
    with open(f'{input_path}/sculpting_study/best_params.json', 'w', encoding="utf-8") as f:
        json.dump(best_params, f)
    
    best_params_path = f'{input_path}/sculpting_study/best_params.json'
    path_to_checkpoint = f'{input_path}/sculpting_study'

    # Load data
    # n.b. the y targets are in the sculpting_study/ folder, different to the regular multiclass training
    X_train = np.load(f'{input_path}/X_train.npy') # use same data as regular multiclass
    X_val = np.load(f'{input_path}/X_val.npy')
    y_train = np.load(f'{input_path}/sculpting_study/y_train.npy') # require different targets # TODO: make routine for creating these
    y_val = np.load(f'{input_path}/sculpting_study/y_val.npy') # require different targets

    # Save 2D histogram of mjj in feature vs. mjj target
    # Ensure both inputs are 1D numpy arrays of equal length
    # x_feat = np.asarray(X_train[:, -4]).reshape(-1)
    # y_targ = np.asarray(y_train).reshape(-1)
    # plt.hist2d(x_feat, y_targ, bins=[100, 100], cmap='Blues')
    # plt.colorbar(label='Counts')
    # plt.xlabel('Mjj Feature')
    # plt.ylabel('Mjj Target')
    # plt.savefig(f'{input_path}/sculpting_study/mjj_feature_vs_target_2d_hist_1.png')
    # plt.close()

    # Debug: Print target statistics
    print(f"DEBUG: y_train shape: {y_train.shape}")
    print(f"DEBUG: y_train min: {np.min(y_train)}, max: {np.max(y_train)}, mean: {np.mean(y_train)}, std: {np.std(y_train)}")
    print(f"DEBUG: y_val shape: {y_val.shape}")
    print(f"DEBUG: y_val min: {np.min(y_val)}, max: {np.max(y_val)}, mean: {np.mean(y_val)}, std: {np.std(y_val)}")
    print()

    weights_for_training = np.load(f'{input_path}/rel_w_train.npy')
    weights_for_val = np.load(f'{input_path}/rel_w_val.npy')



    # Set weight scheme for training
    # if weight_scheme == "weighted_abs":
    #     class_weights_for_training = np.load(f'{input_path}/class_weights_for_training_abs.npy') # TODO: do we need class weights for sculpting study?

    # elif weight_scheme == "weighted_only_positive":
    #     class_weights_for_training = np.load(f'{input_path}/class_weights_only_positive.npy') # TODO: do we need class weights for sculpting study?

    # elif weight_scheme == "weighted_CRUW_abs":
    #     # to be completed
    #     pass

    # elif weight_scheme == "weighted_CRUW_only_positive":
    #     # to be completed
    #     pass

    # class_weights_for_train_no_aboslute = np.load(f'{input_path}/true_class_weights.npy')
    # class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')

    # For regression, targets should be continuous values, not class indices
    # No need to convert to class indices for mjj prediction
    
    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    weights_for_training = torch.tensor(weights_for_training, dtype=torch.float32)
    weights_for_val = torch.tensor(weights_for_val, dtype=torch.float32)
    # class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32)
    # class_weights_for_train_no_aboslute = torch.tensor(class_weights_for_train_no_aboslute, dtype=torch.float32)
    # class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32)

    # Load input features (optional)
    with open(f'{input_path}/input_vars.txt', 'r', encoding="utf-8") as f:
        input_vars = json.load(f)
    print('INFO: Input features are', input_vars, '\n')

    # Load best parameters
    with open(best_params_path, 'r', encoding="utf-8") as f:
        best_params = json.load(f)

    print("Parameters: ", best_params, '\n')

    # Save the parameters in the training folder
    with open(f"{path_to_checkpoint}/params.json", 'w', encoding="utf-8") as f:
        json.dump(best_params, f)

    # Model parameters
    best_num_layers = best_params['num_layers']
    best_num_nodes = best_params['num_nodes']
    best_act_fn_name = best_params['act_fn_name']
    best_act_fn = getattr(nn, best_act_fn_name)
    # Use tuned learning rate as-is
    best_lr = best_params['lr']
    best_weight_decay = best_params['weight_decay']
    best_dropout_prob = best_params['dropout_prob']
    input_size = X_train.shape[1]
    # Robust output sizing for regression targets
    output_size = 1 if y_train.dim() == 1 else y_train.shape[1]

    print(f"DEBUG: Input size: {input_size}, Output size: {output_size}\n")

    # Create datasets
    # train_dataset = CustomDataset(X_train, y_train, class_weights_for_training, class_weights_for_train_no_aboslute)

    train_dataset = CustomDataset(X_train, y_train, weights_for_training)
    val_dataset = CustomDataset(X_val, y_val, weights_for_val)

    # Make another 2D histogram of mjj feature vs target
    # x_feat = np.asarray(train_dataset.X[:,-4]).reshape(-1)
    # y_targ = np.asarray(train_dataset.y).reshape(-1)
    # plt.hist2d(x_feat, y_targ, bins=[100, 100], cmap='Blues')
    # plt.colorbar(label='Counts')
    # plt.xlabel('Mjj Feature')
    # plt.ylabel('Mjj Target')
    # plt.savefig(f'{input_path}/sculpting_study/mjj_feature_vs_target_2d_hist_2.png')
    # plt.close()

    # Create data loaders
    g = torch.Generator().manual_seed(seed)
    batch_size = 1024 #16384 # 8192 # 32768 # 1024 #16384
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Make yet another 2D histogram of mjj feature vs target
    # x_feat = np.asarray(train_loader.dataset.X[:,-4]).reshape(-1)
    # y_targ = np.asarray(train_loader.dataset.y).reshape(-1)
    # plt.hist2d(x_feat, y_targ, bins=[100, 100], cmap='Blues')
    # plt.colorbar(label='Counts')
    # plt.xlabel('Mjj Feature')
    # plt.ylabel('Mjj Target')
    # plt.savefig(f'{input_path}/sculpting_study/mjj_feature_vs_target_2d_hist_3.png')
    # plt.close()

    # Define model, loss function, optimizer, and scheduler
    best_model = MJJPredictorMLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    loss_fn = nn.MSELoss(reduction='none')
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    best_scheduler = ReduceLROnPlateau(best_optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    # Training loop parameters
    print(f"INFO: Training for {max_epoch} epochs", '\n')
    best_loss = np.inf
    best_weights = None
    patience = 50
    counter = 0

    train_loss_hist = []
    train_loss_hist_no_absolute_weights = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    lr_hist = []
    

    # Training loop
    for epoch in range(max_epoch):
        # Training
        train_loss, train_acc, train_loss_no_absolute, train_param_norm = train_one_epoch_regression(best_model, best_optimizer, train_loader, loss_fn, device, epoch)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        train_loss_hist_no_absolute_weights.append(train_loss_no_absolute)

        print(f"Training gradient norm = {train_param_norm}")

        # Validation
        val_loss, val_acc = evaluate_regression(best_model, val_loader, loss_fn, device, epoch)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Scheduler step
        best_scheduler.step(val_loss)
        current_lr = best_optimizer.param_groups[0]['lr']
        lr_hist.append(current_lr)
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")

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
        print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}", '\n')
        print(f"Epoch {epoch} - Train Loss no abs: {train_loss_no_absolute:.4f}", '\n')
        print()
        n_compare = 5
        y_pred_to_print = get_predictions(best_model, train_loader, device, indices=list(range(n_compare)))
        # print(f"DEBUG: len(y_train): {len(y_train)}")
        # print(f"DEBUG: y_train.shape: {y_train.shape}")
        # print(f"DEBUG: y_val.shape: {y_val.shape}")
        # Generate indices safely based on training tensor length
        # dataset_len = X_train.shape[0]
        # rand_inds = list(set(np.random.randint(0, dataset_len, size=min(10000, dataset_len))))
        # y_pred = get_predictions(best_model, train_loader, device, indices=list(rand_inds))
        # if epoch == 0:
        #     y_true = y_train.cpu().numpy()[rand_inds].squeeze()
        #     if len(y_pred) != len(y_true):
        #         print(f"WARNING: y_pred and y_true have different lengths: {len(y_pred)} vs {len(y_true)}")
        #         print(f"DEBUG: rand_inds: {rand_inds}")
        #         print(f"DEBUG: max(rand_inds): {np.max(rand_inds)}, min(rand_inds): {np.min(rand_inds)}")

        print(f"DEBUG: Compare train y_pred vs y_true:")
        for i in range(n_compare):
            print(f"  y_pred: {y_pred_to_print[i]:.4f}, y_true: {y_train[i].item():.4f}")

        # save_y_pred_hist(y_pred, f"{path_to_checkpoint}/plots/training/y_pred/y_pred_epoch{epoch}.png")
        # if epoch == 0:
        #     save_y_true_hist(y_true, f"{path_to_checkpoint}/plots/training/y_true/y_true_epoch{epoch}.png")
        # save_pred_vs_true_2d_hist(y_pred, y_true, f"{path_to_checkpoint}/plots/training/y_pred_vs_true/y_pred_vs_true_epoch{epoch}.png")


        # Print first five values of y_pred vs y_true for a batch

    save_checkpoint(epoch, best_model, best_optimizer, best_scheduler,
                train_loss_hist, train_loss_hist_no_absolute_weights, val_loss_hist, train_acc_hist, val_acc_hist,
                best_weights, best_loss, f"{path_to_checkpoint}/mlp.pth", lr_hist)
    
    # Load the best state of the model
    if best_weights is not None:
        best_model.load_state_dict(best_weights)

    # Save predictions (for regression)
    best_model.eval()
    batch_size = 1024
    y_pred_train_values = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = best_model(X_batch).squeeze()  # Remove extra dimension
            y_pred_train_values.append(y_batch.cpu().numpy())
    y_pred_train_values = np.concatenate(y_pred_train_values, axis=0)

    y_pred_val_values = []
    for i in range(0, len(X_val), batch_size):
        X_batch = X_val[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = best_model(X_batch).squeeze()  # Remove extra dimension
            y_pred_val_values.append(y_batch.cpu().numpy())
    y_pred_val_values = np.concatenate(y_pred_val_values, axis=0)

    y_pred_train_np = y_pred_train_values
    y_train_np = y_train.cpu().numpy()

    y_pred_val_np = y_pred_val_values
    y_val_np = y_val.cpu().numpy()

    # Save predictions
    # n.b. multiclass saves these in /{path_to_checkpoint}/ but we append a /predictions/ folder here
    os.makedirs(f"{path_to_checkpoint}/predictions", exist_ok=True)
    np.save(f"{path_to_checkpoint}/predictions/y_pred_train.npy", y_pred_train_np)
    np.save(f"{path_to_checkpoint}/predictions/y_train.npy", y_train_np)
    np.save(f"{path_to_checkpoint}/predictions/y_pred_val.npy", y_pred_val_np)
    np.save(f"{path_to_checkpoint}/predictions/y_val.npy", y_val_np)
