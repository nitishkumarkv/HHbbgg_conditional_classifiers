import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import copy
from mlp import MLP
import torch.nn.functional as F
import yaml


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, sample_weights, no_aboslute_weights=None):
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.no_aboslute_weights = no_aboslute_weights

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.no_aboslute_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx], self.no_aboslute_weights[idx]
        else:
            return self.X[idx], self.y[idx], self.sample_weights[idx]


# Training and evaluation functions
def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    batch_losses = []
    batch_accs = []
    batch_losses_no_abs = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    for X_batch, y_batch, weights_batch, weights_batch_no in progress_bar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        weights_batch = weights_batch.to(device)
        weights_batch_no = weights_batch_no.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()
        weighted_loss.backward()
        optimizer.step()

        weighted_loss_no_abs = (loss * weights_batch_no).sum() / weights_batch_no.sum()

        # compute weighted accuracy
        correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
        weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()

        batch_losses.append(weighted_loss.item())
        batch_accs.append(weighted_acc.item())
        batch_losses_no_abs.append(weighted_loss_no_abs.item())

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{weighted_loss.item():.4f}',
            'Acc': f'{weighted_acc.item():.4f}',
            'Loss_no_abs': f'{weighted_loss_no_abs.item():.4f}'
        })

    return np.mean(batch_losses), np.mean(batch_accs), np.mean(batch_losses_no_abs)


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    val_losses = []
    val_accs = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
    with torch.no_grad():
        for X_batch, y_batch, weights_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            weights_batch = weights_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()

            # compute weighted accuracy
            correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            weighted_acc = (correct * weights_batch).sum() / weights_batch.sum()

            val_losses.append(weighted_loss.item())
            val_accs.append(weighted_acc.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}',
                'Acc': f'{weighted_acc.item():.4f}'
            })

    return np.mean(val_losses), np.mean(val_accs)


# Save the best model
def save_checkpoint(epoch, model, optimizer, scheduler, train_loss_hist, train_loss_hist_no_absolute_weights, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_hist': train_loss_hist,
        'train_loss_hist_no_absolute_weights': train_loss_hist_no_absolute_weights,
        'val_loss_hist': val_loss_hist,
        'train_acc_hist': train_acc_hist,
        'val_acc_hist': val_acc_hist,
        'lr_hist': lr_hist,
        'best_weights': best_weights,
        'best_loss': best_loss
    }
    torch.save(checkpoint, file_path)
    print(f'Checkpoint saved to {file_path}')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--input_path', type=str, help='Path to the input files')
    parser.add_argument('--training_config_path', type=str, default=10, help='Training configuration path')
    #parser.add_argument('', type=str, help='Path to the best parameters')
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    seed = training_config["random_seed"]
    weight_scheme = training_config["weight_scheme"]

    # --- REPROD SETUP ---
    import random, numpy as np, torch
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
        best_params = {"num_layers": 5, "num_nodes": 1024, "act_fn_name": "ELU", "lr": 2.027496582741043e-05, "weight_decay": 5.159904717896079e-05, "dropout_prob": 0.25, "n_trials": 0}
        with open(f'{input_path}/random_search_1/best_params.json', 'w') as f:
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

    elif weight_scheme == "weighted_only_positive":
        class_weights_for_training = np.load(f'{input_path}/class_weights_only_positive.npy')

    elif weight_scheme == "weighted_CRUW_abs":
        # to be completed
        pass

    elif weight_scheme == "weighted_CRUW_only_positive":
        # to be completed
        pass

    class_weights_for_train_no_aboslute = np.load(f'{input_path}/true_class_weights.npy')
    class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')

    # Convert targets to class indices (if one-hot encoded)
    y_train = np.argmax(y_train, axis=1)
    y_val = np.argmax(y_val, axis=1)

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32)
    class_weights_for_train_no_aboslute = torch.tensor(class_weights_for_train_no_aboslute, dtype=torch.float32)
    class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32)

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

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, class_weights_for_training, class_weights_for_train_no_aboslute)
    val_dataset = CustomDataset(X_val, y_val, class_weights_for_val)

    # Create data loaders
    g = torch.Generator().manual_seed(seed)
    batch_size = 1024 #16384 # 8192 # 32768 # 1024 #16384
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model, loss function, optimizer, and scheduler
    best_model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    best_scheduler = ReduceLROnPlateau(best_optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    # Training loop parameters
    n_epochs = 500
    best_loss = np.inf
    best_weights = None
    patience = 25
    counter = 0

    train_loss_hist = []
    train_loss_hist_no_absolute_weights = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    lr_hist = []

    # Training loop
    for epoch in range(n_epochs):
        # Training
        train_loss, train_acc, train_loss_no_absolute = train_one_epoch(best_model, best_optimizer, train_loader, loss_fn, device)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        train_loss_hist_no_absolute_weights.append(train_loss_no_absolute)

        # Validation
        val_loss, val_acc = evaluate(best_model, val_loader, loss_fn, device)
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

    save_checkpoint(epoch, best_model, best_optimizer, best_scheduler, 
                train_loss_hist, train_loss_hist_no_absolute_weights, val_loss_hist, train_acc_hist, val_acc_hist, 
                best_weights, best_loss, f"{path_to_checkpoint}/mlp.pth")

    # Load the best state of the model
    best_model.load_state_dict(best_weights)

    # Save predictions (optional)
    best_model.eval()
    with torch.no_grad():
        y_pred_train = best_model(X_train.to(device))
        y_pred_val = best_model(X_val.to(device))

    y_pred_train_probs = F.softmax(y_pred_train, dim=1)
    y_pred_train_np = y_pred_train_probs.cpu().detach().numpy()
    y_train_np = y_train.cpu().numpy()

    y_pred_val_probs = F.softmax(y_pred_val, dim=1)
    y_pred_val_np = y_pred_val_probs.cpu().detach().numpy()
    y_val_np = y_val.cpu().numpy()

    # Save predictions
    np.save(f"{path_to_checkpoint}/y_pred_train.npy", y_pred_train_np)
    np.save(f"{path_to_checkpoint}/y_train.npy", y_train_np)
    np.save(f"{path_to_checkpoint}/y_pred_val.npy", y_pred_val_np)
    np.save(f"{path_to_checkpoint}/y_val.npy", y_val_np)