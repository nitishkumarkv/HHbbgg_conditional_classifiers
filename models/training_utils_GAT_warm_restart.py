import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm
import copy
from mlp import MLP
import torch.nn.functional as F
from GAT import GAT_classifier

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n', 'INFO: Used device is', device, '\n')

# Paths
input_path = "../data/gnn_inputs_20241126"
path_to_checkpoint = "train_gnn_inputs_20241126/training_500/"
os.makedirs(path_to_checkpoint, exist_ok=True)
warm_start = True
if warm_start:
    path_to_save = f"{path_to_checkpoint}/warm_start"
    os.makedirs(path_to_save, exist_ok=True)


# Load data
print(f"INFO: Loading data_lst_train")
data_lst_train = torch.load(f'{input_path}/data_lst_train.pt')
print(f"INFO: Loading data_lst_val")
data_lst_val = torch.load(f'{input_path}/data_lst_val.pt')

# Define custom dataset
class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        #return data.x, data.y, data.weight
        return data

# Create datasets
train_dataset = GraphDataset(data_lst_train)
val_dataset = GraphDataset(data_lst_val)

# Create data loaders
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the model
input_dim = data_lst_train[0].x.size(1)
hidden_dim = 128
num_init_mlp_layers = 2
num_final_mlp_layers = 2
num_gat_layers = 2
num_message_passing_layers = 1
num_extra_feat = data_lst_train[0].extra_feat.size(1)
output_dim = data_lst_train[0].y.size(1)
activation = "elu"
seed = 42

lr = 1e-3
dropout = 0.1
weight_decay = 1e-5

model = GAT_classifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_init_mlp_layers=num_init_mlp_layers,
        num_final_mlp_layers=num_final_mlp_layers,
        num_gat_layers=num_gat_layers,
        num_message_passing_layers=num_message_passing_layers,
        num_extra_feat=num_extra_feat,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation
    ).to(device)

print(model)

loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)


# Load previous checkpoint if available
if (os.path.exists(f"{path_to_checkpoint}/GAT.pth")) and (warm_start):
    print("INFO: Loading previous checkpoint for warm start")
    checkpoint = torch.load(f"{path_to_checkpoint}/GAT.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_loss_hist = checkpoint['train_loss_hist']
    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    val_acc_hist = checkpoint['val_acc_hist']
    lr_hist = checkpoint['lr_hist']

    # Adjust starting point
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    best_weights = checkpoint['best_weights']
    print(f"INFO: Resuming from epoch {start_epoch}")
else:
    print("INFO: No previous checkpoint found, starting fresh")
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    lr_hist = []
    start_epoch = 0
    best_loss = np.inf
    best_weights = None

#if len(lr_hist) != 0:
#    lr = lr_hist[-1]

# Define loss function, optimizer, and scheduler


# Training and evaluation functions
def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    batch_losses = []
    batch_accs = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    for data in progress_bar:
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(data)
        y_true = data.y.argmax(dim=1)
        weights = data.weight.squeeze()  # Ensure weights are 1D

        # Loss computation
        loss = loss_fn(y_pred, y_true)
        weighted_loss = (loss * weights).sum() / weights.sum()

        # Backward pass
        weighted_loss.backward()
        optimizer.step()

        # Accuracy computation
        correct = (torch.argmax(y_pred, dim=1) == y_true).float()
        weighted_acc = (correct * weights).sum() / weights.sum()

        batch_losses.append(weighted_loss.item())
        batch_accs.append(weighted_acc.item())

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{weighted_loss.item():.4f}',
            'Acc': f'{weighted_acc.item():.4f}'
        })

    return np.mean(batch_losses), np.mean(batch_accs)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    val_losses = []
    val_accs = []

    progress_bar = tqdm(data_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)

            # Forward pass
            y_pred = model(data)
            y_true = data.y.argmax(dim=1)
            weights = data.weight.squeeze()  # Ensure weights are 1D

            # Loss computation
            loss = loss_fn(y_pred, y_true)
            weighted_loss = (loss * weights).sum() / weights.sum()

            # Accuracy computation
            correct = (torch.argmax(y_pred, dim=1) == y_true).float()
            weighted_acc = (correct * weights).sum() / weights.sum()

            val_losses.append(weighted_loss.item())
            val_accs.append(weighted_acc.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{weighted_loss.item():.4f}',
                'Acc': f'{weighted_acc.item():.4f}'
            })

    return np.mean(val_losses), np.mean(val_accs)


# Training loop parameters
n_epochs = 50
best_loss = np.inf
best_weights = None
patience = 75
counter = 0

# Training loop
for epoch in range(n_epochs):
    # Training
    train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, loss_fn, device)
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # Validation
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    # Scheduler step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lr_hist.append(current_lr)
    print(f"Epoch {epoch}: Current learning rate = {current_lr}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        print(f"Counter: {counter}")
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}", '\n')

# Save the best model
def save_checkpoint(epoch, model, optimizer, scheduler, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_hist': train_loss_hist,
        'val_loss_hist': val_loss_hist,
        'train_acc_hist': train_acc_hist,
        'val_acc_hist': val_acc_hist,
        'lr_hist': lr_hist,
        'best_weights': best_weights,
        'best_loss': best_loss
    }
    torch.save(checkpoint, file_path)
    print(f'Checkpoint saved to {file_path}')

save_checkpoint(epoch, model, optimizer, scheduler, 
                train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, 
                best_weights, best_loss, f"{path_to_save}/GAT.pth")


# Load the best state of the model
model.load_state_dict(best_weights)

model.eval()

# Placeholder lists to store predictions and labels
y_pred_train_probs_list = []
y_train_np_list = []
weights_train_list = []

y_pred_val_probs_list = []
y_val_np_list = []
weights_val_list = []

print("INFO: Generating predictions for training set")
# Generate predictions for training set
with torch.no_grad():
    for data in train_loader:
        data = data.to(device)
        y_pred_train = model(data)
        y_pred_train_probs = F.softmax(y_pred_train, dim=1).cpu().detach().numpy()
        y_train_np = data.y.cpu().numpy()
        weights_train = data.weight.cpu().numpy()

        y_pred_train_probs_list.append(y_pred_train_probs)
        y_train_np_list.append(y_train_np)
        weights_train_list.append(weights_train)

print("INFO: Generating predictions for training set")
# Generate predictions for validation set
with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        y_pred_val = model(data)
        y_pred_val_probs = F.softmax(y_pred_val, dim=1).cpu().detach().numpy()
        y_val_np = data.y.cpu().numpy()
        weights_val = data.weight.cpu().numpy()

        y_pred_val_probs_list.append(y_pred_val_probs)
        y_val_np_list.append(y_val_np)
        weights_val_list.append(weights_val)

# Concatenate all batches into single arrays
y_pred_train_np = np.concatenate(y_pred_train_probs_list, axis=0)
y_train_np = np.concatenate(y_train_np_list, axis=0)
weights_train_np = np.concatenate(weights_train_list, axis=0)

y_pred_val_np = np.concatenate(y_pred_val_probs_list, axis=0)
y_val_np = np.concatenate(y_val_np_list, axis=0)
weights_val_np = np.concatenate(weights_val_list, axis=0)

print("INFO: Saving predictions and weights")
# Save predictions and weights
np.save(f"{path_to_save}/y_pred_train.npy", y_pred_train_np)
np.save(f"{path_to_save}/y_train.npy", y_train_np)
np.save(f"{path_to_save}/weights_train.npy", weights_train_np)

np.save(f"{path_to_save}/y_pred_val.npy", y_pred_val_np)
np.save(f"{path_to_save}/y_val.npy", y_val_np)
np.save(f"{path_to_save}/weights_val.npy", weights_val_np)

# also save the hyperparameters used
hyperparameters = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "num_init_mlp_layers": num_init_mlp_layers,
    "num_final_mlp_layers": num_final_mlp_layers,
    "num_gat_layers": num_gat_layers,
    "num_message_passing_layers": num_message_passing_layers,
    "num_extra_feat": num_extra_feat,
    "output_dim": output_dim,
    "activation": activation,
    "seed": seed,
    "lr": lr,
    "dropout": dropout,
    "weight_decay": weight_decay,
    "n_epochs": epoch,
    "best_loss": best_loss
}

with open(f"{path_to_save}/hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f)

