import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import tqdm
import copy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import optuna
import os
import awkward as ak
import mplhep as hep
from mlp import MLP
import pickle

# the MLP model is already imported
# define all other things that are required for training:
# take the inputs, apply standardization, separate and randomize inputs into training/test sets, optimization function, loss function, backpropagation and weight updation, etc.

input_path="../data/inputs_for_MLP_20241120/"
best_params_path = 'test_random_search/best_params.json'
path_to_checkpoint = "train_inputs_for_MLP_20241120/parmas1_rm_softmax/"
os.makedirs(path_to_checkpoint, exist_ok=True)

X_train = np.load(f'{input_path}/X_train.npy')
X_val = np.load(f'{input_path}/X_val.npy')
#X_test = np.load(f'{input_path}/X_test.npy')
y_train = np.load(f'{input_path}/y_train.npy')
y_val = np.load(f'{input_path}/y_val.npy')
#y_test = np.load(f'{input_path}/y_test.npy')
rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
#rel_w_test = np.load(f'{input_path}/rel_w_test.npy')
class_weights_for_training = np.load(f'{input_path}/class_weights_for_training.npy')
class_weights_for_train_no_aboslute = np.load(f'{input_path}/class_weights_for_train_no_aboslute.npy')
class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')
#class_weights_for_test = np.load(f'{input_path}/class_weights_for_test.npy')

classes = ["is_non_resonant_bkg", "is_ttH_bkg", "is_single_H_bkg", "is_GluGluToHH_sig"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('INFO: Used device is', device)

# load and print input features
with open(f'{input_path}/input_vars.txt', 'r') as f:
    input_vars = json.load(f)
print('INFO: Input features are', input_vars)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
#X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
#y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
rel_w_train = torch.tensor(rel_w_train, dtype=torch.float32).to(device)
#rel_w_test = torch.tensor(rel_w_test, dtype=torch.float32).to(device)
rel_w_val = torch.tensor(rel_w_val, dtype=torch.float32).to(device)
class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).to(device)
class_weights_for_train_no_aboslute = torch.tensor(class_weights_for_train_no_aboslute, dtype=torch.float32).to(device)
class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).to(device)
#class_weights_for_test = torch.tensor(class_weights_for_test, dtype=torch.float32).to(device)


with open(best_params_path, 'r') as f:
    best_params = json.load(f)

best_params = {'num_layers': 5, 'num_nodes': 128, 'act_fn_name': 'SELU', 'lr': 0.0001032386980817001, 'weight_decay': 3.885730450373218e-06, 'dropout_prob': 0.3}
best_params = {'num_layers': 2, 'num_nodes': 512, 'act_fn_name': 'ELU', 'lr': 0.00014644057501430737, 'weight_decay': 9.090352494480058e-05, 'dropout_prob': 0.1}
#best_params = {'num_layers': 5, 'num_nodes': 256, 'act_fn_name': 'ELU', 'lr': 6.578352626266264e-05, 'weight_decay': 2.413568912194892e-05, 'dropout_prob': 0.0}

print("Parameters: ", best_params)

# also save the parameters in the training folder
with open(f"{path_to_checkpoint}/params.json", 'w') as f:
    json.dump(best_params, f)


best_num_layers = best_params['num_layers']
best_num_nodes = best_params['num_nodes']
best_act_fn_name = best_params['act_fn_name']
best_act_fn = getattr(nn, best_act_fn_name)
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_dropout_prob = best_params['dropout_prob']
input_size = X_train.shape[1]
output_size = y_train.shape[1]

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss_hist, train_loss_hist_no_aboslute, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_hist': train_loss_hist,
        'train_loss_hist_no_aboslute': train_loss_hist_no_aboslute,
        'val_loss_hist': val_loss_hist,
        'train_acc_hist': train_acc_hist,
        'val_acc_hist': val_acc_hist,
        'best_weights': best_weights,
        'best_loss': best_loss
    }
    torch.save(checkpoint, file_path)
    print(f'Checkpoint saved to {file_path}')


# best_num_layers = 2
# best_num_nodes = 256
# #best_act_fn_name = best_params['act_fn_name']
# best_act_fn = nn.ReLU
# best_lr = 0.0001
# best_weight_decay = 1e-4
# best_dropout_prob = 0
# input_size = X_train.shape[1]
# output_size = 4

best_model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='none')
best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
best_scheduler = ReduceLROnPlateau(best_optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

#prepare model and training parameters
n_epochs = 400
batch_size = 1024
batches_per_epoch = len(X_train) // batch_size

best_loss = np.inf
best_weights = None
patience = 75
counter = 0

train_loss_hist = []
train_loss_hist_no_aboslute = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

#training loop
for epoch in range(n_epochs):
    batch_loss = []
    batch_loss_no_aboslute = []
    batch_acc = []
    best_model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            end=start+batch_size
            X_batch = X_train[start:end].clone().detach()
            y_batch = y_train[start:end].clone().detach()
            weights_batch = class_weights_for_training[start:end].clone().detach()
            weights_batch_no_absolute = class_weights_for_train_no_aboslute[start:end].clone().detach()
            # forward pass
            y_pred = best_model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss*weights_batch).sum() / weights_batch.sum()
            weighted_loss_no_absolute = (loss*weights_batch_no_absolute).sum() / weights_batch_no_absolute.sum()
            # backward pass
            best_optimizer.zero_grad()
            weighted_loss.backward()
            best_optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            batch_loss.append(float(weighted_loss))
            batch_loss_no_aboslute.append(float(weighted_loss_no_absolute))
            batch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(weighted_loss),
                acc=float(acc)
            )

    #evaluation
    best_model.eval()
    with torch.no_grad():
    #calculate loss and acc
        y_pred = best_model(X_val)
        val_loss = loss_fn(y_pred, y_val)
        weighted_val_loss = (val_loss*class_weights_for_val).sum() / class_weights_for_val.sum()
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()

        # Scheduler step
        best_scheduler.step(weighted_val_loss)
        for param_group in best_optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")
        
        ce = float(np.mean(weighted_val_loss.cpu().detach().numpy()))
        acc = float(acc)
        train_loss_hist.append(np.mean(batch_loss))
        train_loss_hist_no_aboslute.append(np.mean(batch_loss_no_aboslute))
        train_acc_hist.append(np.mean(batch_acc))
        val_loss_hist.append(ce)
        val_acc_hist.append(acc)

        #save the best parameters
        if ce < best_loss:
            best_loss = ce
            best_weights = copy.deepcopy(best_model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"Counter: {counter}")
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

#load the best state of the model
best_model.load_state_dict(best_weights)

#training prediction for the best model
y_pred_train = best_model(X_train)
y_pred_np = torch.argmax(y_pred_train, 1).cpu().numpy()
y_train_np = torch.argmax(y_train, 1).cpu().numpy()

#validation prediction for the best model
y_pred_val = best_model(X_val)
y_pred_np = torch.argmax(y_pred_val, 1).cpu().numpy()
y_val_np = torch.argmax(y_val, 1).cpu().numpy()



save_checkpoint(epoch, best_model, best_optimizer, best_scheduler, 
                train_loss_hist, train_loss_hist_no_aboslute, val_loss_hist, train_acc_hist, val_acc_hist, 
                best_weights, best_loss, f"{path_to_checkpoint}/mlp.pth")

np.save(f"{path_to_checkpoint}/y_pred_train.npy", y_pred_train.cpu().detach().numpy())
np.save(f"{path_to_checkpoint}/y_train_np.npy", y_train_np)
np.save(f"{path_to_checkpoint}y_train_np.npy", y_train_np)

np.save(f"{path_to_checkpoint}/y_pred_val.npy", y_pred_val.cpu().detach().numpy())
np.save(f"{path_to_checkpoint}y_pred_np.npy", y_pred_np)
np.save(f"{path_to_checkpoint}y_val_np.npy", y_val_np)