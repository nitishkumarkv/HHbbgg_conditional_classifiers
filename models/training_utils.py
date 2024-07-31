from mlp import MLP

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import json
import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# the MLP model is already imported
# define all other things that are required for training:
# take the inputs, apply standardization, separate and randomize inputs into training/test sets, optimization function, loss function, backpropagation and weight updation, etc.

input_file = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/dummy_input_for_mlp.parquet'
inputs = ak.from_parquet(input_file)

# load variables from json file
input_var_json = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/input_variables.json"
with open(input_var_json) as f:
    vars_for_training = json.load(f)["mlp"]

# set parameters
input_size = len(vars_for_training)
num_layers = 3
num_nodes = 512
output_size = 4
act_fn = nn.ReLU

# create the model
model = MLP(input_size, num_layers, num_nodes, output_size, act_fn)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('INFO: Used device is', device)
model.to(device)

# Remove default values
processed_vars = []
for var in vars_for_training:
    var_array = ak.to_numpy(inputs[var])
    var_array = var_array[var_array > -998]
    processed_vars.append(var_array)

# Stack the processed variables into a feature matrix
X = np.column_stack([ak.to_numpy(inputs[var]) for var in vars_for_training])
y = np.column_stack([ak.to_numpy(inputs[cls]) for cls in ["is_non_resonant_bkg", "is_ttH_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]])

# Ensure the labels correspond to the filtered features
mask = np.all(np.column_stack([ak.to_numpy(inputs[var]) for var in vars_for_training]) > -998, axis=1)
y = y[mask]

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
print(X_tensor)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

#standardization
scaler = preprocessing.StandardScaler().fit(X_tensor.cpu().numpy())
X_scaled = scaler.transform(X_tensor.cpu().numpy())
X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(device)

#check standardization
for entry in X_scaled.mean(axis=0):
    if entry < 1.0*10**(-6):
        mean_calc = 'correct'
    else:
        mean_calc = 'incorrect'

for entry in X_scaled.std(axis=0):
    if entry == 1.0000:
        std_calc = 'correct'
    else:
        std_calc = 'incorrect'

if mean_calc == 'correct' and std_calc == 'correct':
    print('INFO: standardization works')
else:
    print('INFO: standardization not correct')

#seperate and randomize
X_train, X_test_val, y_train, y_test_val = train_test_split(X_scaled.cpu().numpy(), y_tensor.cpu().numpy(), train_size=0.6, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

#loss function, optimization function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#prepare model and training parameters
n_epochs = 25
batch_size = 32
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_loss = np.inf
best_weights = None
patience = 15
counter = 0

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

#training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []

    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            end=start+batch_size
            X_batch = X_train[start:end].clone().detach()
            y_batch = y_train[start:end].clone().detach()
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    #evaluation
    model.eval()

    #calculate loss and acc
    y_pred = model(X_val)
    ce = loss_fn(y_pred, y_val)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)

    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    val_loss_hist.append(ce)
    val_acc_hist.append(acc)

    #save the best parameters
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    
    #early stopping
    if ce < best_loss:
        best_loss = ce
        best_weights = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1 
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

#load the best state of the model
model.load_state_dict(best_weights)

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/loss_plot')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/acc_plot')
plt.clf()