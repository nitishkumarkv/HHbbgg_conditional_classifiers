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

# set parameters
input_size = 5
num_layers = 3
num_nodes = 512
output_size = 4

# create the model
model = MLP(input_size, num_layers, num_nodes, output_size)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_file = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/dummy_input_for_mlp.parquet'
inputs = ak.from_parquet(input_file)

# load variables from json file
input_var_json = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/input_variables.json"
with open(input_var_json) as f:
    vars_for_training = json.load(f)["mlp"]

X = np.column_stack([ak.to_numpy(inputs[var]) for var in vars_for_training])
y = np.column_stack([ak.to_numpy(inputs[cls]) for cls in ["is_non_resonant_bkg", "is_ttH_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

#standardization
scaler = preprocessing.StandardScaler().fit(X_tensor)
X_scaled = scaler.transform(X_tensor)
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

#seperate and randomize
X_train, X_test_val, y_train, y_test_val = train_test_split(X_scaled, y_tensor, train_size=0.6, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5, shuffle=True)

#loss function, optimization function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#prepare model and training parameters
n_epochs = 3
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

#training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = torch.tensor(X_train[start:start+batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_train[start:start+batch_size], dtype=torch.float32)
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
    y_pred = model(X_val)
    ce = loss_fn(y_pred, y_val)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

model.load_state_dict(best_weights)

plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/loss_plot')
plt.clf()
 
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/acc_plot')
plt.clf()