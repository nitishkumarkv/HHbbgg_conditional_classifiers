from mlp import MLP

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import json
import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report

# the MLP model is already imported
# define all other things that are required for training:
# take the inputs, apply standardization, separate and randomize inputs into training/test sets, optimization function, loss function, backpropagation and weight updation, etc.

tensors_for_training = torch.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/tensors_for_training', weights_only=True)

X_train = tensors_for_training['X_train']
X_val = tensors_for_training['X_val']
X_test = tensors_for_training['X_test']
y_train = tensors_for_training['y_train']
y_val = tensors_for_training['y_val']
y_test = tensors_for_training['y_test']
n_inputs = tensors_for_training['n_inputs']
weights = tensors_for_training['weights'] * 1000000
print(weights)

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

# set parameters
input_size = n_inputs
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

#loss function, optimization function
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#prepare model and training parameters
n_epochs = 15
batch_size = 32
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_loss = np.inf
best_weights = None
patience = 10
counter = 0

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

#training loop
for epoch in range(n_epochs):
    batch_loss = []
    batch_acc = []

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

            batch_loss.append(float(loss))
            batch_acc.append(float(acc))
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

    train_loss_hist.append(np.mean(batch_loss))
    train_acc_hist.append(np.mean(batch_acc))
    val_loss_hist.append(ce)
    val_acc_hist.append(acc)

    #save the best parameters
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
        
        #create cm for best model state
        # Convert tensors to numpy arrays for confusion matrix calculation
        y_pred_np = torch.argmax(y_pred, 1).cpu().numpy()
        y_val_np = torch.argmax(y_val, 1).cpu().numpy()
        cm = confusion_matrix(y_val_np, y_pred_np, normalize='true')
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2g', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/cm_plot')
        plt.clf()
    
    #early stopping
    if ce < best_loss:
        best_loss = ce
        counter = 0
    else:
        counter += 1 
    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

#load the best state of the model
model.load_state_dict(best_weights)
#torch.save(model, '/net/scratch_cms3a/seiler/public')

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