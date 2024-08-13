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
from mlp import MLP

# the MLP model is already imported
# define all other things that are required for training:
# take the inputs, apply standardization, separate and randomize inputs into training/test sets, optimization function, loss function, backpropagation and weight updation, etc.

input_path="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_sw10/"

X_train = np.load(f'{input_path}/X_train.npy')
X_val = np.load(f'{input_path}/X_val.npy')
X_test = np.load(f'{input_path}/X_test.npy')
y_train = np.load(f'{input_path}/y_train.npy')
y_val = np.load(f'{input_path}/y_val.npy')
y_test = np.load(f'{input_path}/y_test.npy')
rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
rel_w_test = np.load(f'{input_path}/rel_w_test.npy')
inc_rel_w_train = np.load(f'{input_path}/inc_rel_w_train.npy')
inc_rel_w_val = np.load(f'{input_path}/inc_rel_w_val.npy')
inc_rel_w_test = np.load(f'{input_path}/inc_rel_w_test.npy')
class_weights_for_training = np.load(f'{input_path}/class_weights_for_training.npy')

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

X_train[X_train == -999] = -9
X_val[X_val == -999] = -9
X_test[X_test == -999] = -9

# Use GPU if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('INFO: Used device is', device)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
rel_w_train = torch.tensor(rel_w_train, dtype=torch.float32).to(device)
rel_w_test = torch.tensor(rel_w_test, dtype=torch.float32).to(device)
rel_w_val = torch.tensor(rel_w_val, dtype=torch.float32).to(device)
inc_rel_w_train = torch.tensor(inc_rel_w_train, dtype=torch.float32).to(device)
inc_rel_w_test = torch.tensor(inc_rel_w_test, dtype=torch.float32).to(device)
inc_rel_w_val = torch.tensor(inc_rel_w_val, dtype=torch.float32).to(device)
class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).to(device)

best_params_path = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/best_hyperparams.json'

with open(best_params_path, 'r') as f:
    best_params = json.load(f)
    print(best_params)

best_num_layers = best_params['num_layers']
best_num_nodes = best_params['num_nodes']
best_act_fn_name = best_params['act_fn_name']
best_act_fn = getattr(nn, best_act_fn_name)
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_dropout_prob = best_params['dropout_prob']
input_size = X_train.shape[1]
output_size = 4

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
n_epochs = 350
batch_size = 512
batches_per_epoch = len(X_train) // batch_size

best_loss = np.inf
best_weights = None
patience = 50
counter = 0

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

#training loop
for epoch in range(n_epochs):
    batch_loss = []
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
            # forward pass
            y_pred = best_model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss*weights_batch).sum() / weights_batch.sum()
            # backward pass
            best_optimizer.zero_grad()
            weighted_loss.backward()
            best_optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            batch_loss.append(float(weighted_loss))
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
        weighted_val_loss = (val_loss*rel_w_val).sum() / rel_w_val.sum()
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()

        # Scheduler step
        best_scheduler.step(weighted_val_loss)
        for param_group in best_optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")
        
        ce = float(np.mean(weighted_val_loss.cpu().detach().numpy()))
        acc = float(acc)
        train_loss_hist.append(np.mean(batch_loss))
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
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")

#load the best state of the model
best_model.load_state_dict(best_weights)
#torch.save(best_model, '/net/scratch_cms3a/seiler/public')

#validation prediction for the best model
y_pred_val = best_model(X_val)
y_pred_np = torch.argmax(y_pred_val, 1).cpu().numpy()
y_val_np = torch.argmax(y_val, 1).cpu().numpy()

threshhold=0.5
mask = y_pred_np > threshhold
y_pred_flt = y_pred_np[mask]
y_val_flt = y_val_np[mask]
rel_w_val_np = rel_w_val.cpu().numpy()
rel_w_val_flt = rel_w_val_np[mask]

path_for_plots="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models_sw10/"
os.makedirs(path_for_plots)

# Plot confusion matrix
cm = confusion_matrix(y_val_flt, y_pred_flt, normalize='true', sample_weight=rel_w_val_flt)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='.2g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(f'{path_for_plots}/cm_plot')
plt.clf()

#ROC
y_val_bin = label_binarize(y_val_np, classes = range(output_size))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(output_size):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_val[:, i].cpu().detach(), sample_weight=rel_w_val.cpu().numpy())
    # Ensure fpr is strictly increasing
    fpr[i], tpr[i] = zip(*sorted(zip(fpr[i], tpr[i])))
    fpr[i] = np.array(fpr[i])
    tpr[i] = np.array(tpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']
for i, color in zip(range(output_size), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (One-vs-All)')
plt.legend(loc="lower right")
plt.savefig(f'{path_for_plots}/roc_plot')
plt.clf()

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/loss_plot')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f'{path_for_plots}/acc_plot')
plt.clf()