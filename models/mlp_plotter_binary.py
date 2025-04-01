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

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss_hist = checkpoint['train_loss_hist']
    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    #train_loss_hist_no_aboslute = checkpoint['train_loss_hist_no_aboslute']
    val_acc_hist = checkpoint['val_acc_hist']
    lr_hist = checkpoint['lr_hist']
    best_weights = checkpoint['best_weights']
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {file_path}, resuming from epoch {start_epoch + 1}')
    return start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss

inputs_for_MLP = "../data/inputs_for_binary_MLP_20250121/"
input_path="train_inputs_for_binary_MLP_20250121/best/"
path_for_plots = f"{input_path}/plots/"
os.makedirs(path_for_plots, exist_ok=True)
path_to_checkpoint = f"{input_path}/mlp.pth"







start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss = load_checkpoint(path_to_checkpoint)




colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/loss_plot.png')
plt.clf()

#plot loss function
#plt.plot(train_loss_hist_no_aboslute, label="train")
#plt.plot(val_loss_hist, label="validation")
#plt.xlabel("epochs")
#plt.ylabel("cross entropy")
#plt.legend()
#plt.savefig(f'{path_for_plots}/true_loss_plot.png')
#plt.clf()

# plot learning rate
plt.plot(lr_hist)
plt.xlabel("epochs")
plt.ylabel("learning rate")
plt.savefig(f'{path_for_plots}/lr_plot.png')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f'{path_for_plots}/acc_plot.png')
plt.clf()


# load predictions
y_pred_val = np.load(f"{input_path}/y_pred_val.npy")
y_val = np.load(f'{inputs_for_MLP}/y_val.npy')[:, 0]
#rel_w_val = np.load(f'{inputs_for_MLP}/rel_w_val.npy')
rel_w_val = np.load(f'{inputs_for_MLP}/class_weights_for_val.npy')

y_pred_train = np.load(f"{input_path}/y_pred_train.npy")
y_train = np.load(f'{inputs_for_MLP}/y_train.npy')
#rel_w_train = np.load(f'{inputs_for_MLP}/rel_w_train.npy')
rel_w_train = np.load(f'{inputs_for_MLP}/class_weights_for_train_no_aboslute.npy')




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import combinations

class_names = ["GluGluToHH", "VBFToHH_sig"]

# One-vs-One ROC Curves
# For each pair of classes
glu_idx = class_names.index("GluGluToHH")

# List of other class indices

# Iterate over GluGluToHH vs each other class individually
plt.figure(figsize=(8, 6))

GluGluToHH_one_vs_one_roc = {}
# Iterate over GluGluToHH vs each other class individually

i = glu_idx  # Index of GluGluToHH
class_name_i = class_names[0]
class_name_j = class_names[1]
# Select samples belonging to class i or class j
y_true_binary = y_val
y_score = y_pred_val  # Use the probability for class i (GluGluToHH)
weights = rel_w_val
# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
fpr, tpr = zip(*sorted(zip(fpr, tpr)))
roc_auc = auc(fpr, tpr)
# Plot the ROC curve on the same figure
plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')
# store the AUC
GluGluToHH_one_vs_one_roc[f"GluGluToHH_tpr"] = tpr
GluGluToHH_one_vs_one_roc[f"GluGluToHH_fpr"] = fpr

# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
#plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the combined plot
plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')
plt.clf()

#save the auc scores
with open(f'{path_for_plots}/GluGluToHH_vs_all.json', 'w') as f:
    json.dump(GluGluToHH_one_vs_one_roc, f)



"""y_val = y_train
y_pred_val = y_pred_train
rel_w_val = rel_w_train

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    class_name = class_names[i]
    y_true_binary = y_val[:, i]
    y_score = y_pred_val[:, i]
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=rel_w_val)
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:0.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
#plt.title('One-vs-All ROC Curves', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{path_for_plots}/train_roc_curve_one_vs_all.png')
plt.clf()

# One-vs-One ROC Curves
# For each pair of classes
glu_idx = class_names.index("VBFToHH_sig")

# List of other class indices
other_classes = [i for i in range(n_classes) if i != glu_idx]

# Iterate over GluGluToHH vs each other class individually
plt.figure(figsize=(8, 6))

# Iterate over GluGluToHH vs each other class individually
for j in other_classes:
    i = glu_idx  # Index of GluGluToHH
    class_name_i = class_names[i]
    class_name_j = class_names[j]
    # Select samples belonging to class i or class j
    idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
    y_true_binary = y_val[idx, i]
    y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
    weights = rel_w_val[idx]
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve on the same figure
    plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
#plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the combined plot
plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
plt.close()

# One-vs-One ROC Curves
# For each pair of classes
glu_idx = class_names.index("GluGluToHH")

# List of other class indices
other_classes = [i for i in range(n_classes) if i != glu_idx]

# Iterate over GluGluToHH vs each other class individually
plt.figure(figsize=(8, 6))

# Iterate over GluGluToHH vs each other class individually
for j in other_classes:
    i = glu_idx  # Index of GluGluToHH
    class_name_i = class_names[i]
    class_name_j = class_names[j]
    # Select samples belonging to class i or class j
    idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
    y_true_binary = y_val[idx, i]
    y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
    weights = rel_w_val[idx]
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve on the same figure
    plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR', fontsize=12)
plt.ylabel('TPR', fontsize=12)
#plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the combined plot
plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
plt.close()"""