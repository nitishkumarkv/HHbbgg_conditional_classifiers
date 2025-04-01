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

def load_checkpoint(file_path, model, optimizer, scheduler):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss_hist = checkpoint['train_loss_hist']
    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    val_acc_hist = checkpoint['val_acc_hist']
    best_weights = checkpoint['best_weights']
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {file_path}, resuming from epoch {start_epoch + 1}')
    return start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss

input_path="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm/"
best_params_path = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/test_best_hyperparams.json'
path_for_plots = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/performance_test/"
input_vars_path = os.path.join(os.path.dirname(__file__), "input_variables.json")
path_for_hists="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/selected_signal_hists/"
out_path = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/"
path_to_checkpoint = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/mlp_checkpoint/"

X_train = np.load(f'{input_path}/X_train.npy')
X_val = np.load(f'{input_path}/X_val.npy')
X_test = np.load(f'{input_path}/X_test.npy')
y_train = np.load(f'{input_path}/y_train.npy')
y_val = np.load(f'{input_path}/y_val.npy')
y_test = np.load(f'{input_path}/y_test.npy')
rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
rel_w_test = np.load(f'{input_path}/rel_w_test.npy')
class_weights_for_training = np.load(f'{input_path}/class_weights_for_training.npy')
class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')
class_weights_for_test = np.load(f'{input_path}/class_weights_for_test.npy')

classes=["is_non_resonant_bkg", "is_ttH_bkg", "is_single_H_bkg", "is_GluGluToHH_sig"]

X_train[X_train == -999] = -9
X_val[X_val == -999] = -9
X_test[X_test == -999] = -9

# Use GPU if available
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
class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).to(device)
class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).to(device)
class_weights_for_test = torch.tensor(class_weights_for_test, dtype=torch.float32).to(device)



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

best_model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
best_scheduler = ReduceLROnPlateau(best_optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)


start_epoch, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_weights, best_loss = load_checkpoint(f'{path_to_checkpoint}/mlp.pth', best_model, best_optimizer, best_scheduler)

y_pred_val = np.load(f"{path_to_checkpoint}/y_pred_val.npy")
y_pred_val = torch.tensor(y_pred_val, dtype=torch.float32).to(device)
y_pred_np = np.load(f"{path_to_checkpoint}/y_pred_np.npy")
y_val_np = np.load(f"{path_to_checkpoint}/y_val_np.npy")


colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']

#plot loss function
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/loss_plot')
plt.clf()

#plot loss function
plt.plot(train_loss_hist_no_aboslute, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.savefig(f'{path_for_plots}/true_loss_plot')
plt.clf()

#plot accuracy
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(f'{path_for_plots}/acc_plot')
plt.clf()

#plot misidentified signal

with open(input_vars_path, 'r') as f:
    data = json.load(f)
var_names = data["mlp"]["vars"]

def get_var_index(var_name):
    return var_names.index(var_name)

"""with open('/home/home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm/mean_std_dict.pkl', 'rb') as f:
    mean_std_dict = pickle.load(f)

mean = mean_std_dict["mean"]
std = mean_std_dict["std_dev"]"""

var_list=["pt", "lead_eta", "sublead_pt", "sublead_eta", "lead_bjet_eta", "lead_bjet_phi",
            "lead_bjet_btagPNetB", "sublead_bjet_eta", "sublead_bjet_phi", "DeltaR_j1g1", "DeltaR_j2g1",
            "absCosThetaStar_gg", "n_leptons", "n_jets", "pholead_PtOverM", "phosublead_PtOverM",
            "dijet_pt", "chi_t0", "M_X", "DeltaPhi_j1MET", "DeltaPhi_j2MET", "VBF_first_jet_pt", "VBF_first_jet_eta", "VBF_first_jet_phi", "VBF_first_jet_mass", "VBF_first_jet_charge",
            "VBF_first_jet_btagPNetB", "VBF_second_jet_pt", "VBF_second_jet_eta", "VBF_second_jet_phi", "VBF_second_jet_mass", "VBF_second_jet_charge", "VBF_second_jet_btagPNetB", "VBF_dijet_pt", "VBF_dijet_eta", "VBF_dijet_phi",
            "VBF_dijet_mass", "VBF_dijet_charge", "VBF_first_jet_PtOverM", "VBF_second_jet_PtOverM", "VBF_first_jet_index", "VBF_second_jet_index"]

# , "VBF_DeltaR_jb_min", "VBF_DeltaR_jg_min", "VBF_Cgg", "VBF_Cbb"
#class_3_as_2_idx = (y_val_np == 3) & (y_pred_np == 2)
#class_2_as_3_idx = (y_val_np == 2) & (y_pred_np == 3)
#class_2_as_2_idx = (y_val_np == 2) & (y_pred_np == 2)
#class_3_as_3_idx = (y_val_np == 3) & (y_pred_np == 3)
#
#X_val_flt = X_val.clone()
#X_val_flt[X_val < -8] = -999
#X_val_unsc = (X_val_flt.cpu() * std) + mean


"""def plot_misidentified_signal(var_name, n_bins):
    index = get_var_index(var_name)
    var_class_3_as_2 = X_val_unsc[class_3_as_2_idx, index].cpu().numpy()
    var_class_3_as_2 = var_class_3_as_2[var_class_3_as_2 > -10]
    var_class_2_as_3 = X_val_unsc[class_2_as_3_idx, index].cpu().numpy()
    var_class_2_as_3 = var_class_2_as_3[var_class_2_as_3 > -10]
    var_class_2_as_2 = X_val_unsc[class_2_as_2_idx, index].cpu().numpy()
    var_class_2_as_2 = var_class_2_as_2[var_class_2_as_2 > -10]
    var_class_3_as_3 = X_val_unsc[class_3_as_3_idx, index].cpu().numpy()
    var_class_3_as_3 = var_class_3_as_3[var_class_3_as_3 > -10]

    if len(var_class_3_as_2) == 0 or len(var_class_2_as_3) == 0 or len(var_class_2_as_2) == 0 or len(var_class_3_as_3) == 0:
        print(f"Skipping {var_name} because one of the arrays is empty after filtering.")
        return index

    minx = int(min(var_class_3_as_2.min(), var_class_2_as_3.min(), var_class_2_as_2.min(), var_class_3_as_3.min()) -1)
    maxx = int(max(var_class_3_as_2.max(), var_class_2_as_3.max(), var_class_2_as_2.max(), var_class_3_as_3.max()) +1)
    plt.figure()
    binning = np.linspace(int(minx), int(maxx), int(n_bins))
    Hist, Edges = np.histogram(var_class_3_as_2, bins=binning, density=True)
    hep.histplot((Hist, Edges), histtype='step', color=colors[0], label=f'{classes[3]} predicted as {classes[2]}')
    Hist, Edges = np.histogram(var_class_2_as_3, bins=binning, density=True)
    hep.histplot((Hist, Edges), histtype='step', color=colors[1], label=f'{classes[2]} predicted as {classes[3]}')
    Hist, Edges = np.histogram(var_class_2_as_2, bins=binning, density=True)
    hep.histplot((Hist, Edges), histtype='step', color=colors[2], label=f'{classes[2]} correctly predicted as {classes[2]}')
    Hist, Edges = np.histogram(var_class_3_as_3, bins=binning, density=True)
    hep.histplot((Hist, Edges), histtype='step', color=colors[3], label=f'{classes[3]} correctly predicted as {classes[3]}')
    plt.xlabel(f"{var_name}")
    plt.ylabel("Events per bin")
    plt.xlim([int(minx), int(maxx)])
    plt.ylim(bottom=0)
    plt.legend(loc='lower center')
    hep.cms.label("Private work", data=False, year="2022", com=13.6)
    plt.savefig(f'{path_for_hists}/MisIdSignal_plot_{var_name}')
    plt.close()
    return index

for i in range(len(var_list)):
    plot_misidentified_signal(var_list[i], 41)"""

"""threshhold=0.5
mask = torch.max(y_pred_val, dim=1)[0] > threshhold
y_pred_flt = y_pred_np[mask.cpu().numpy()]
y_val_flt = y_val_np[mask.cpu().numpy()]
rel_w_val_np = class_weights_for_val.cpu().numpy()
rel_w_val_flt = rel_w_val_np[mask.cpu().numpy()]

# Plot confusion matrix
cm = confusion_matrix(y_val_flt, y_pred_flt, normalize='true', sample_weight=rel_w_val_flt)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='.2g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix, threshhold = {threshhold}')
plt.savefig(f'{path_for_plots}/cm_plot')
plt.clf()"""

#ROC one vs. all
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


with open(f"{out_path}/fpr_dict.pkl", 'wb') as f:
    pickle.dump(fpr, f)

with open(f"{out_path}/tpr_dict.pkl", 'wb') as f:
    pickle.dump(tpr, f)

# Plot ROC curves
plt.figure()
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

#ROC one vs. one
fpr = dict()
tpr = dict()
roc_auc = dict()
combinations_to_plot = [(2, 0), (2, 1), (3, 0), (3, 1)]

plt.figure()

for (i, j) in combinations_to_plot:
    # Extract the binary labels for classes i and j
    mask = np.logical_or(y_val_np == i, y_val_np == j)
    y_true_bin = y_val_bin[mask][:, [i, j]]
    y_scores = y_pred_val[mask][:, [i, j]].cpu().detach().numpy()
    
    # True labels: i -> 0, j -> 1
    y_true = np.argmax(y_true_bin, axis=1)
    y_score = y_scores[:, 1]  # Score for class j

    # Compute ROC curve and ROC area for this pair
    fpr[(i, j)], tpr[(i, j)], _ = roc_curve(y_true, y_score)
    roc_auc[(i, j)] = auc(fpr[(i, j)], tpr[(i, j)])
    
    # Plot the ROC curve
    plt.plot(fpr[(i, j)], tpr[(i, j)], lw=2,
             label=f'ROC curve {classes[i]} vs. {classes[j]} (area = {roc_auc[(i, j)]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Selected Class Pairs')
plt.legend(loc="lower right")
plt.savefig(f'{path_for_plots}/combined_roc_curves')
plt.clf()