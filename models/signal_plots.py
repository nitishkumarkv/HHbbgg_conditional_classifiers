import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import pickle

input_path="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm/"

X_val = np.load(f'{input_path}/X_val.npy')
class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')

# Use GPU if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('INFO: Used device is', device)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).to(device)

path_to_checkpoint = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/mlp_checkpoint/"
y_pred_val = np.load(f"{path_to_checkpoint}/y_pred_val.npy") # model(X_val)
y_pred_val = torch.tensor(y_pred_val, dtype=torch.float32).to(device)
y_pred_np = np.load(f"{path_to_checkpoint}/y_pred_np.npy") # predicted class number
y_val_np = np.load(f"{path_to_checkpoint}/y_val_np.npy") # true class number

# apply threshhold and create tensors for all events that pass the threshhold
threshhold=0.5
mask = torch.max(y_pred_val, dim=1)[0] > threshhold #create mask
y_pred_flt = y_pred_np[mask.cpu().numpy()] #predictions that pass the threshhold
y_val_flt = y_val_np[mask.cpu().numpy()] #labels
rel_w_val_np = class_weights_for_val.cpu().numpy()
rel_w_val_flt = rel_w_val_np[mask.cpu().numpy()] #weight for each event that passes the threshhold

# sum of weights for each class for events that pass the threshhold 
sum_of_weights_ggF_flt = rel_w_val_flt[y_val_flt == 2].sum()
sum_of_weights_VBF_flt = rel_w_val_flt[y_val_flt == 3].sum()

# load input variables
input_vars_path = "/home/home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/input_variables.json"
with open(input_vars_path, 'r') as f:
    data = json.load(f)
var_names = data["mlp"]["vars"]

# load mean and std used for standradization
with open('/home/home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm/mean_std_dict.pkl', 'rb') as f:
    mean_std_dict = pickle.load(f)
mean = mean_std_dict["mean"]
std = mean_std_dict["std_dev"]

#load labels for x-axis
with open("/home/home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/labels.json", 'r') as f:
    labels = json.load(f)

# remove standardization
X_val_unsc = (X_val.cpu() * std) + mean
X_val_unsc[X_val < -8] = -999
X_val_flt = X_val_unsc[mask.cpu().numpy()]

# mask for class indices
class_3_as_2_idx = (y_val_flt == 3) & (y_pred_flt == 2)
class_2_as_3_idx = (y_val_flt == 2) & (y_pred_flt == 3)

path_for_hists="/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/misid_signal_plots/"

colors = ['royalblue', 'darkorange']
classes=["GluGluToHH", "VBFToHH"]

def plot_misidentified_signal(var_name, n_bins):
    index = var_names.index(var_name)
    var_class_3_as_2 = X_val_flt[class_3_as_2_idx, index].cpu().numpy()
    var_class_2_as_3 = X_val_flt[class_2_as_3_idx, index].cpu().numpy()

    # do not plot -999 values
    mask_class_3_as_2 = var_class_3_as_2 > -10
    mask_class_2_as_3 = var_class_2_as_3 > -10

    var_class_3_as_2 = var_class_3_as_2[mask_class_3_as_2]
    weight_class_3_as_2 = rel_w_val_flt[class_3_as_2_idx][mask_class_3_as_2]/sum_of_weights_ggF_flt
    
    var_class_2_as_3 = var_class_2_as_3[mask_class_2_as_3]
    weight_class_2_as_3 = rel_w_val_flt[class_2_as_3_idx][mask_class_2_as_3]/sum_of_weights_VBF_flt

    if len(var_class_3_as_2) == 0 or len(var_class_2_as_3) == 0:
        print(f"Skipping {var_name} because one of the arrays is empty after filtering.")
        return index
    
    minx = min(var_class_3_as_2.min(), var_class_2_as_3.min())-0.1
    maxx = max(var_class_3_as_2.max(), var_class_2_as_3.max())+0.1
    plt.figure()
    plt.figure(figsize=(10, 10))
    binning = np.linspace(int(minx), int(maxx), int(n_bins))
    Hist, Edges = np.histogram(var_class_3_as_2, bins=binning, weights=weight_class_3_as_2)
    hep.histplot((Hist, Edges), histtype='step', color=colors[0], label=f'{classes[1]} predicted as {classes[0]}')
    Hist, Edges = np.histogram(var_class_2_as_3, bins=binning, weights=weight_class_2_as_3)
    hep.histplot((Hist, Edges), histtype='step', color=colors[1], label=f'{classes[0]} predicted as {classes[1]}')
    plt.xlabel(labels[var_name])
    plt.ylabel("Events per bin")
    plt.xlim([minx, maxx])
    plt.yscale("log")
    current_ylim = plt.gca().get_ylim()  
    plt.ylim([current_ylim[0], current_ylim[1] * 1.2]) 

    plt.ylim(bottom=0)
    plt.legend()
    hep.cms.label("Private work", data=False, year="2022", com=13.6)
    plt.savefig(f'{path_for_hists}/MisIdSignal_plot_{var_name}')
    plt.close()
    return index

for i in range(len(var_names)):
    plot_misidentified_signal(var_names[i], 41)
