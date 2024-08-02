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
from sklearn.preprocessing import StandardScaler

input_file = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/dummy_input_for_mlp.parquet'
inputs = ak.from_parquet(input_file)

# load variables from json file
input_var_json = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/input_variables.json"
with open(input_var_json) as f:
    vars_for_training = json.load(f)["mlp"]

n_inputs=len(vars_for_training)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('INFO: Used device is', device)
#model.to(device)

# Stack the processed variables into a feature matrix
classes=["is_non_resonant_bkg", "is_ttH_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]
X = np.column_stack([ak.to_numpy(inputs[var]) for var in vars_for_training])
y = np.column_stack([ak.to_numpy(inputs[cls]) for cls in classes])

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

def standardize_tensors(X_train, X_val, X_test, threshold=-998, device='cpu'):
    def standardize_single_tensor(X_tensor, scaler=None, fit=False):
        #convert to numpy 
        X_np = X_tensor.cpu().numpy()
        # create mask
        valid_mask = X_np > threshold
        #copy array
        X_for_stnd = np.copy(X_np)
        #replace default values with nan
        X_for_stnd[~valid_mask] = np.nan
        #standardize each column
        mean_train=[]
        std_train=[]
        mean_val_test=[]
        std_val_test=[]
        for col in range(X_np.shape[1]):
            #prepare column
            col_data = X_for_stnd[:, col].reshape(-1, 1)
            valid_col_data = col_data[~np.isnan(col_data)]
            #standrardize
            if fit: #use fit_transform
                scaler[col] = StandardScaler()
                col_data_scaled = scaler[col].fit_transform(valid_col_data.reshape(-1, 1)).flatten()
                mean_train.append(np.mean(col_data_scaled))
                std_train.append(np.std(col_data_scaled))
            else: #only X_train is fitted
                col_data_scaled = scaler[col].transform(valid_col_data.reshape(-1, 1)).flatten()
                mean_val_test.append(np.mean(col_data_scaled))
                std_val_test.append(np.std(col_data_scaled))

            X_for_stnd[~np.isnan(X_for_stnd[:, col]), col] = col_data_scaled

        #if mean_train:
            #print('INFO: X_train mean=', mean_train, 'X_train std=', std_train)
        #else:
            #print('INFO: mean=', mean_val_test,  'std =', std_val_test)

        #convert array to original shape and create tensor
        X_scaled = np.where(np.isnan(X_for_stnd), -999, X_for_stnd)
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(X_tensor.device)
        return X_scaled_tensor, scaler
    
    scaler={}

    # Standardize training data with fit_transform
    X_train_scaled, scaler = standardize_single_tensor(X_train, scaler, fit=True)

    # Standardize validation and test data with transform
    X_val_scaled, _ = standardize_single_tensor(X_val, scaler, fit=False)
    X_test_scaled, _ = standardize_single_tensor(X_test, scaler, fit=False)

    return X_train_scaled, X_val_scaled, X_test_scaled

#seperate and randomize
X_train, X_test_val, y_train, y_test_val = train_test_split(X_tensor.cpu().numpy(), y_tensor.cpu().numpy(), train_size=0.6, shuffle=True, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5, shuffle=True, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

X_train, X_val, X_test = standardize_tensors(X_train, X_val, X_test)

#calculate weights
classes=np.unique(y, axis=0)
rev_counts=[]
for entry in classes:
    count=np.sum(np.all(y == entry, axis=1))
    rev_counts.append(count)
class_counts=rev_counts[::-1]

weights=[]
for count in class_counts:
    weights.append(1/count)
weights=torch.tensor(weights, dtype=torch.float32).to(device)

save_path = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/tensors_for_training"

torch.save({
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'n_inputs': n_inputs,
    'weights': weights,
}, save_path)