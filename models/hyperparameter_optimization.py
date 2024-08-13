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
from mlp import MLP

# the MLP model is already imported
# define all other things that are required for training:
# take the inputs, apply standardization, separate and randomize inputs into training/test sets, optimization function, loss function, backpropagation and weight updation, etc.

X_train = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/X_train.npy')
X_val = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/X_val.npy')
X_test = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/X_test.npy')
y_train = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/y_train.npy')
y_val = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/y_val.npy')
y_test = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/y_test.npy')
rel_w_train = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/rel_w_train.npy')
rel_w_val = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/rel_w_val.npy')
rel_w_test = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/rel_w_test.npy')
inc_rel_w_train = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/inc_rel_w_train.npy')
inc_rel_w_val = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/inc_rel_w_val.npy')
inc_rel_w_test = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/inc_rel_w_test.npy')
class_weights_for_training = np.load('/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp/class_weights_for_training.npy')

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

X_train[X_train == -999] = -9
X_val[X_val == -999] = -9
X_test[X_test == -999] = -9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Optuna Optimierungsfunktion
def objective(trial):
    # Hyperparameter definieren
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_nodes = trial.suggest_categorical('num_nodes', [128, 256, 512, 1024])
    act_fn_name = trial.suggest_categorical('act_fn', ['ReLU', 'ELU', 'SELU'])
    act_fn = getattr(nn, act_fn_name)
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.5)

    # Modell erstellen
    input_size = X_train.shape[1]
    output_size = 4
    model = MLP(input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob)
    model.to(device)

    # Loss function, optimizer und scheduler
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

    n_epochs = 250
    batch_size = 1024
    batches_per_epoch = len(X_train) // batch_size

    best_loss = np.inf
    patience = 30
    counter = 0

    for epoch in range(n_epochs):
        model.train()
        for i in range(batches_per_epoch):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            weights_batch = class_weights_for_training[start:end]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            weighted_loss = (loss * weights_batch).sum() / weights_batch.sum()
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            val_loss = loss_fn(y_pred, y_val)
            weighted_val_loss = (val_loss * rel_w_val).sum() / rel_w_val.sum()

        scheduler.step(weighted_val_loss)
        ce = float(weighted_val_loss.cpu().detach().numpy())

        if ce < best_loss:
            best_loss = ce
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    return best_loss

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)

# use best hyper params
best_params = study.best_params
print(f"Beste Hyperparameter: {best_params}")

best_num_layers = best_params['num_layers']
best_num_nodes = best_params['num_nodes']
best_act_fn_name = best_params['act_fn']
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_dropout_prob = best_params['dropout_prob']

# save all the numpy arrays
out_path='/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/best_hyperparams.json'
print("\n INFO: saving best hyperparameters")

best_params = {
    'num_layers': best_num_layers,
    'num_nodes': best_num_nodes,
    'act_fn_name': best_act_fn_name,
    'lr': best_lr,
    'weight_decay': best_weight_decay,
    'dropout_prob': best_dropout_prob
}

with open(out_path, 'w') as f:
    json.dump(best_params, f)