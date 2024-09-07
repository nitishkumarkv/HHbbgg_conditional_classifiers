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
from optuna.samplers import RandomSampler
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline

from mlp import MLP

input_path = "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm/"

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

classes=["non resonant bkg", "ttH bkg", "GluGluToHH sig", "VBFToHH sig"]

X_train[X_train == -999] = -9
X_val[X_val == -999] = -9
X_test[X_test == -999] = -9

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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

# Optuna Optimierungsfunktion
def objective(trial):
    # define hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_nodes = trial.suggest_categorical('num_nodes', [128, 256, 512, 1024])
    act_fn_name = trial.suggest_categorical('act_fn', ['ReLU', 'ELU', 'SELU'])
    act_fn = getattr(nn, act_fn_name)
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.25)
    weight_increase = trial.suggest_categorical('weight_increase', [2, 5, 10, 20, 30])

    inc_class_weights_for_training = class_weights_for_training[y_train == [0, 0, 1, 0]]*weight_increase
    inc_class_weights_for_training = inc_class_weights_for_training[y_train == [0, 0, 0, 1]]*weight_increase

    # create model
    input_size = X_train.shape[1]
    output_size = 4
    model = MLP(input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob)
    model.to(device)

    # Loss function, optimizer und scheduler
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

    n_epochs = 300
    batch_size = 1024
    batches_per_epoch = len(X_train) // batch_size

    best_loss = np.inf
    patience = 75
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
            weighted_val_loss = (val_loss * class_weights_for_val).sum() / class_weights_for_val.sum()

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
n_trials = 40
study_name = "random_sampler_40"
storage = optuna.storages.RDBStorage('sqlite:///example.db')
study = optuna.create_study(storage=storage, sampler=RandomSampler(), study_name=study_name, direction='minimize')
study.optimize(objective, n_trials=n_trials)

path_for_plots = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/test_random_search/'

plt.rcParams.update({
    'font.size': 6,         # Schriftgröße für alle Texte
    'axes.titlesize': 6,    # Schriftgröße für Titel der Achsen
    'axes.labelsize': 6,    # Schriftgröße für Achsenbeschriftungen
    'xtick.labelsize': 6,    # Schriftgröße für x-Achsen-Ticks
    'ytick.labelsize': 6,    # Schriftgröße für y-Achsen-Ticks
    'legend.fontsize': 6,    # Schriftgröße für Legenden
    'figure.titlesize': 10   # Schriftgröße für den Figurentitel
})

#plots
plot_optimization_history(study)
plt.savefig(f'{path_for_plots}/history_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_parallel_coordinate(study)
plt.savefig(f'{path_for_plots}/parallel_coordinate_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_intermediate_values(study)
plt.savefig(f'{path_for_plots}/intermediate_values_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_param_importances(study)
plt.savefig(f'{path_for_plots}/param_importances_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_contour(study)
plt.savefig(f'{path_for_plots}/contour_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_slice(study)
plt.savefig(f'{path_for_plots}/slice_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_rank(study)
plt.savefig(f'{path_for_plots}/rank_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_edf(study)
plt.savefig(f'{path_for_plots}/edf_plot', dpi=500, bbox_inches="tight")
plt.clf()
plot_timeline(study)
plt.savefig(f'{path_for_plots}/timeline_plot', dpi=500, bbox_inches="tight")
plt.clf()

# use best hyper params
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

best_num_layers = best_params['num_layers']
best_num_nodes = best_params['num_nodes']
best_act_fn_name = best_params['act_fn']
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_dropout_prob = best_params['dropout_prob']
weight_increase = best_params['weight_increase']

# save all the numpy arrays
out_path='/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/test_random_search_params.json'
print("\n INFO: saving best hyperparameters")

best_params = {
    'num_layers': best_num_layers,
    'num_nodes': best_num_nodes,
    'act_fn_name': best_act_fn_name,
    'lr': best_lr,
    'weight_decay': best_weight_decay,
    'dropout_prob': best_dropout_prob,
    'weight_increase': weight_increase,
    'n_trials': n_trials
}

with open(out_path, 'w') as f:
    json.dump(best_params, f)