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
import os
import yaml

from mlp import MLP


# Optuna Optimierungsfunktion
def objective(trial):
    # define hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_nodes = trial.suggest_categorical('num_nodes', [128, 256, 512, 1024])
    act_fn_name = trial.suggest_categorical('act_fn_name', ['ReLU', 'ELU', 'SELU'])
    act_fn = getattr(nn, act_fn_name)
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    dropout_prob = trial.suggest_categorical('dropout_prob', [0.15, 0.2, 0.3, 0.4, 0.5])

    # create model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLP(input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob)
    model.to(device)

    # Loss function, optimizer und scheduler
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    n_epochs = 500
    batch_size = 1024
    batches_per_epoch = len(X_train) // batch_size

    best_loss = np.inf
    patience = 25
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




def perform_random_search(input_path, ntrial = 1):

    path_for_plots = f'random_search_{input_path.split("/")[-2]}'
    os.makedirs(path_for_plots, exist_ok=True)

    X_train = np.load(f'{input_path}/X_train.npy')
    X_val = np.load(f'{input_path}/X_val.npy')
    ##X_test = np.load(f'{input_path}/X_test.npy')
    y_train = np.load(f'{input_path}/y_train.npy')
    y_val = np.load(f'{input_path}/y_val.npy')
    #y_test = np.load(f'{input_path}/y_test.npy')
    rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
    rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
    #rel_w_test = np.load(f'{input_path}/rel_w_test.npy')
    class_weights_for_training = np.load(f'{input_path}/class_weights_for_training.npy')
    class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')
    #class_weights_for_test = np.load(f'{input_path}/class_weights_for_test.npy')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    #X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    #y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    rel_w_train = torch.tensor(rel_w_train, dtype=torch.float32).to(device)
    #rel_w_test = torch.tensor(rel_w_test, dtype=torch.float32).to(device)
    rel_w_val = torch.tensor(rel_w_val, dtype=torch.float32).to(device)
    class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).to(device)
    class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).to(device)

    n_trials = ntrial
    study_name = f"{path_for_plots}_{n_trials}_trials"
    storage = optuna.storages.RDBStorage('sqlite:///example.db')
    study = optuna.create_study(storage=storage, sampler=RandomSampler(), study_name=study_name, direction='minimize')
    study.optimize(objective, n_trials=n_trials)

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
    best_act_fn_name = best_params['act_fn_name']
    best_lr = best_params['lr']
    best_weight_decay = best_params['weight_decay']
    best_dropout_prob = best_params['dropout_prob']

    # save all the numpy arrays
    print("\n INFO: saving best hyperparameters")

    best_params = {
        'num_layers': best_num_layers,
        'num_nodes': best_num_nodes,
        'act_fn_name': best_act_fn_name,
        'lr': best_lr,
        'weight_decay': best_weight_decay,
        'dropout_prob': best_dropout_prob,
        'n_trials': n_trials
    }

    with open(f"{path_for_plots}/best_params.json", 'w') as f:
        json.dump(best_params, f)

# run for main
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--input_path', type=str, help='Path to the input files')
    parser.add_argument('--training_config_path', type=str, default=10, help='Training configuration path')
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    n_trials = training_config["num_random_search"]
    weight_scheme = training_config["weight_scheme"]

    input_path= args.input_path
    path_for_plots = f'{input_path}/random_search_1/'
    print(f"INFO: Saving plots to {path_for_plots}")
    os.makedirs(path_for_plots, exist_ok=True)

    X_train = np.load(f'{input_path}/X_train.npy')
    X_val = np.load(f'{input_path}/X_val.npy')
    ##X_test = np.load(f'{input_path}/X_test.npy')
    y_train = np.load(f'{input_path}/y_train.npy')
    y_val = np.load(f'{input_path}/y_val.npy')
    #y_test = np.load(f'{input_path}/y_test.npy')
    rel_w_train = np.load(f'{input_path}/rel_w_train.npy')
    rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
    #rel_w_test = np.load(f'{input_path}/rel_w_test.npy')

    # set weight scheme for training
    if weight_scheme == "weighted_abs":
        class_weights_for_training = np.load(f'{input_path}/class_weights_for_training_abs.npy')

    elif weight_scheme == "weighted_only_positive":
        class_weights_for_training = np.load(f'{input_path}/class_weights_only_positive.npy')

    elif weight_scheme == "weighted_CRUW_abs":
        # to be completed
        pass

    elif weight_scheme == "weighted_CRUW_only_positive":
        # to be completed
        pass

    class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')
    #class_weights_for_test = np.load(f'{input_path}/class_weights_for_test.npy')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    #X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    #y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    rel_w_train = torch.tensor(rel_w_train, dtype=torch.float32).to(device)
    #rel_w_test = torch.tensor(rel_w_test, dtype=torch.float32).to(device)
    rel_w_val = torch.tensor(rel_w_val, dtype=torch.float32).to(device)
    class_weights_for_training = torch.tensor(class_weights_for_training, dtype=torch.float32).to(device)
    class_weights_for_val = torch.tensor(class_weights_for_val, dtype=torch.float32).to(device)
    #class_weights_for_test = torch.tensor(class_weights_for_test, dtype=torch.float32).to(device)

    # Optuna study
    study_name = f"{path_for_plots}_{n_trials}_trials____"
    storage = optuna.storages.RDBStorage('sqlite:///example.db')
    study = optuna.create_study(storage=storage, sampler=RandomSampler(), study_name=study_name, direction='minimize')
    study.optimize(objective, n_trials=n_trials)



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
    best_act_fn_name = best_params['act_fn_name']
    best_lr = best_params['lr']
    best_weight_decay = best_params['weight_decay']
    best_dropout_prob = best_params['dropout_prob']

    # save all the numpy arrays
    print("\n INFO: saving best hyperparameters")

    best_params = {
        'num_layers': best_num_layers,
        'num_nodes': best_num_nodes,
        'act_fn_name': best_act_fn_name,
        'lr': best_lr,
        'weight_decay': best_weight_decay,
        'dropout_prob': best_dropout_prob,
        'n_trials': n_trials
    }

    with open(f"{path_for_plots}/best_params.json", 'w') as f:
        json.dump(best_params, f)

