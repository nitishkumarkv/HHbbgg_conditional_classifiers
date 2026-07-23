import torch
from mlp import MLP
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import yaml

import pickle
import pyarrow.parquet as pq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.kfold_helper import get_fold_values


def get_prediction(model_dict_path, model_path, X, output_size=None):

    if isinstance(model_dict_path, str):
        with open(model_dict_path, 'r') as f:
            best_params = json.load(f)
            #print(best_params)
    else:
        best_params = model_dict_path

    best_num_layers = best_params['num_layers']
    best_num_nodes = best_params['num_nodes']
    best_act_fn_name = best_params['act_fn_name']
    best_act_fn = getattr(nn, best_act_fn_name)
    best_dropout_prob = best_params['dropout_prob']
    input_size = X.shape[1]

    # Determine output_size: priority order: parameter > best_params > infer from checkpoint
    model_state = None
    if output_size is None:
        if 'output_size' in best_params:
            output_size = best_params['output_size']
            print(f"Using output_size from best_params: {output_size}")
        else:
            # Infer from model checkpoint
            model_state = torch.load(model_path, weights_only=False)
            # Find the last linear layer in state dict
            for key in reversed(list(model_state['model_state_dict'].keys())):
                if 'weight' in key and 'layers' in key:
                    output_size = model_state['model_state_dict'][key].shape[0]
                    print(f"Inferred output_size from checkpoint: {output_size}")
                    break
            if output_size is None:
                raise ValueError("Could not determine output_size. Please provide it as a parameter.")
    else:
        print(f"Using provided output_size: {output_size}")

    model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    model.to(device)

    # Load model state (reuse if already loaded for inference)
    if model_state is None:
        model_state = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_state['model_state_dict'])

    model.eval()
    batch_size = 1024
    y_preds = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_preds.append(y_batch.cpu().numpy())
    
    y = np.concatenate(y_preds, axis=0)
    print(y.shape)

    return y

def get_prediction_binary(model_dict_path, model_path, X):

    if isinstance(model_dict_path, str):
        with open(model_dict_path, 'r') as f:
            best_params = json.load(f)
            #print(best_params)
    else:
        best_params = model_dict_path

    best_num_layers = best_params['num_layers']
    best_num_nodes = best_params['num_nodes']
    best_act_fn_name = best_params['act_fn_name']
    best_act_fn = getattr(nn, best_act_fn_name)
    best_dropout_prob = best_params['dropout_prob']
    input_size = X.shape[1]
    output_size = 1

    model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    model.to(device)
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['model_state_dict'])

    model.eval()
    batch_size = 1024
    y_preds = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = torch.sigmoid(y_batch)
            y_preds.append(y_batch.cpu().numpy())
    
    y = np.concatenate(y_preds, axis=0)
    print(y.shape)

    return y

def get_prediction_parquet(model_dict_path, model_path, X_path, output_size=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(np.load(X_path), dtype=torch.float32).to(device)
    print(f"Getting prediction for {X_path}")
    pred = get_prediction(model_dict_path, model_path, X, output_size=output_size)
    print(np.sum(pred, axis=1))

    print(f"Saving prediction for {X_path} \n")
    output_path = X_path.replace("X.npy", "y.npy")
    np.save(output_path, pred)

    return pred

def get_prediction_kfold(samples_path, model_base, k_folds, output_size,
                         eras, config, ext='', from_parquet=False):
    device = torch.device('cuda:'+config["cuda_device"] if torch.cuda.is_available() else 'cpu')

    fill_nan = -9

    if from_parquet:
        with open(os.path.join(model_base, "input_vars.txt"), 'r') as f:
            feature_names = json.load(f)
        with open(os.path.join(model_base, "mean_std_dict.pkl"), 'rb') as f:
            ms = pickle.load(f)
        mean = np.asarray(ms["mean"], dtype=np.float32)
        std = np.asarray(ms["std_dev"], dtype=np.float32)

    hparams_path = os.path.join(model_base, "fold_0", "after_random_search_best1", "params.json")
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)

    best_num_layers = hparams['num_layers']
    best_num_nodes = hparams['num_nodes']
    best_act_fn = getattr(nn, hparams['act_fn_name'])
    best_dropout_prob = hparams['dropout_prob']

    fold_models = {}
    for fi in range(k_folds):
        ckpt_path = os.path.join(model_base, f"fold_{fi}", "after_random_search_best1", "mlp.pth")
        model_state = torch.load(ckpt_path, weights_only=False, map_location=device)
        in_dim = model_state["model_state_dict"]["model.0.weight"].shape[1]
        model = MLP(in_dim, best_num_layers, best_num_nodes, output_size,
                    best_act_fn, best_dropout_prob).to(device)
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        fold_models[fi] = model
        print(f"  loaded fold_{fi} model")

    if from_parquet:
        def load_X_and_events(pq_path):
            table = pq.read_table(pq_path, columns=feature_names + ["event"])
            n = table.num_rows
            X = np.empty((n, len(feature_names)), dtype=np.float32)
            for i, name in enumerate(feature_names):
                X[:, i] = table.column(name).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            mask = (X < -998.0)
            X[mask] = np.nan
            X = (X - mean) / std
            X = np.nan_to_num(X, nan=fill_nan)
            event_raw = table.column("event").to_numpy(zero_copy_only=False)
            return X, event_raw
    else:
        def load_X_and_events(sample_dir):
            pq_path = f"{sample_dir}/events.parquet"
            X = np.load(f"{sample_dir}/X.npy")
            table = pq.read_table(pq_path, columns=["event"])
            event_raw = table.column("event").to_numpy(zero_copy_only=False)
            return X, event_raw

    def infer(model, X_batch):
        with torch.no_grad():
            yb = F.softmax(model(X_batch), dim=1)
            return yb.cpu().numpy()

    batch_size = 65536

    for era in eras:
        samples = config["samples_info"][era].keys()
        for sample in samples:
            sample_dir = f"{samples_path}/individual_samples/{era}/{sample}"
            pq_path = f"{sample_dir}/events.parquet"

            if not os.path.exists(pq_path):
                print(f"  Skipping {era}/{sample}: missing events.parquet")
                continue
            if not from_parquet and not os.path.exists(f"{sample_dir}/X.npy"):
                print(f"  Skipping {era}/{sample}: missing X.npy")
                continue

            X, event_raw = load_X_and_events(sample_dir if not from_parquet else pq_path)
            fold_assignment = get_fold_values(event_raw, k_folds)

            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            y_full = np.empty((X.shape[0], output_size), dtype=np.float32)
            for fi in range(k_folds):
                mask = (fold_assignment == fi)
                if mask.sum() == 0:
                    continue
                y_full[mask] = infer(fold_models[fi], X_t[mask])

            out_path = f"{sample_dir}/y{ext}.npy"
            np.save(out_path, y_full)
            print(f"  {era}/{sample}: {X.shape[0]} events -> {out_path}")
            print(f"    fold_counts: {[(fi, int((fold_assignment == fi).sum())) for fi in range(k_folds)]}")

    data_eras = config["samples_info"]["data"].keys()
    for era in data_eras:
        sample_dir = f"{samples_path}/individual_samples_data/{era}"
        pq_path = f"{sample_dir}/events.parquet"

        if not os.path.exists(pq_path):
            print(f"  Skipping data {era}: missing events.parquet")
            continue
        if not from_parquet and not os.path.exists(f"{sample_dir}/X.npy"):
            print(f"  Skipping data {era}: missing X.npy")
            continue

        X, event_raw = load_X_and_events(sample_dir if not from_parquet else pq_path)
        fold_assignment = get_fold_values(event_raw, k_folds)

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_full = np.empty((X.shape[0], output_size), dtype=np.float32)
        for fi in range(k_folds):
            mask = (fold_assignment == fi)
            if mask.sum() == 0:
                continue
            y_full[mask] = infer(fold_models[fi], X_t[mask])

        out_path = f"{sample_dir}/y{ext}.npy"
        np.save(out_path, y_full)
        print(f"  data/{era}: {X.shape[0]} events -> {out_path}")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--model_folder', type=str, help='Path to the model folder')
    parser.add_argument('--samples_path', type=str, help='Path to the samples')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--get_pred_nominal', action='store_true', help='Get predictions for nominal samples')
    parser.add_argument('--get_pred_sys', action='store_true', help='Get predictions for systematics samples')
    parser.add_argument('--k-folds', type=int, default=0, help='Number of folds for k-fold prediction (0 = single model)')
    parser.add_argument('--from-parquet', action='store_true', help='Read features from events.parquet instead of pre-built X.npy')
    args = parser.parse_args()

    model_folder = args.model_folder
    model_dict_path = f"{model_folder}/params.json"
    model_path = f"{model_folder}/mlp.pth"

    # load the configuration yaml files
    training_config_path = f"{args.config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    samples_path = args.samples_path
    eras = training_config["samples_info"]["eras"]

    # Get output_size from training config
    output_size = len(training_config["classes"])
    print(f"Number of output classes from training config: {output_size}")

    k_folds = args.k_folds

    if k_folds > 1:
        mode = "parquet" if args.from_parquet else "X.npy"
        print(f"INFO: Running k-fold prediction (K={k_folds}), input mode: {mode}")
        get_prediction_kfold(samples_path, args.model_folder, k_folds,
                             output_size, eras, training_config,
                             ext='', from_parquet=args.from_parquet)
        if args.get_pred_sys:
            get_prediction_kfold(samples_path, args.model_folder, k_folds,
                                 output_size, eras, training_config,
                                 ext='_sys', from_parquet=args.from_parquet)
        exit(0)

    if args.get_pred_nominal:

        for era in eras:
            samples = training_config["samples_info"][era].keys()
            for sample in samples:
                inputs_path = f"{samples_path}/individual_samples/{era}/{sample}"

                device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
                #device = 'cpu'
                print("Device: ", device)
                X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                print(f"Getting prediction for {sample} in {era} era")
                pred = get_prediction(model_dict_path, model_path, X, output_size=output_size)
                print(np.sum(pred, axis=1))
                # save the prediction
                print(f"Saving prediction for {sample} in {era} era \n")
                np.save(f"{samples_path}/individual_samples/{era}/{sample}/y.npy", pred)

        data_samples = training_config["samples_info"]["data"].keys()
        for data_sample in data_samples:
            inputs_path = f"{samples_path}/individual_samples_data/{data_sample}"
            device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
            X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

            print(f"Getting prediction for {data_sample}")
            pred = get_prediction(model_dict_path, model_path, X, output_size=output_size)
            print(np.sum(pred, axis=1))
            # save the prediction
            print(f"Saving prediction for {data_sample} \n")
            np.save(f"{samples_path}/individual_samples_data/{data_sample}/y.npy", pred)

    elif args.get_pred_sys:
        for era in eras:   
            samples = training_config["samples_info"][era].keys()
            for sample in samples:
                if sample in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
                    continue
                for sys in training_config["systematics"]:
                    inputs_path = f"{samples_path}/individual_samples/{era}/{sample}/{sys}/"

                    device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
                    #device = 'cpu'
                    print("Device: ", device)
                    X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                    print(f"Getting prediction for {sample} in {era} era")
                    pred = get_prediction(model_dict_path, model_path, X, output_size=output_size)
                    print(np.sum(pred, axis=1))
                    # save the prediction
                    print(f"Saving prediction for {sample} in {era} era for {sys} \n")
                    np.save(f"{samples_path}/individual_samples/{era}/{sample}/{sys}/y.npy", pred)