import os
import torch
from models.mlp import MLP
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import yaml
from utils.device import get_torch_device

# DEPENDS ON:
#   <configs>/training_config.yaml
#   <model_folder>/params.json
#   <model_folder>/mlp.pth
#   if nominal:
#       <samples_path>/individual_samples/<era>/<sample>/X.npy
#    if syst:
#       <samples_path>/individual_samples/<era>/<sample>/<syst>/X.npy
# CREATES:
#   if nominal:
#       <samples_path>/individual_samples/<era>/<sample>/y.npy
#   if syst:
#       <samples_path>/individual_samples/<era>/<sample>/<syst>/y.npy


def get_prediction(model_dict_path, model_path, X):
    """Generate multi-class predictions.

    Supports passing either a torch.Tensor already on device or a numpy array / memmap.
    If a numpy array is provided, it will be sliced and transferred to the GPU in
    mini-batches, avoiding loading the entire feature matrix onto the device.
    """

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
    output_size = 4

    # Resolve device (use global if defined, else default heuristic)
    try:
        dev = device  # type: ignore[name-defined]
    except NameError:
        dev = get_torch_device()
    model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(dev)
    model.to(dev)
    model_state = torch.load(model_path, map_location=dev, weights_only=False)
    model.load_state_dict(model_state['model_state_dict'])

    model.eval()
    batch_size = int(os.getenv('PRED_BATCH_SIZE', '1024'))
    y_preds = []
    is_torch = isinstance(X, torch.Tensor)
    nrows = X.shape[0]
    for i in range(0, nrows, batch_size):
        if is_torch:
            X_batch = X[i:i + batch_size].to(dev)
        else:
            # numpy / memmap path: create tensor per chunk
            X_np = X[i:i + batch_size]
            X_batch = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(dev)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_preds.append(y_batch.cpu().numpy())
    
    y = np.concatenate(y_preds, axis=0)
    print(y.shape)

    return y

def get_prediction_binary(model_dict_path, model_path, X):
    """Generate binary predictions.

    Accepts torch.Tensor or numpy array / memmap and processes in mini-batches.
    """

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

    try:
        dev = device  # type: ignore[name-defined]
    except NameError:
        dev = get_torch_device()
    model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(dev)
    model.to(dev)
    model_state = torch.load(model_path, map_location=dev)
    model.load_state_dict(model_state['model_state_dict'])

    model.eval()
    batch_size = int(os.getenv('PRED_BATCH_SIZE', '1024'))
    y_preds = []
    is_torch = isinstance(X, torch.Tensor)
    nrows = X.shape[0]
    for i in range(0, nrows, batch_size):
        if is_torch:
            X_batch = X[i:i + batch_size].to(dev)
        else:
            X_np = X[i:i + batch_size]
            X_batch = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(dev)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = torch.sigmoid(y_batch)
            y_preds.append(y_batch.cpu().numpy())
    
    y = np.concatenate(y_preds, axis=0)
    print(y.shape)

    return y

def get_prediction_parquet(model_dict_path, model_path, X_path):

    # No need to stage full tensor on GPU; pass memmap to get_prediction
    X = np.load(X_path, mmap_mode='r')
    print(f"Getting prediction for {X_path}")
    pred = get_prediction(model_dict_path, model_path, X)
    print(np.sum(pred, axis=1))

    print(f"Saving prediction for {X_path} \n")
    output_path = X_path.replace("X.npy", "y.npy")
    np.save(output_path, pred)

    return pred

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--model_folder', type=str, help='Path to the model folder')
    parser.add_argument('--samples_path', type=str, help='Path to the samples')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--get_pred_nominal', action='store_true', help='Get predictions for nominal samples')
    parser.add_argument('--get_pred_sys', action='store_true', help='Get predictions for systematics samples')
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

    if args.get_pred_nominal:

        for era in eras:
            samples = training_config["samples_info"][era].keys()
            for sample in samples:
                inputs_path = f"{samples_path}/individual_samples/{era}/{sample}"

                device = get_torch_device(training_config.get("cuda_device"))
                #device = 'cpu'
                print("Device: ", device)
                # Memmap large feature matrix to avoid full GPU allocation
                X = np.load(f'{inputs_path}/X.npy', mmap_mode='r')

                print(f"Getting prediction for {sample} in {era} era")
                #pred = get_prediction(model_dict_path, model_path, X)
                pred = get_prediction(model_dict_path, model_path, X)
                print(np.sum(pred, axis=1))
                # save the prediction
                print(f"Saving prediction for {sample} in {era} era \n")
                np.save(f"{samples_path}/individual_samples/{era}/{sample}/y.npy", pred)

        data_samples = training_config["samples_info"]["data"].keys()
        for data_sample in data_samples:
            inputs_path = f"{samples_path}/individual_samples_data/{data_sample}"
            device = get_torch_device(training_config.get("cuda_device"))
            X = np.load(f'{inputs_path}/X.npy', mmap_mode='r')

            print(f"Getting prediction for {data_sample}")
            #pred = get_prediction(model_dict_path, model_path, X)
            pred = get_prediction(model_dict_path, model_path, X)
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

                    device = get_torch_device(training_config.get("cuda_device"))
                    #device = 'cpu'
                    print("Device: ", device)
                    X = np.load(f'{inputs_path}/X.npy', mmap_mode='r')

                    print(f"Getting prediction for {sample} in {era} era")
                    #pred = get_prediction(model_dict_path, model_path, X)
                    pred = get_prediction(model_dict_path, model_path, X)
                    print(np.sum(pred, axis=1))
                    # save the prediction
                    print(f"Saving prediction for {sample} in {era} era for {sys} \n")
                    np.save(f"{samples_path}/individual_samples/{era}/{sample}/{sys}/y.npy", pred)
