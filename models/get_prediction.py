import torch
from mlp import MLP
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F


def get_prediction(model_dict_path, model_path, X):

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

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--model_folder', type=str, help='Path to the model folder')
    parser.add_argument('--samples_path', type=str, help='Path to the samples')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    args = parser.parse_args()

    model_folder = args.model_folder
    model_dict_path = f"{model_folder}/params.json"
    model_path = f"{model_folder}/mlp.pth"
    
    # open config file
    with open(f"{args.config_path}/samples_and_classes.json", 'r') as f:
        samples_and_classes = json.load(f)

    samples_path = args.samples_path
    samples = samples_and_classes["sample_to_class_pred"].keys()

    for era in ["preEE", "postEE", "preBPix", "postBPix"]:
            for sample in samples:
                inputs_path = f"{samples_path}/individual_samples/{era}/{sample}"

                device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
                #device = 'cpu'
                print("Device: ", device)
                X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                print(f"Getting prediction for {sample} in {era} era")
                #pred = get_prediction(model_dict_path, model_path, X)
                pred = get_prediction(model_dict_path, model_path, X)
                print(np.sum(pred, axis=1))
                # save the prediction
                print(f"Saving prediction for {sample} in {era} era \n")
                np.save(f"{samples_path}/individual_samples/{era}/{sample}/y.npy", pred)

    data_samples = samples_and_classes["samples_data"].keys()
    for data_sample in data_samples:
        inputs_path = f"{samples_path}/individual_samples_data/{data_sample}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

        print(f"Getting prediction for {data_sample}")
        #pred = get_prediction(model_dict_path, model_path, X)
        pred = get_prediction(model_dict_path, model_path, X)
        print(np.sum(pred, axis=1))
        # save the prediction
        print(f"Saving prediction for {data_sample} \n")
        np.save(f"{samples_path}/individual_samples_data/{data_sample}/y.npy", pred)