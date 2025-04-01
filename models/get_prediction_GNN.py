import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm
import copy
from mlp import MLP
import torch.nn.functional as F
from GAT import GAT_classifier


def get_prediction(model_dict_path, model_path, X):

    #with open(model_dict_path, "r") as f:
    #    hyperparameters = json.load(f)
    # load model_dict_path
    with open(model_dict_path, 'r') as f:
        hyperparameters = json.load(f)

    input_dim = hyperparameters["input_dim"]
    hidden_dim = hyperparameters["hidden_dim"]
    num_init_mlp_layers = hyperparameters["num_init_mlp_layers"]
    num_final_mlp_layers = hyperparameters["num_final_mlp_layers"]
    num_gat_layers = hyperparameters["num_gat_layers"]
    num_message_passing_layers = hyperparameters["num_message_passing_layers"]
    num_extra_feat = hyperparameters["num_extra_feat"]
    output_dim = hyperparameters["output_dim"]
    activation = hyperparameters["activation"]
    seed = hyperparameters["seed"]
    lr = hyperparameters["lr"]
    dropout = hyperparameters["dropout"]
    weight_decay = hyperparameters["weight_decay"]
    n_epochs = hyperparameters["n_epochs"]
    best_loss = hyperparameters["best_loss"]

    model = GAT_classifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_init_mlp_layers=num_init_mlp_layers,
        num_final_mlp_layers=num_final_mlp_layers,
        num_gat_layers=num_gat_layers,
        num_message_passing_layers=num_message_passing_layers,
        num_extra_feat=num_extra_feat,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation
    )

    model.to(device)
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['model_state_dict'])


    y_pred_probs_list = []

    model.eval()
    batch_size = 1024
    with torch.no_grad():
        for data in X:
            data = data.to(device)
            y_pred = model(data)
            y_pred_probs = F.softmax(y_pred, dim=1).cpu().detach().numpy()

            y_pred_probs_list.append(y_pred_probs)

    y_pred_probs_list = np.concatenate(y_pred_probs_list, axis=0)

    return y_pred_probs_list

if __name__ == "__main__":
    model_folder = "train_gnn_inputs_20241203/training_500/"
    model_dict_path = f"{model_folder}/hyperparameters.json"
    model_path = f"{model_folder}/GAT.pth"


    samples_path = "../data/gnn_inputs_20241203_new/individual_samples/"
    samples = [
            "GGJets",
            "GJetPt20To40",
            "GJetPt40",
            "TTGG",
            "ttHtoGG_M_125",
            "GluGluHToGG_M_125",
            "VBFHToGG_M_125",
            "VHtoGG_M_125",
            "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00",
            "VBFHHto2B2G_CV_1_C2V_1_C3_1"
            ]

    for era in ["preEE", "postEE"]:
            for sample in samples:
                inputs_path = f"{samples_path}/{era}/{sample}"

                device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
                X = torch.load(f'{inputs_path}/data_lst.pt')

                print(f"Getting prediction for {sample} in {era} era")
                pred = get_prediction(model_dict_path, model_path, X)
                print(np.sum(pred, axis=1))
                # save the prediction
                print(f"Saving prediction for {sample} in {era} era \n")
                np.save(f"{samples_path}/{era}/{sample}/y.npy", pred)