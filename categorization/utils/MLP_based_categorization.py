import torch
import torch.nn as nn

import awkward as ak
import numpy as np
import json
import os
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import yaml
from tqdm.auto import tqdm

    
class MLP(nn.Module):
    def __init__(self, input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, num_nodes))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(act_fn())
            layers.append(nn.Dropout(p=dropout_prob))

        # Output layer
        layers.append(nn.Linear(num_nodes, output_size+1))
        #layers.append(nn.Softmax(dim=1))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)
    

class CustomDataset(Dataset):
    def __init__(self, X, extra_feat):
        self.X = X
        self.extra_feat = extra_feat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.extra_feat[idx]


def prep_inputs(path, sample_list, outpath):
    if not os.path.exists(f"{outpath}/X_train.npy"):
        comb_scores = []
        combe_extra_feat = []

        os.makedirs(outpath, exist_ok=True)

        for era in ["preEE", "postEE"]:
            for sample in sample_list:
                print(f"Processing {sample} in {era}")
                inputs_path = f"{path}/{era}/{sample}"
                
                # load scores
                scores = np.load(f"{inputs_path}/y_after_random_search_best1.npy")

                events = ak.from_parquet(f"{inputs_path}/events.parquet")
                diphoton_mass = np.array(events["mass"])
                dijet_mass = np.array(events["nonRes_dijet_mass"])
                is_signal = np.ones_like(diphoton_mass) if "GluGlutoHHto2B2G" in sample else np.zeros_like(diphoton_mass)
                sample_weights = np.load(f"{inputs_path}/rel_w.npy")
                extra_feat = np.concatenate([sample_weights.reshape(-1, 1), diphoton_mass.reshape(-1, 1), dijet_mass.reshape(-1, 1), is_signal.reshape(-1, 1)], axis=1)

                comb_scores.append(scores)
                combe_extra_feat.append(extra_feat)

        comb_scores = np.concatenate(comb_scores, axis=0)
        combe_extra_feat = np.concatenate(combe_extra_feat, axis=0)

        # print sum of total background and signal weights under the peak
        print(f"Total background weight under the peak: {combe_extra_feat[(combe_extra_feat[:, 3] == 0) & (combe_extra_feat[:, 1] > 120) & (combe_extra_feat[:, 1] <130)][:, 0].sum()}")
        print(f"Total signal weight under the peak: {combe_extra_feat[(combe_extra_feat[:, 3] == 1) & (combe_extra_feat[:, 1] > 120) & (combe_extra_feat[:, 1] <130)][:, 0].sum()}")

        # train and validation split
        scores_train, scores_val, extra_feat_train, extra_feat_val = train_test_split(comb_scores, combe_extra_feat, test_size=0.3, random_state=42)

        # standardize the scores and save the mean and std
        # mean = np.mean(scores_train, axis=0)
        # std = np.std(scores_train, axis=0)
        # scores_train = (scores_train - mean) / std
        # scores_val = (scores_val - mean) / std

        np.save(f"{outpath}/X_train.npy", scores_train)
        np.save(f"{outpath}/X_val.npy", scores_val)
        np.save(f"{outpath}/extra_feat_train.npy", extra_feat_train)
        np.save(f"{outpath}/extra_feat_val.npy", extra_feat_val)
        # np.save(f"{outpath}/mean.npy", mean)
        # np.save(f"{outpath}/std.npy", std)

    return 0

def asymptotic_significance(sig_weights, bkg_weights):
    s = torch.sum(sig_weights, dim=0)
    b = torch.sum(bkg_weights, dim=0)

    #Z = torch.sqrt(2 * ((s + b) * torch.log(1 + s/b) - s))
    Z = s/torch.sqrt(b + 1e-10)
    #Z = (2 * ((s + b) * torch.log(1 + s/b) - s))
    #print("Z: ", Z)

    return Z


def significance_loss(y_pred, extra_feat):

    # clamp the output to avoid log(0)
    y_pred = torch.clamp(y_pred, 1e-6, 1-1e-6)
    

    sig_under_the_mass_peak_mask = ((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 1)).reshape(-1, 1)
    bkg_under_the_mass_peak_mask = ((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 0)).reshape(-1, 1)
    bkg_mask = (extra_feat[:, 3] == 0).reshape(-1, 1)

    # get signal category loss
    sig_weights_for_significance = extra_feat[:, 0].reshape(-1, 1) * sig_under_the_mass_peak_mask * y_pred[:, :-1]
    bkg_weights_for_significance = extra_feat[:, 0].reshape(-1, 1) * bkg_under_the_mass_peak_mask * y_pred[:, :-1]

    #print("y_pred for signal under mass peak", y_pred[((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 1))])
    #print("y_pred for bkg under mass peak", y_pred[((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 0))])

    # scale the weights to the weights of total sample
    #sig_weights_for_significance = 0.12256 * sig_weights_for_significance / torch.sum(sig_weights_for_significance, dim=0)
    #bkg_weights_for_significance = 6016.005 * bkg_weights_for_significance / torch.sum(bkg_weights_for_significance, dim=0)

    #print("sig_weights_for_significance", sig_weights_for_significance)
    #print("bkg_weights_for_significance", bkg_weights_for_significance)
    # print("sum extra_feat[:, 0].reshape(-1, 1)", torch.sum(extra_feat[:, 0].reshape(-1, 1), dim=0))

    # print()
    #print("y_pred under peak signal", torch.mean(y_pred[((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 1))], dim=0))
    #print("y_pred bkg: ", torch.mean(y_pred[((extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, 3] == 0))], dim=0))
    # print()
    # print("sum sig_weights_for_significance", torch.sum(sig_weights_for_significance, dim=0))
    # print("sum bkg_weights_for_significance", torch.sum(bkg_weights_for_significance, dim=0))

    Z = asymptotic_significance(sig_weights_for_significance, bkg_weights_for_significance)
    # print("Z", Z)
    Z_sum_quad = torch.sqrt(torch.sum(Z**2))
    # print("Z_sum_quad", Z_sum_quad)

    # get background category loss by counting the amount of signal in it
    sig_in_bkg_category = torch.sum(extra_feat[:, 0].reshape(-1, 1) * sig_under_the_mass_peak_mask * y_pred[:, -1].reshape(-1, 1))
    bkg_in_bkg_category = torch.sum(extra_feat[:, 0].reshape(-1, 1) * bkg_mask * y_pred[:, -1].reshape(-1, 1))
    # print("sig_in_bkg_category", sig_in_bkg_category)

    # get the total loss
    #loss = 100 * (-10 * Z_sum_quad - sig_in_bkg_category)
    #loss = -100000 * Z_sum_quad - 1000 * sig_in_bkg_category
    #print("Z.shape: ", Z.shape[0])
    w_sig_cat_lst = [10, 5, 1]
    loss = (-1000 * (sum([Z[i] * w_sig_cat_lst[i] for i in range(Z.shape[0])]))) - (1 * sig_in_bkg_category) #+ (1 * bkg_in_bkg_category)

    

    # get number of sig and bkg under the mass peak
    num_sig_under_the_mass_peak = torch.sum(extra_feat[:, 0].reshape(-1, 1) * sig_under_the_mass_peak_mask)
    num_bkg_under_the_mass_peak = torch.sum(extra_feat[:, 0].reshape(-1, 1) * bkg_under_the_mass_peak_mask)

    return loss, Z_sum_quad, sig_in_bkg_category, Z#, num_sig_under_the_mass_peak, num_bkg_under_the_mass_peak

def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch):

    model.train()

    # Intialize emtpy list of metric variables
    loss_batch = []
    Z_sum_quad_batch = []
    sig_in_bkg_category_batch = []
    Z_batch = []
    num_sig_under_the_mass_peak_batch = []
    num_bkg_under_the_mass_peak_batch = []

    # Initialize progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", leave=False)
    for X_batch, extra_feat_batch in progress_bar:
        X_batch = X_batch.to(device)
        extra_feat_batch = extra_feat_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        #loss, Z_sum_quad, sig_in_bkg_category, Z, num_sig_under_the_mass_peak, num_bkg_under_the_mass_peak = criterion(y_pred, extra_feat_batch)
        loss, Z_sum_quad, sig_in_bkg_category, Z = criterion(y_pred, extra_feat_batch)
        loss.backward()
        optimizer.step()

        #print("loss", loss.item())
        loss_batch.append(loss.item())
        Z_sum_quad_batch.append(Z_sum_quad.item())
        sig_in_bkg_category_batch.append(sig_in_bkg_category.item())
        Z_batch.append([Z[i].item() for i in range(Z.shape[0])])
        #num_sig_under_the_mass_peak_batch.append(num_sig_under_the_mass_peak.item())
        #num_bkg_under_the_mass_peak_batch.append(num_bkg_under_the_mass_peak.item())

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Z_sum_quad': f'{Z_sum_quad.item():.4f}',
            'sig_in_bkg_category': f'{sig_in_bkg_category.item():.4f}',
            'Z': f'{[Z[i].item() for i in range(Z.shape[0])]}'
        })

    return np.mean(loss_batch), np.mean(Z_sum_quad_batch), np.mean(sig_in_bkg_category_batch), np.mean(np.array(Z_batch), axis=0)#, np.sum(num_sig_under_the_mass_peak_batch), np.sum(num_bkg_under_the_mass_peak_batch), y_pred

def evaluate(model, criterion, val_loader, device, epoch):
    model.eval()

    # Intialize emtpy list of metric variables
    loss_batch = []
    Z_sum_quad_batch = []
    sig_in_bkg_category_batch = []
    Z_batch = []
    num_sig_under_the_mass_peak_batch = []
    num_bkg_under_the_mass_peak_batch = []

    # Initialize progress bar
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
    with torch.no_grad():
        for X_batch, extra_feat_batch in progress_bar:
            X_batch = X_batch.to(device)
            extra_feat_batch = extra_feat_batch.to(device)

            y_pred = model(X_batch)
            #loss, Z_sum_quad, sig_in_bkg_category, Z, num_sig_under_the_mass_peak, num_bkg_under_the_mass_peak = criterion(y_pred, extra_feat_batch)
            loss, Z_sum_quad, sig_in_bkg_category, Z = criterion(y_pred, extra_feat_batch)

            loss_batch.append(loss.item())
            Z_sum_quad_batch.append(Z_sum_quad.item())
            sig_in_bkg_category_batch.append(sig_in_bkg_category.item())
            Z_batch.append([Z[i].item() for i in range(Z.shape[0])])
            #num_sig_under_the_mass_peak_batch.append(num_sig_under_the_mass_peak.item())
            #num_bkg_under_the_mass_peak_batch.append(num_bkg_under_the_mass_peak.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Z_sum_quad': f'{Z_sum_quad.item():.4f}',
                'sig_in_bkg_category': f'{sig_in_bkg_category.item():.4f}',
                'Z': f'{[Z[i].item() for i in range(Z.shape[0])]}'
            })

    return np.mean(loss_batch), np.mean(Z_sum_quad_batch), np.mean(sig_in_bkg_category_batch), np.mean(np.array(Z_batch), axis=0)#, np.sum(num_sig_under_the_mass_peak_batch), np.sum(num_bkg_under_the_mass_peak_batch), y_pred

#def plot_vs_epoch()


def train_MLP_based_categorization(input_path, model_config_path, device):

    # Load the data
    X_train = np.load(f"{input_path}/X_train.npy")
    X_val = np.load(f"{input_path}/X_val.npy")
    extra_feat_train = np.load(f"{input_path}/extra_feat_train.npy")
    extra_feat_val = np.load(f"{input_path}/extra_feat_val.npy")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    extra_feat_train = torch.tensor(extra_feat_train, dtype=torch.float32)
    extra_feat_val = torch.tensor(extra_feat_val, dtype=torch.float32)

    # Load the model config yaml
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # create dataset
    train_dataset = CustomDataset(X_train, extra_feat_train)
    val_dataset = CustomDataset(X_val, extra_feat_val)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=model_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config["batch_size"], shuffle=False)
    #train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    
    model = MLP(input_size=X_train.shape[1], num_layers=model_config["num_layers"], num_nodes=model_config["num_nodes"], output_size=model_config["num_categories"], act_fn=getattr(nn, model_config["act_fn"]), dropout_prob=model_config["dropout_prob"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["lr"], weight_decay=model_config["weight_decay"])
    criterion = significance_loss

    if model_config["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=model_config["patience_scheduler"], min_lr=1e-6)
    elif model_config["scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=model_config["T_max"] , eta_min=1e-7)
    elif model_config["scheduler"] == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=model_config["gamma"])
    elif model_config["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=model_config["step_size"], gamma=model_config["gamma"])

    
    

    n_epochs = model_config["n_epochs"]
    best_loss = np.inf
    patience = model_config["patience_early_stopping"]
    counter = 0

    train_loss_epoch = []
    train_Z_sum_quad_epoch = []
    train_sig_in_bkg_category_epoch = []
    train_Z_epoch = []
    val_loss_epoch = []
    val_Z_sum_quad_epoch = []
    val_sig_in_bkg_category_epoch = []
    val_Z_epoch = []

    Z_all = []
    bkg_side_band_lst = []

    lr_lst = []

    for epoch in range(n_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")

        loss, Z_sum_quad, sig_in_bkg_category, Z = train_one_epoch(model, optimizer, criterion, train_loader, device, epoch)
        train_loss_epoch.append(loss)
        train_Z_sum_quad_epoch.append(Z_sum_quad)
        train_sig_in_bkg_category_epoch.append(sig_in_bkg_category)
        train_Z_epoch.append(Z)

        val_loss, val_Z_sum_quad, val_sig_in_bkg_category, val_Z = evaluate(model, criterion, val_loader, device, epoch)
        val_loss_epoch.append(val_loss)
        val_Z_sum_quad_epoch.append(val_Z_sum_quad)
        val_sig_in_bkg_category_epoch.append(val_sig_in_bkg_category)
        val_Z_epoch.append(val_Z)

        # Update the learning rate
        scheduler.step()
        
        lr_lst.append(current_lr)
        print(f"Epoch {epoch}: Current learning rate = {current_lr}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter == patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch} - Train Z_sum_quad: {Z_sum_quad:.7f}, Val Z_sum_quad: {val_Z_sum_quad:.7f}")
        print(f"Epoch {epoch} - Train sig_in_bkg_category: {sig_in_bkg_category:.7f}, Val sig_in_bkg_category: {val_sig_in_bkg_category:.7f}")
        print(f"Epoch {epoch} - Train Z: {Z}, Val Z: {Z}")

        # get prediction for train and val
        model.eval()
        with torch.no_grad():
            y_train_pred = np.argmax(model(X_train.to(device)).cpu().numpy(), axis=1)
            y_val_pred = np.argmax(model(X_val.to(device)).cpu().numpy(), axis=1)
        
        # concatenate train and val pred, and also extra_feat
        y_pred_argmax = np.concatenate([y_train_pred, y_val_pred], axis=0)
        extra_feat = np.concatenate([extra_feat_train, extra_feat_val], axis=0)

        signal_under_mass_peak = np.sum(extra_feat[(extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, -1]==0)][:, 0])
        print("###########")
        print("signal_under_mass_peak", signal_under_mass_peak)
        print("###########")
        bkg_under_mass_peak = np.sum(extra_feat[(extra_feat[:, 1] > 120) & (extra_feat[:, 1] < 130) & (extra_feat[:, -1]==1)][:, 0])
        print("###########")
        print("bkg_under_mass_peak", bkg_under_mass_peak)
        print("###########")

        # get assymptotic significance for train and val combined for category predicted as signal regions
        extra_feat_sig_category = extra_feat[y_pred_argmax==0]
        sig_under_the_peak = np.sum(extra_feat_sig_category[(extra_feat_sig_category[:, -1]==1) & (extra_feat_sig_category[:, 1] > 120) & (extra_feat_sig_category[:, 1] < 130)][:, 0])
        bkg_under_the_peak = np.sum(extra_feat_sig_category[(extra_feat_sig_category[:, -1]==0) & (extra_feat_sig_category[:, 1] > 120) & (extra_feat_sig_category[:, 1] < 130)][:, 0])

        # get background in side band
        bkg_side_band = np.sum(extra_feat_sig_category[(extra_feat_sig_category[:, -1]==0) & ((extra_feat_sig_category[:, 1] < 120) | (extra_feat_sig_category[:, 1] > 130))][:, 0])
        bkg_side_band_lst.append(bkg_side_band)

        Z = np.sqrt(2 * ((sig_under_the_peak + bkg_under_the_peak) * np.log(1 + sig_under_the_peak/bkg_under_the_peak) - sig_under_the_peak))
        print((sig_under_the_peak + bkg_under_the_peak) * np.log(1 + sig_under_the_peak/bkg_under_the_peak))
        Z_all.append(Z)

        
        print(f"Epoch {epoch} - sig_under_the_peak: {sig_under_the_peak}", "\n")
        print(f"Epoch {epoch} - bkg_under_the_peak: {bkg_under_the_peak}", "\n")
        print(f"Epoch {epoch} - Z: {Z}")
        print(f"Epoch {epoch} - bkg_side_band: {bkg_side_band}", "\n")
    
    plt.plot([i for i in range(len(Z_all))], Z_all, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Z")
    plt.savefig("Z_vs_epoch.png")
    plt.clf()

    plt.plot([i for i in range(len(bkg_side_band_lst))], bkg_side_band_lst, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("bkg")
    plt.savefig("bkg_vs_epoch.png")
        





