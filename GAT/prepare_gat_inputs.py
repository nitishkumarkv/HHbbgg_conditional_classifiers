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
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "gat"
        self.input_var_json = os.path.join(os.path.dirname(__file__), "input_vars.json")
        self.sample_to_class = {
            "GGJets": "is_non_resonant_bkg",
            "GJetPt20To40": "is_non_resonant_bkg",
            "GJetPt40": "is_non_resonant_bkg",
            "ttHToGG": "is_ttH_bkg",
            "GluGluToHH": "is_GluGluToHH_sig",
            "VBFToHH": "is_VBFToHH_sig"
            }
        self.classes = ["is_non_resonant_bkg", "is_ttH_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]

    def load_vars(self, path):
        print(path)
        with open(path) as f:
            vals = json.load(f)
        return vals

    def load_parquet(self, path, N_files=-1):

        file_list = glob.glob(path + '*.parquet')
        file_list_ = file_list[:N_files]

        events = ak.from_parquet(file_list_)
        print(f"INFO: loaded parquet files from the path {path}")

        sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in file_list)

        return events, sum_genw_beforesel

    def get_relative_xsec_weight(self, events, sum_genw_beforesel, sample_type, weight_increase=10):
        
        dict_xsec = {
            "GGJets": 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "ttHToGG": 0.5700e3 * 0.00227,  # cross sectio of ttH * BR(HToGG)
            "GluGluToHH": 0.0311e3 * 0.00227 * 0.582 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HToGG) * 2 for two combination ### have to recheck if this is correct. 
            "VBFToHH": 0.00173e3 * 0.00227 * 0.582 * 2  # cross sectio of VBFToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct.
        }
        sum_of_weights = np.sum(events.weight / sum_genw_beforesel)
        events["rel_xsec_weight"] = (events.weight / sum_genw_beforesel) * dict_xsec[sample_type]
        return events, sum_of_weights

    def get_weights_for_training(self, y_train, rel_w_train):

        class_weights_for_training = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            cls_bool = (y_train[:, i] == 1)
            abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
            class_weights_for_training = class_weights_for_training + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))
            print(f"(number of events: sum of class_weights_for_training) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training[y_train[:, i] == 1])})")

        return class_weights_for_training

    def get_weights_for_val_test(self, y_val, rel_w_val):

        class_weights_for_val = ak.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            cls_bool = (y_val[:, i] == 1)
            abs_rel_xsec_weight_for_class = rel_w_val * cls_bool
            class_weights_for_val = class_weights_for_val + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))
            
        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

        return class_weights_for_val

    def train_test_split(self, X, Y, weights):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, weights_train, weights_test_val = train_test_split(X, Y, weights, train_size=0.6, shuffle=True, random_state=42)
        X_val, X_test, y_val, y_test, weights_val, weights_test = train_test_split(X_test_val, y_test_val, weights_test_val, train_size=0.5, shuffle=True, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test

    def nanmin(self, tensor, dim, keepdim=False):
        mask = torch.isnan(tensor)
        tensor = tensor.clone()
        tensor[mask] = float('inf')
        min_val, _ = torch.min(tensor, dim=dim, keepdim=keepdim)
        min_val[min_val == float('inf')] = float('nan')
        return min_val

    def nanmax(self, tensor, dim, keepdim=False):
        mask = torch.isnan(tensor)
        tensor = tensor.clone()
        tensor[mask] = float('-inf')
        max_val, _ = torch.max(tensor, dim=dim, keepdim=keepdim)
        max_val[max_val == float('-inf')] = float('nan')
        return max_val

    def min_max_scale(self, list_of_tensors):
        stacked_tensors = torch.cat([tensor[:, :4] for tensor in list_of_tensors], dim=0)
        mask = ~torch.isnan(stacked_tensors)
        min_vals = torch.min(stacked_tensors[mask].view(-1, 4), dim=0, keepdim=True)[0]
        max_vals = torch.max(stacked_tensors[mask].view(-1, 4), dim=0, keepdim=True)[0]
        scaled_tensors = []
        for tensor in list_of_tensors:
            to_scale = tensor[:, :4]
            not_to_scale = tensor[:, 4:]
            scaled_tensor = (to_scale - min_vals) / (max_vals - min_vals)
            scaled_tensor[torch.isnan(scaled_tensor)] = float('nan')
            scaled_tensor = torch.cat((scaled_tensor, not_to_scale), dim=1)
            scaled_tensors.append(scaled_tensor)
        return scaled_tensors

    # create vectors for gat

    def convert_to_object_vectors(self, X, relative_weights):
        var_mapping = {
            'photon_pt1': 'lead_pt', 'photon_eta1': 'lead_eta', 'photon_phi1': 'lead_phi', 'photon_id1': "lead_mvaID",
            'photon_pt2': 'sublead_pt', 'photon_eta2': 'sublead_eta', 'photon_phi2': 'sublead_phi', 'photon_id2': "sublead_mvaID",
            'jet_pt1': 'jet1_pt', 'jet_eta1': 'jet1_eta', 'jet_phi1': 'jet1_phi', 'jet_id1': "jet1_btagPNetB",
            'jet_pt2': 'jet2_pt', 'jet_eta2': 'jet2_eta', 'jet_phi2': 'jet2_phi', 'jet_id2': "jet2_btagPNetB",
            'jet_pt3': 'jet3_pt', 'jet_eta3': 'jet3_eta', 'jet_phi3': 'jet3_phi', 'jet_id3': "jet3_btagPNetB",
            'jet_pt4': 'jet4_pt', 'jet_eta4': 'jet4_eta', 'jet_phi4': 'jet4_phi', 'jet_id4': "jet4_btagPNetB",
            'jet_pt5': 'jet5_pt', 'jet_eta5': 'jet5_eta', 'jet_phi5': 'jet5_phi', 'jet_id5': "jet5_btagPNetB",
            'jet_pt6': 'jet6_pt', 'jet_eta6': 'jet6_eta', 'jet_phi6': 'jet6_phi', 'jet_id6': "jet6_btagPNetB",
            'bjet_pt1': 'lead_bjet_pt', 'bjet_eta1': 'lead_bjet_eta', 'bjet_phi1': 'lead_bjet_phi', 'bjet_id1': "lead_bjet_btagPNetB",
            'bjet_pt2': 'sublead_bjet_pt', 'bjet_eta2': 'sublead_bjet_eta', 'bjet_phi2': 'sublead_bjet_phi', 'bjet_id2': "sublead_bjet_btagPNetB",
            'VBFjet_pt1': 'VBF_first_jet_pt', 'VBFjet_eta1': 'VBF_first_jet_eta', 'VBFjet_phi1': 'VBF_first_jet_phi', 'VBFjet_id1': "VBF_first_jet_btagPNetB",
            'VBFjet_pt2': 'VBF_second_jet_pt', 'VBFjet_eta2': 'VBF_second_jet_eta', 'VBFjet_phi2': 'VBF_second_jet_phi', 'VBFjet_id2': "VBF_second_jet_btagPNetB",
            'lepton_pt1': 'lepton1_pt', 'lepton_eta1': 'lepton1_eta', 'lepton_phi1': 'lepton1_phi', 'lepton_gen1': 'lepton1_generation', 'lepton_id1': "lepton1_mvaID",
            'lepton_pt2': 'lepton2_pt', 'lepton_eta2': 'lepton2_eta', 'lepton_phi2': 'lepton2_phi', 'lepton_gen2': 'lepton2_generation', 'lepton_id2': "lepton2_mvaID",
            'lepton_pt3': 'lepton3_pt', 'lepton_eta3': 'lepton3_eta', 'lepton_phi3': 'lepton3_phi', 'lepton_gen3': 'lepton3_generation', 'lepton_id3': "lepton3_mvaID",
            'lepton_pt4': 'lepton4_pt', 'lepton_eta4': 'lepton4_eta', 'lepton_phi4': 'lepton4_phi', 'lepton_gen4': 'lepton4_generation',  'lepton_id4': "lepton4_mvaID"
        }

        n_photons = 2
        n_jets = 6
        n_bjets = 2
        n_leptons = 4
        n_VBFjets = 2

        data_for_training = []
        rel_weights = []

        for i in range(len(X)):
            event_tensors = []  # save tensor for each event
            
            # Photons
            for k in range(n_photons):
                pt = X.iloc[i][var_mapping[f'photon_pt{k+1}']]
                eta = X.iloc[i][var_mapping[f'photon_eta{k+1}']]
                phi = X.iloc[i][var_mapping[f'photon_phi{k+1}']]
                Id = X.iloc[i][var_mapping[f'photon_id{k+1}']]
                photon_tensor = torch.tensor([pt, eta, phi, Id, 1, 0, 0, 0, 0, 0], dtype=torch.float)
                event_tensors.append(photon_tensor)
            
            # Jets
            for k in range(n_jets):
                pt = X.iloc[i][var_mapping[f'jet_pt{k+1}']]
                eta = X.iloc[i][var_mapping[f'jet_eta{k+1}']]
                phi = X.iloc[i][var_mapping[f'jet_phi{k+1}']]
                Id = X.iloc[i][var_mapping[f'jet_id{k+1}']]
                jet_tensor = torch.tensor([pt, eta, phi, Id, 0, 0, 1, 0, 0, 0], dtype=torch.float)
                event_tensors.append(jet_tensor)

            # BJets
            for k in range(n_bjets):
                pt = X.iloc[i][var_mapping[f'bjet_pt{k+1}']]
                eta = X.iloc[i][var_mapping[f'bjet_eta{k+1}']]
                phi = X.iloc[i][var_mapping[f'bjet_phi{k+1}']]
                Id = X.iloc[i][var_mapping[f'bjet_id{k+1}']]
                bjet_tensor = torch.tensor([pt, eta, phi, Id, 0, 1, 0, 0, 0, 0], dtype=torch.float)
                event_tensors.append(bjet_tensor)

            # VBFJets
            for k in range(n_bjets):
                pt = X.iloc[i][var_mapping[f'VBFjet_pt{k+1}']]
                eta = X.iloc[i][var_mapping[f'VBFjet_eta{k+1}']]
                phi = X.iloc[i][var_mapping[f'VBFjet_phi{k+1}']]
                Id = X.iloc[i][var_mapping[f'VBFjet_id{k+1}']]
                bjet_tensor = torch.tensor([pt, eta, phi, Id, 0, 0, 0, 1, 0, 0], dtype=torch.float)
                event_tensors.append(bjet_tensor)

            # Leptons
            for k in range(n_leptons):
                pt = X.iloc[i][var_mapping[f'lepton_pt{k+1}']]
                eta = X.iloc[i][var_mapping[f'lepton_eta{k+1}']]
                phi = X.iloc[i][var_mapping[f'lepton_phi{k+1}']]
                Id = X.iloc[i][var_mapping[f'lepton_id{k+1}']]
                gen = X.iloc[i][var_mapping[f'lepton_gen{k+1}']]
                lepton_tensor = torch.tensor([pt, eta, phi, Id, 0, 0, 0, 0, int(gen == 1), int(gen == 2)], dtype=torch.float)
                event_tensors.append(lepton_tensor)
            
            num_nodes = len(event_tensors)
            if isinstance(relative_weights, pd.DataFrame):
                event_weight = [relative_weights.iloc[i, 0]] * num_nodes
            elif isinstance(relative_weights, pd.Series):
                event_weight = [relative_weights.iloc[i]] * num_nodes
            rel_weights.append(event_weight)

            event_tensor = torch.stack(event_tensors)
            data_for_training.append(event_tensor)

        return data_for_training, rel_weights

    def generate_edges(self, num_nodes):
        # create a fully connected graph
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j: 
                    edges.append((i, j))
        return torch.tensor(edges).t()

    def prep_input_for_mlp(self, samples_path, out_path, fill_nan = -999):

        #os.makedirs(out_path)

        comb_inputs = []

        for samples in self.sample_to_class.keys():

            events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{samples}/nominal/", -1)

            # get only valid events, cuts not applied in HiggsDNA
            events = events[(events.mass > 100) | (events.mass < 180)]
            events = events[(events.dijet_mass > 70) | (events.dijet_mass < 190)]

            # get relative weights according to cross section of the process
            events, sum_of_weights = self.get_relative_xsec_weight(events, sum_genw_beforesel, samples)

            # add the bools for each class
            for cls in self.classes:  # first intialize everything to zero
                events[cls] = ak.zeros_like(events.eta)

            events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded
            comb_inputs.append(events)

            print(f"INFO: Number of events in {samples}: {len(events)}")

        comb_inputs = ak.concatenate(comb_inputs, axis=0)

        comb_inputs = pd.DataFrame(ak.to_list(comb_inputs))

        for cls in self.classes:
            print("\n", f"INFO: Number of events in {cls}: {sum(comb_inputs[cls])}")

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]
        vars_for_training = vars_config["vars"]
        
        X = comb_inputs[vars_for_training]
        Y = comb_inputs[[cls for cls in self.classes]]
        relative_weights = comb_inputs["rel_xsec_weight"]
        
        X = X.where(X >= -998, np.nan)

        #create objevt tensors
        data_for_training, rel_weights = self.convert_to_object_vectors(X, relative_weights)

        #min max scaling
        scaled_data = [self.min_max_scale(data_for_training)]
        tensor_list = scaled_data[0]
        
        data_for_gat = []
        for tensor in tensor_list:
            mask = ~torch.isnan(tensor).any(dim=1)
            cleaned_tensor = tensor[mask]
            data_for_gat.append(cleaned_tensor)

        for element in data_for_gat:
            element = element.values

        Y=Y.values

        X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test = self.train_test_split(data_for_gat, Y, relative_weights)

        class_weights_for_training = self.get_weights_for_training(y_train, rel_w_train)
        class_weights_for_val = self.get_weights_for_val_test(y_val, rel_w_val)
        class_weights_for_test = self.get_weights_for_val_test(y_test, rel_w_test)

        # save all the numpy arrays
        print("\n INFO: saving inputs for gat")
        torch.save(X_train, f"{out_path}/X_train")
        torch.save(X_val, f"{out_path}/X_val")
        torch.save(X_test, f"{out_path}/X_test")

        torch.save(y_train, f"{out_path}/y_train")
        torch.save(y_val, f"{out_path}/y_val")
        torch.save(y_test, f"{out_path}/y_test")

        torch.save(rel_w_train, f"{out_path}/rel_w_train")
        torch.save(rel_w_val, f"{out_path}/rel_w_val")
        torch.save(rel_w_test, f"{out_path}/rel_w_test")
        
        torch.save(class_weights_for_training, f"{out_path}/class_weights_for_training")
        torch.save(class_weights_for_val, f"{out_path}/class_weights_for_val")
        torch.save(class_weights_for_test, f"{out_path}/class_weights_for_test")

if __name__ == "__main__":
    out = PrepareInputs()
    # out.prep_input("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models")
    out.prep_input_for_mlp("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples_v2/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/GAT/inputs")