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
import gc


class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = os.path.join(os.path.dirname(__file__), "input_variables.json")
        #self.sample_to_class = {
        #    "GGJets": "is_non_resonant_bkg",
        #    "GJetPt20To40": "is_non_resonant_bkg",
        #    "GJetPt40": "is_non_resonant_bkg",
        #    "ttHToGG": "is_ttH_bkg",
        #    "GluGluToH": "is_single_H_bkg",
        #    "GluGluToHH": "is_GluGluToHH_sig",
        #    }
        self.sample_to_class = {
            "GGJets": "is_non_resonant_bkg",
            "GJetPt20To40": "is_non_resonant_bkg",
            "GJetPt40": "is_non_resonant_bkg",
            "TTGG": "is_non_resonant_bkg",
            "ttHtoGG_M_125": "is_ttH_bkg",
            "GluGluHToGG_M_125": "is_single_H_bkg",
            "VBFHToGG_M_125": "is_single_H_bkg",
            "VHtoGG_M_125": "is_single_H_bkg",
            "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00": "is_GluGluToHH_sig",
            "VBFHHto2B2G_CV_1_C2V_1_C3_1": "is_VBFToHH_sig",
            }
        self.class_num = {
            "is_non_resonant_bkg": 0,
            "is_ttH_bkg": 1,
            "is_single_H_bkg": 2,
            "is_GluGluToHH_sig": 3,
            "is_VBFToHH_sig": 4
        }
        self.node_features = ["pt", "eta", "phi", "mass", "score", "is_photon", "is_b_tagged_jet", "is_VBFjet", "is_jet", "is_electron", "is_muon", "is_MET"]
        self.save_prelim_inputs = True
        self.extra_features = ["n_jets", "n_electrons", "n_muons", "nonRes_M_X", "fixedGridRhoAll", "era"]

    def load_vars(self, path):
        print(path)
        with open(path) as f:
            vars = json.load(f)
        return vars

    def load_parquet(self, path, N_files=-1):
        print(path)

        file_list = glob.glob(path + '*.parquet')
        if N_files == -1:
            file_list_ = file_list
        else:
            file_list_ = file_list[:N_files]
        
        print(file_list_)

        events = ak.from_parquet(file_list_)
        #print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        events = ak.from_parquet(file_list_)
        #print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        return events

    def get_relative_xsec_weight(self, events, sample_type, era):

        dict_xsec = {
            "GGJets": 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.02391e3,  # cross sectio of TTGG 0.01696, copilot: 0.502
            "ttHtoGG_M_125": 0.5700e3 * 0.00227,  # cross sectio of ttH * BR(HToGG)
            "GluGluHToGG_M_125": 52.23e3 * 0.00227,  # cross sectio of GluGluHToGG * BR(HToGG)
            "VBFHToGG_M_125": 4.078e3 * 0.00227,
            "VHtoGG_M_125": 2.4009e3 * 0.00227,
            "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00": 0.0311e3 * 0.00227 * 0.582 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HToGG) * 2 for two combination ### have to recheck if this is correct. 
            "VBFHHto2B2G_CV_1_C2V_1_C3_1": 0.00173e3 * 0.00227 * 0.582 * 2  # cross sectio of VBFToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct.
        }
        luminosities = {
        "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
        "postEE": 26.67  # Integrated luminosity for postEE in fb^-1
        }

        events["rel_xsec_weight"] = (events.weight) * dict_xsec[sample_type] * luminosities[era]

        return events

    def get_weights_for_training(self, y_train, rel_w_train):

        class_weights_for_training = torch.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            cls_bool = (y_train[:, i] == 1)
            abs_rel_xsec_weight_for_class = torch.abs(rel_w_train) * cls_bool
            class_weights_for_training = class_weights_for_training + (abs_rel_xsec_weight_for_class / torch.sum(abs_rel_xsec_weight_for_class))

        for i in range(y_train.shape[1]):
            print(f"(number of events: sum of class_weights_for_training) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training[y_train[:, i] == 1])})")

        return class_weights_for_training

    def get_weights_for_val_test(self, y_val, rel_w_val):

        class_weights_for_val = torch.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            cls_bool = (y_val[:, i] == 1)
            rel_xsec_weight_for_class = rel_w_val * cls_bool
            class_weights_for_val = class_weights_for_val + (rel_xsec_weight_for_class / torch.sum(rel_xsec_weight_for_class))

        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

        return class_weights_for_val

    def train_test_split(self, comb_jagged_arr, comb_extra_features, comb_class_labels, comb_relative_xsec, train_ratio=0.7):

        from sklearn.model_selection import train_test_split

        num_events = len(comb_jagged_arr)
        indices = ak.Array(range(num_events))

        train_indices, val_indices = train_test_split(
            indices,
            train_size=train_ratio,
            shuffle=True,
            random_state=42
        )


        comb_jagged_arr_train = comb_jagged_arr[train_indices]
        comb_jagged_arr_val = comb_jagged_arr[val_indices]
        comb_extra_features_train = comb_extra_features[train_indices]
        comb_extra_features_val = comb_extra_features[val_indices]
        comb_class_labels_train = comb_class_labels[train_indices]
        comb_class_labels_val = comb_class_labels[val_indices]
        comb_relative_xsec_train = comb_relative_xsec[train_indices]
        comb_relative_xsec_val = comb_relative_xsec[val_indices]

        print("len(comb_jagged_arr_train): ", len(comb_jagged_arr_train))
        print("len(comb_jagged_arr_val): ", len(comb_jagged_arr_val))

        return comb_jagged_arr_train, comb_jagged_arr_val, comb_extra_features_train, comb_extra_features_val, comb_class_labels_train, comb_class_labels_val, comb_relative_xsec_train, comb_relative_xsec_val

    def standardize(self, X, mean, std):
        return (X - mean) / std

    def min_max_scale(self, X, min, max):
        return (X - min) / (max - min)

    def prep_jagged_array(self, events, ignore_value = -999):

        # os.makedirs(out_path)

        # define objects and features
        obj_features = {
            "lead_photon": {
                "pt": "lead_pt",
                "eta": "lead_eta",
                "phi": "lead_phi",
                "mass": 0.0,
                "score": "lead_mvaID",
                "is_photon": 1.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 0.0,
                "is_jet": 0.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "sublead_photon": {
                "pt": "sublead_pt",
                "eta": "sublead_eta",
                "phi": "sublead_phi",
                "mass": 0.0,
                "score": "sublead_mvaID",
                "is_photon": 1.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 0.0,
                "is_jet": 0.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "lead_bjet": {
                "pt": "nonRes_lead_bjet_pt",
                "eta": "nonRes_lead_bjet_eta",
                "phi": "nonRes_lead_bjet_phi",
                "mass": "nonRes_lead_bjet_mass",
                "score": "nonRes_lead_bjet_btagPNetB",
                "is_photon": 0.0,
                "is_b_tagged_jet": 1.0,
                "is_VBFjet": 0.0,
                "is_jet": 1.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "sublead_bjet": {
                "pt": "nonRes_sublead_bjet_pt",
                "eta": "nonRes_sublead_bjet_eta",
                "phi": "nonRes_sublead_bjet_phi",
                "mass": "nonRes_sublead_bjet_mass",
                "score": "nonRes_sublead_bjet_btagPNetB",
                "is_photon": 0.0,
                "is_b_tagged_jet": 1.0,
                "is_VBFjet": 0.0,
                "is_jet": 1.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "VBF_first_jet": {
                "pt": "VBF_first_jet_pt",
                "eta": "VBF_first_jet_eta",
                "phi": "VBF_first_jet_phi",
                "mass": "VBF_first_jet_mass",
                "score": "VBF_first_jet_btagPNetQvG",
                "is_photon": 0.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 1.0,
                "is_jet": 1.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "VBF_second_jet": {
                "pt": "VBF_second_jet_pt",
                "eta": "VBF_second_jet_eta",
                "phi": "VBF_second_jet_phi",
                "mass": "VBF_second_jet_mass",
                "score": "VBF_second_jet_btagPNetQvG",
                "is_photon": 0.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 1.0,
                "is_jet": 1.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "MET": {
                "pt": "puppiMET_pt",
                "eta": 0.0,
                "phi": "puppiMET_phi",
                "mass": 0.0,
                "score": 0.0,
                "is_photon": 0.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 0.0,
                "is_jet": 0.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 1.0,
            },
            "jets": {
                "pt": "jet_pt",
                "eta": "jet_eta",
                "phi": "jet_phi",
                "mass": "jet_mass",
                "score": "jet_btagPNetB",
                "is_photon": 0.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 0.0,
                "is_jet": 1.0,
                "is_electron": 0.0,
                "is_muon": 0.0,
                "is_MET": 0.0,
            },
            "leptons": {
                "pt": "lepton_pt",
                "eta": "lepton_eta",
                "phi": "lepton_phi",
                "mass": "lepton_mass",
                "score": 0.0,
                "is_photon": 0.0,
                "is_b_tagged_jet": 0.0,
                "is_VBFjet": 0.0,
                "is_jet": 0.0,
                "is_electron": "lepton_generation",
                "is_muon": "lepton_generation",
                "is_MET": 0.0,
            }
        }

        num_events = len(events)
        arr_to_save = {}
        for feat in obj_features["lead_photon"].keys():
            for obj in obj_features.keys():
                if not obj in ["jets", "leptons"]:
                    if feat not in arr_to_save.keys():
                        arr_to_save[feat] = ak.to_numpy(events[obj_features[obj][feat]]) if isinstance(obj_features[obj][feat], str) else np.ones(num_events)*obj_features[obj][feat]
                        arr_to_save[feat] = arr_to_save[feat].reshape(num_events, 1)
                    else:
                        arr_temp = ak.to_numpy(events[obj_features[obj][feat]]) if isinstance(obj_features[obj][feat], str) else np.ones(num_events)*obj_features[obj][feat]
                        arr_to_save[feat] = np.concatenate((arr_to_save[feat], arr_temp.reshape(num_events, 1)), axis=1)

                elif obj in ["jets"]:
                    for i in range(1, 11):
                        var_temp = obj_features[obj][feat]
                        if isinstance(var_temp, str):
                            var_temp = f"{var_temp[:3]}{i}{var_temp[3:]}"
                        arr_temp = ak.to_numpy(events[var_temp]) if isinstance(var_temp, str) else np.ones(num_events)*var_temp
                        arr_to_save[feat] = np.concatenate((arr_to_save[feat], arr_temp.reshape(num_events, 1)), axis=1)


                elif obj in ["leptons"]:
                    for i in range(1, 5):
                        var_temp = obj_features[obj][feat]
                        if isinstance(var_temp, str):
                            var_temp = f"{var_temp[:6]}{i}{var_temp[6:]}"
                        if feat == "is_electron":
                            arr_temp = ak.to_numpy(ak.where(events[var_temp] == 2, 0.0, events[var_temp]))
                        elif feat == "is_muon":
                            arr_temp = ak.to_numpy(ak.where(events[var_temp] == 1, 0, ak.where(events[var_temp] == 2, 1, events[var_temp])))
                        else:
                            arr_temp = ak.to_numpy(events[var_temp]) if isinstance(var_temp, str) else np.ones(num_events)*var_temp

                        arr_to_save[feat] = np.concatenate((arr_to_save[feat], arr_temp.reshape(num_events, 1)), axis=1)

        for key in arr_to_save.keys():
            arr_to_save[key] = ak.Array(arr_to_save[key])
        ak_arr_to_save = ak.zip(arr_to_save)

        # remove -999 values
        ak_arr_to_save = ak_arr_to_save[ak_arr_to_save.pt != ignore_value]

        num_particles = ak.sum(ak.num(ak_arr_to_save.pt))

        for field in ak_arr_to_save.fields:
            assert ak.sum(ak_arr_to_save[field]==-999.0) == 0
            assert ak.sum(ak.num(ak_arr_to_save[field])) == num_particles

        return ak_arr_to_save

    def prelim_inputs(self, samples_path, outpath):

        for era in ["preEE", "postEE"]:
            for samples in self.sample_to_class.keys():

                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", -1)

                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", f"{samples_path}/{era}/{samples}/nominal/", -1)

                #if era == "preEE":
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/final_dump/{samples}/nominal/"
                #else:
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/postEE/{samples}/nominal/"
                #events = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", -1)
                events = self.load_parquet(f"{samples_path}/{era}/{samples}/", -1)
            
                # event selection
                diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
                dijet_mass_cut = ((events.nonRes_dijet_mass > 70) & (events.nonRes_dijet_mass < 190))
                nonRes = (events.nonRes_has_two_btagged_jets == True)
                
                events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]

                events["fixedGridRhoAll"] = ak.nan_to_num(events.fixedGridRhoAll)
                print("sum None fixedGridRhoAll: ", ak.sum(ak.is_none(events.fixedGridRhoAll)))
                events = events[~ak.is_none(events.fixedGridRhoAll)]

                events["era"] = ak.zeros_like(events.eta) if era == "preEE" else ak.ones_like(events.eta)

                print(f"INFO: Number of events after selection for {samples}: {len(events)}")
                full_path_to_save = f"{outpath}/{era}/{samples}/"
                ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                # get relative weights according to cross section of the process and save them
                #os.makedirs(f"{samples_path}/jagged_arr/{era}/{samples}/", exist_ok=True)
                events = self.get_relative_xsec_weight(events, samples, era)
                #torch.save(torch.tensor(events["rel_xsec_weight"], dtype=torch.float32), f"{samples_path}/jagged_arr/{era}/{samples}/rel_xsec_weight.pt")

                # get jagged array
                print(f"INFO: Preparing jagged array for {samples} \n")
                #ak_arr_to_save = self.prep_jagged_array(events)
                #os.makedirs(f"{samples_path}/jagged_arr/{era}/{samples}/", exist_ok=True)
                #ak.to_parquet(ak_arr_to_save, f"{samples_path}/jagged_arr/{era}/{samples}/jagged_array.parquet")

                # save extra features
                #df = pd.DataFrame(ak.to_list(events))
                #extra_features = df[self.extra_features].values
                #extra_features = torch.tensor(extra_features, dtype=torch.float32)
                #torch.save(extra_features, f"{samples_path}/jagged_arr/{era}/{samples}/extra_features.pt")

                # save class labels
                #class_events = np.zeros((len(ak_arr_to_save), len(self.class_num)))
                #class_events[:, self.class_num[self.sample_to_class[samples]]] = 1.0
                #class_events = torch.tensor(class_events, dtype=torch.float32)
                #torch.save(class_events, f"{samples_path}/jagged_arr/{era}/{samples}/class_labels.pt")

    def extra_features_torch(self, extra_features_path, sample_path):

        if os.path.exists(extra_features_path):
            print("extra features already exists")
            extra_features = torch.load(extra_features_path)
        else:
            events = self.load_parquet(sample_path, sample_path, -1)
            df = pd.DataFrame(ak.to_list(events))
            extra_features = df[self.extra_features].values
            extra_features = torch.tensor(extra_features, dtype=torch.float32)
        
        return extra_features

        
    def merge_transform_scale(self, samples_path, input_path, scale_type="standard", pt_log = True): # samples_path

        out_path = f"{input_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)

        # open saved scale values
        with open(f"{input_path}/scale_values.json", "r") as f:
            scale_values = json.load(f)
        with open(f"{input_path}/extra_features_scale_values.json", "r") as f:
            extra_features_scale_values = json.load(f)

                # save scale values
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/scale_values.json", "w") as f:
            json.dump(scale_values, f)
        print(extra_features_scale_values)
        print(type(extra_features_scale_values))
        with open(f"{out_path}/extra_features_scale_values.json", "w") as f:
            json.dump(extra_features_scale_values, f)

        

        for era in ["preEE", "postEE"]:
            for sample in self.sample_to_class.keys():

                jagged_array = ak.from_parquet(f"{input_path}/jagged_arr/{era}/{sample}/jagged_array.parquet")
                extra_features = self.extra_features_torch(f"{input_path}/jagged_arr/{era}/{sample}/extra_features.pt", f"{samples_path}/{era}/{sample}/nominal/")
                class_labels = torch.load(f"{input_path}/jagged_arr/{era}/{sample}/class_labels.pt")
                relative_xsec = torch.load(f"{input_path}/jagged_arr/{era}/{sample}/rel_xsec_weight.pt")

                # transform variables
                print("INFO: Transforming variables")
                if pt_log:
                    jagged_array["pt"] = np.log(jagged_array["pt"])

                ############# scaling #############
                # scale node variables for training
                print("INFO: Scaling variables, using: ", scale_type)
                jagged_arr_scaled = {}

                for feat in self.node_features:
                    feat_arr = jagged_array[feat]

                    if scale_type == "min_max":
                        min_val = scale_values[feat]["min"]
                        max_val = scale_values[feat]["max"]
                        jagged_arr_scaled[feat] = self.min_max_scale(feat_arr, min_val, max_val)

                    elif scale_type == "standard":
                        mean = scale_values[feat]["mean"]
                        std = scale_values[feat]["std"]
                        jagged_arr_scaled[feat] = self.standardize(feat_arr, mean, std)

                jagged_arr_scaled = ak.zip(jagged_arr_scaled)


                # scale extra features
                if scale_type == "min_max":
                    extra_features_min = torch.tensor(extra_features_scale_values["min"], dtype=torch.float32)
                    extra_features_max = torch.tensor(extra_features_scale_values["max"], dtype=torch.float32)
                    extra_features_scale_values = {
                        "min": extra_features_min,
                        "max": extra_features_max
                    }
                    extra_features_scaled = self.min_max_scale(extra_features, extra_features_min, extra_features_max)
                elif scale_type == "standard":
                    extra_features_mean = torch.tensor(extra_features_scale_values["mean"], dtype=torch.float32)
                    extra_features_std = torch.tensor(extra_features_scale_values["std"], dtype=torch.float32)
                    extra_features_scale_values = {
                        "mean": extra_features_mean,
                        "std": extra_features_std
                    }
                    extra_features_scaled = self.standardize(extra_features, extra_features_mean, extra_features_std)

                print("INFO: Preparing Data list for training")
                data_lst = self.prep_Data_lst(jagged_arr_scaled, extra_features_scaled, class_labels, relative_xsec)
                print("INFO: Saving Data list for training")
                os.makedirs(f"{out_path}/{era}/{sample}/", exist_ok=True)
                #torch.save(data_lst, f"{out_path}/{era}/{sample}/data_lst.pt")
                np.save(f"{out_path}/{era}/{sample}/rel_w.npy", relative_xsec.numpy())



        return 0

    
    def prep_comp_graph_wo_self_loops(self, num_particles):
        edge_idx = []
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    edge_idx.append([i, j])
        return torch.tensor(edge_idx, dtype=torch.long).t().contiguous()

    def prep_Data_lst(self, comb_jagged_arr, comb_extra_features, comb_class_labels, comb_weight):

        concat_arr = ak.concatenate([comb_jagged_arr[self.node_features[i]][:, :, None] for i in range(len(self.node_features))], axis=-1)

        data_lst = [
            Data(
                x=torch.from_numpy(ak.to_numpy(event).astype(np.float32)),
                extra_feat=torch.reshape(extra_features, (1, -1)),
                edge_index=self.prep_comp_graph_wo_self_loops(len(event)),
                y=torch.reshape(class_label, (1, -1)),
                weight=torch.reshape(weight, (1, -1))
            )
            for event, extra_features, class_label, weight in zip(concat_arr, comb_extra_features, comb_class_labels, comb_weight)
        ]

        return data_lst

    def prep_input_GNN(self, samples_path, inputs_path):

        
#
#        #if self.save_prelim_inputs:
#        #    self.prelim_inputs(samples_path)
#
#        #comb_jagged_arr_scaled, comb_extra_features, comb_class_labels, comb_relative_xsec, scale_values = self.merge_transform_scale(samples_path)
#        jagged_arr_scaled, extra_features_scaled, relative_xsec, scale_values, extra_features_scale_values = self.merge_transform_scale(samples_path, inputs_path, pt_log=False)
#
#        
#
#        
#
#        # save relative cross section weights
#        torch.save(comb_relative_xsec_train, f"{out_path}/relative_xsec_train.pt")
#        torch.save(comb_relative_xsec_val, f"{out_path}/relative_xsec_val.pt")
#        torch.save(class_weights_for_training_no_absolute, f"{out_path}/class_weights_for_training_no_absolute.pt")
#
#        # also save the node features and extra features list
#        with open(f"{out_path}/node_features.json", "w") as f:
#            json.dump(self.node_features, f)
#        with open(f"{out_path}/extra_features.json", "w") as f:
#            json.dump(self.extra_features, f)
#
#
#        # clear memory
#        del comb_jagged_arr_train_scaled, comb_extra_features_train_scaled, comb_class_labels_train, class_weights_for_training, class_weights_for_training_no_absolute, comb_relative_xsec_train, scale_values, extra_features_scale_values, comb_jagged_arr_val_scaled, comb_extra_features_val_scaled, comb_class_labels_val, class_weights_for_val, comb_relative_xsec_val
#        gc.collect()
#
        return 0


if __name__ == "__main__":

    samples_path = "samples/"
    #samples_path = "samples_for_mlp_dummy"
    outPath = "gnn_inputs_20241203_new/individual_samples"

    prep = PrepareInputs()

    #out = prep.merge_transform_scale(samples_path, "gnn_inputs_20241203_new_copy")
    out = prep.prelim_inputs(samples_path, outPath)
    #print("data_lst", data_lst_train[0])
    #print(data_lst_train[0].x.shape)
    #print(data_lst_train[0].keys())
    print("INFO: Done with preparing inputs for GNN")