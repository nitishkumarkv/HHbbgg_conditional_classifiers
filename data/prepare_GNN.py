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
            "TTGG": "is_non_resonant_bkg",
            "ttHtoGG_M_125": "is_ttH_bkg",
            "GluGluHToGG_M_125": "is_single_H_bkg",
            "GluGluToHH": "is_GluGluToHH_sig",
            }
        self.class_num = {
            "is_non_resonant_bkg": 0,
            "is_ttH_bkg": 1,
            "is_single_H_bkg": 2,
            "is_GluGluToHH_sig": 3,
        }
        self.node_features = ["pt", "eta", "phi", "score", "is_photon", "is_b_tagged_jet", "is_VBFjet", "is_jet", "is_electron", "is_muon", "is_MET"]
        self.save_prelim_inputs = False
        self.extra_features = ["n_jets", "n_electrons", "n_muons"]

    def load_vars(self, path):
        print(path)
        with open(path) as f:
            vars = json.load(f)
        return vars

    def load_parquet(self, path, N_files=-1):

        file_list = glob.glob(path + '*.parquet')
        if N_files == -1:
            file_list_ = file_list
        else:
            file_list_ = file_list[:N_files]

        events = ak.from_parquet(file_list_)
        print(f"INFO: loaded parquet files from the path {path}")

        sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in file_list)

        sample_type = os.path.basename(os.path.dirname(path))
        events["sample_type"] = ak.Array([sample_type] * len(events))

        return events, sum_genw_beforesel

    def get_relative_xsec_weight(self, events, sum_genw_beforesel, sample_type):

        dict_xsec = {
            "GGJets": 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.01696e3,  # cross sectio of TTGG 0.01696, copilot: 0.502
            "ttHtoGG_M_125": 0.5700e3 * 0.00227,  # cross sectio of ttH * BR(HToGG)
            "GluGluHToGG_M_125": 52.23e3 * 0.00227,  # cross sectio of GluGluHToGG * BR(HToGG)
            "GluGluToHH": 0.0311e3 * 0.00227 * 0.582 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HToGG) * 2 for two combination ### have to recheck if this is correct. 
            "VBFToHH": 0.00173e3 * 0.00227 * 0.582 * 2  # cross sectio of VBFToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct.
        }

        events["rel_xsec_weight"] = (events.weight / sum_genw_beforesel) * dict_xsec[sample_type]
        return events

    def get_weights_for_training(self, y_train, rel_w_train):

        class_weights_for_training = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            cls_bool = (y_train[:, i] == 1)
            abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
            class_weights_for_training = class_weights_for_training + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

        for i in range(y_train.shape[1]):
            print(f"(number of events: sum of class_weights_for_training) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training[y_train[:, i] == 1])})")

        return class_weights_for_training


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
                "score": "VBF_first_jet_btagPNetB",
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
                "score": "VBF_second_jet_btagPNetB",
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

    def prelim_inputs(self, samples_path):

        for samples in self.sample_to_class.keys():

            print(f"INFO: Preparing preliminary inputs for {samples_path}/{samples}/nominal/")

            events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{samples}/nominal/", -1)

            # event selection
            diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
            dijet_mass_cut = ((events.nonRes_dijet_mass > 70) & (events.nonRes_dijet_mass < 190))
            nonRes = (events.nonRes_has_two_btagged_jets == True)
            events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]

            # get relative weights according to cross section of the process and save them
            events = self.get_relative_xsec_weight(events, sum_genw_beforesel, samples)
            torch.save(torch.tensor(events["rel_xsec_weight"], dtype=torch.float32), f"{samples_path}/jagged_arr/{samples}/rel_xsec_weight.pt")

            # get jagged array
            print(f"INFO: Preparing jagged array for {samples}")
            ak_arr_to_save = self.prep_jagged_array(events)
            os.makedirs(f"{samples_path}/jagged_arr/{samples}/", exist_ok=True)
            ak.to_parquet(ak_arr_to_save, f"{samples_path}/jagged_arr/{samples}/jagged_array.parquet")

            # save extra features
            #df = pd.DataFrame(ak.to_list(events))
            #extra_features = df[self.extra_features].values
            #extra_features = torch.tensor(extra_features, dtype=torch.float32)
            #torch.save(extra_features, f"{samples_path}/jagged_arr/{samples}/extra_features.pt")

            # save class labels
            class_events = np.zeros((len(ak_arr_to_save), 4))
            class_events[:, self.class_num[self.sample_to_class[samples]]] = 1.0
            class_events = torch.tensor(class_events, dtype=torch.float32)
            torch.save(class_events, f"{samples_path}/jagged_arr/{samples}/class_labels.pt")

    def extra_features_torch(self, sample_path):

        events, sum_genw_beforesel = self.load_parquet(sample_path, -1)
        df = pd.DataFrame(ak.to_list(events))
        extra_features = df[self.extra_features].values
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        
        return extra_features

        
    def merge_transform_scale(self, samples_path, scale_type="standard", pt_log = True): # samples_path

        # merge jagged_arr, extra_features, class_lables, relative_xsec from different samples
        print("INFO: Merging jagged arrays, class labels and relative cross section weights from different samples")
        lst_jagged_arr = [ak.from_parquet(f"{samples_path}/jagged_arr/{samples}/jagged_array.parquet") for samples in self.sample_to_class.keys()]

        node_feat = lst_jagged_arr[0].fields
        assert all(sample.fields == node_feat for sample in lst_jagged_arr), "All samples must have the same features."
        assert all(sample.fields == self.node_features for sample in lst_jagged_arr), "All samples must have the same features."

        comb_jagged_arr = {field: ak.concatenate([sample[field] for sample in lst_jagged_arr], axis=0) for field in node_feat}
        comb_jagged_arr = ak.zip(comb_jagged_arr)

        # clear lst_jagged_arr from meomory
        del lst_jagged_arr
        gc.collect()

        # merge extra features, class labels and relative cross section weights
        #comb_extra_features = torch.cat([torch.load(f"{samples_path}/jagged_arr/{sample}/extra_features.pt") for sample in self.sample_to_class.keys()], dim=0)
        comb_extra_features = torch.cat([self.extra_features_torch(f"{samples_path}/{sample}/nominal/") for sample in self.sample_to_class.keys()], dim=0)
        comb_class_labels = torch.cat([torch.load(f"{samples_path}/jagged_arr/{sample}/class_labels.pt") for sample in self.sample_to_class.keys()], dim=0)
        comb_relative_xsec = torch.cat([torch.load(f"{samples_path}/jagged_arr/{sample}/rel_xsec_weight.pt") for sample in self.sample_to_class.keys()], dim=0)


        # transform variables
        print("INFO: Transforming variables")
        if pt_log:
            comb_jagged_arr["pt"] = np.log(comb_jagged_arr["pt"])


        # sclae variables for training
        print("INFO: Scaling variables, using: ", scale_type)
        comb_jagged_arr_scaled = {}
        scale_values = {}

        for feat in self.node_features:  
            feat_arr = comb_jagged_arr[feat]

            if scale_type == "min_max":
                min_val = ak.min(ak.flatten(feat_arr))
                max_val = ak.max(ak.flatten(feat_arr))
                scale_values[feat] = {
                    "min": min_val,
                    "max": max_val,
                }
                comb_jagged_arr_scaled[feat] = self.min_max_scale(feat_arr, min_val, max_val)

            elif scale_type == "standard":
                mean = ak.mean(ak.flatten(feat_arr))
                std = ak.std(ak.flatten(feat_arr))
                scale_values[feat] = {
                    "mean": mean,
                    "std": std,
                }
                comb_jagged_arr_scaled[feat] = self.standardize(feat_arr, mean, std)

        comb_jagged_arr_scaled = ak.zip(comb_jagged_arr_scaled)

        return comb_jagged_arr_scaled, comb_extra_features, comb_class_labels, comb_relative_xsec, scale_values
    
    def prep_comp_graph_wo_self_loops(self, num_particles):
        edge_idx = []
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    edge_idx.append([i, j])
        return torch.tensor(edge_idx, dtype=torch.long).t().contiguous()

    def prep_Data_lst(self, comb_jagged_arr, comb_extra_features, comb_class_labels):

        print("INFO: Preparing Data list")
        concat_arr = ak.concatenate([comb_jagged_arr[self.node_features[i]][:, :, None] for i in range(len(self.node_features))], axis=-1)

        data_lst = [
            Data(
                x=torch.from_numpy(ak.to_numpy(event).astype(np.float32)),
                extra_feat=torch.reshape(extra_features, (1, -1)),
                edge_index=self.prep_comp_graph_wo_self_loops(len(event)),
                y=torch.reshape(class_label, (1, -1))
            )
            for event, extra_features, class_label in zip(concat_arr, comb_extra_features, comb_class_labels)
        ]

        return data_lst

    def prep_input_GNN(self, samples_path, out_path):

        if self.save_prelim_inputs:
            self.prelim_inputs(samples_path)

        comb_jagged_arr_scaled, comb_extra_features, comb_class_labels, comb_relative_xsec, scale_values = self.merge_transform_scale(samples_path)

        # save scale values
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/scale_values.json", "w") as f:
            json.dump(scale_values, f)

        # prepare and save Data list
        data_lst = self.prep_Data_lst(comb_jagged_arr_scaled, comb_extra_features, comb_class_labels)
        torch.save(data_lst, f"{out_path}/data_list.pt")

        # clear memory
        del comb_jagged_arr_scaled, comb_class_labels, comb_relative_xsec, scale_values
        gc.collect()

        return data_lst


if __name__ == "__main__":

    samples_path = "gnn_dummy_input_folder"

    prep = PrepareInputs()

    data_lst = prep.prep_input_GNN("gnn_dummy_input_folder", "gnn_data_lst")
    print("data_lst", data_lst[0])
    print(data_lst[0].x.shape)
    print("INFO: Done with preparing inputs for GNN")