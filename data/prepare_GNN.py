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

#    def load_parquet_simp(self, path, N_files=-1):
#
#        file_list = glob.glob(path + '*.parquet')
#        if N_files == -1:
#            file_list_ = file_list
#        else:
#            file_list_ = file_list[:N_files]
#
#        events = ak.from_parquet(file_list_)
#
#        sample_type = os.path.basename(os.path.dirname(path))
#        #events["sample_type"] = ak.Array([sample_type] * len(events))
#
#        return events
#
#    def load_parquet(self, path, path2, N_files=-1):
#
#        file_list = glob.glob(path + '*.parquet')
#        if N_files == -1:
#            file_list_ = file_list
#        else:
#            file_list_ = file_list[:N_files]
#
#        events = ak.from_parquet(file_list_)
#        #print(events.fields)
#        #print(f"INFO: loaded parquet files from the path {path}")
#
#        file_list_for_w = glob.glob(path2 + '*.parquet')
#        if N_files == -1:
#            file_list_for_w_ = file_list_for_w
#        else:
#            file_list_for_w_ = file_list_for_w[:N_files]
#
#        events = ak.from_parquet(file_list_)
#        #print(events.fields)
#        print(f"INFO: loaded parquet files from the path {path}")
#
#        sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in file_list_for_w)
#        print(f"INFO: sum of genw before selection: {sum_genw_beforesel}")
#
#        sample_type = os.path.basename(os.path.dirname(path))
#        #events["sample_type"] = ak.Array([sample_type] * len(events))
#
#        return events, sum_genw_beforesel

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

    def prelim_inputs(self, samples_path):

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

                # get relative weights according to cross section of the process and save them
                os.makedirs(f"{self.output_path}/jagged_arr/{era}/{samples}/", exist_ok=True)
                events = self.get_relative_xsec_weight(events, samples, era)
                torch.save(torch.tensor(events["rel_xsec_weight"], dtype=torch.float32), f"{self.output_path}/jagged_arr/{era}/{samples}/rel_xsec_weight.pt")

                # get jagged array
                print(f"INFO: Preparing jagged array for {samples} \n")
                ak_arr_to_save = self.prep_jagged_array(events)
                os.makedirs(f"{self.output_path}/jagged_arr/{era}/{samples}/", exist_ok=True)
                ak.to_parquet(ak_arr_to_save, f"{self.output_path}/jagged_arr/{era}/{samples}/jagged_array.parquet")

                # save extra features
                df = pd.DataFrame(ak.to_list(events))
                extra_features = df[self.extra_features].values
                extra_features = torch.tensor(extra_features, dtype=torch.float32)
                torch.save(extra_features, f"{self.output_path}/jagged_arr/{era}/{samples}/extra_features.pt")

                # save class labels
                class_events = np.zeros((len(ak_arr_to_save), len(self.class_num)))
                class_events[:, self.class_num[self.sample_to_class[samples]]] = 1.0
                class_events = torch.tensor(class_events, dtype=torch.float32)
                torch.save(class_events, f"{self.output_path}/jagged_arr/{era}/{samples}/class_labels.pt")

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

        
    def merge_transform_scale(self, samples_path, scale_type="standard", pt_log = True): # samples_path

        # merge jagged_arr, extra_features, class_lables, relative_xsec from different samples
        print("INFO: Merging jagged arrays, class labels and relative cross section weights from different samples")
        lst_jagged_arr_preEE = [ak.from_parquet(f"{self.output_path}/jagged_arr/preEE/{samples}/jagged_array.parquet") for samples in self.sample_to_class.keys()]
        lst_jagged_arr_postEE = [ak.from_parquet(f"{self.output_path}/jagged_arr/postEE/{samples}/jagged_array.parquet") for samples in self.sample_to_class.keys()]
        lst_jagged_arr = lst_jagged_arr_preEE + lst_jagged_arr_postEE

        node_feat = lst_jagged_arr[0].fields
        assert all(sample.fields == node_feat for sample in lst_jagged_arr), "All samples must have the same features."
        assert all(sample.fields == self.node_features for sample in lst_jagged_arr), "All samples must have the same features."

        comb_jagged_arr = {field: ak.concatenate([sample[field] for sample in lst_jagged_arr], axis=0) for field in node_feat}
        comb_jagged_arr = ak.zip(comb_jagged_arr)

        # clear lst_jagged_arr from meomory
        del lst_jagged_arr
        gc.collect()

        # merge extra features, class labels and relative cross section weights
        comb_extra_features_preEE = [self.extra_features_torch(f"{self.output_path}/jagged_arr/preEE/{sample}/extra_features.pt", f"{samples_path}/preEE/{sample}/nominal/") for sample in self.sample_to_class.keys()]
        comb_extra_features_postEE = [self.extra_features_torch(f"{self.output_path}/jagged_arr/postEE/{sample}/extra_features.pt", f"{samples_path}/postEE/{sample}/nominal/") for sample in self.sample_to_class.keys()]
        comb_extra_features = torch.cat(comb_extra_features_preEE + comb_extra_features_postEE, dim=0)

        comb_class_labels_preEE = [torch.load(f"{self.output_path}/jagged_arr/preEE/{sample}/class_labels.pt") for sample in self.sample_to_class.keys()]
        comb_class_labels_postEE = [torch.load(f"{self.output_path}/jagged_arr/postEE/{sample}/class_labels.pt") for sample in self.sample_to_class.keys()]
        comb_class_labels = torch.cat(comb_class_labels_preEE + comb_class_labels_postEE, dim=0)

        comb_relative_xsec_preEE = [torch.load(f"{self.output_path}/jagged_arr/preEE/{sample}/rel_xsec_weight.pt") for sample in self.sample_to_class.keys()]
        comb_relative_xsec_postEE = [torch.load(f"{self.output_path}/jagged_arr/postEE/{sample}/rel_xsec_weight.pt") for sample in self.sample_to_class.keys()]
        comb_relative_xsec = torch.cat(comb_relative_xsec_preEE + comb_relative_xsec_postEE, dim=0)

        # transform variables
        print("INFO: Transforming variables")
        if pt_log:
            comb_jagged_arr["pt"] = np.log(comb_jagged_arr["pt"])

        # train val split
        comb_jagged_arr_train, comb_jagged_arr_val, comb_extra_features_train, comb_extra_features_val, comb_class_labels_train, comb_class_labels_val, comb_relative_xsec_train, comb_relative_xsec_val = self.train_test_split(comb_jagged_arr, comb_extra_features, comb_class_labels, comb_relative_xsec)


        ############# scaling #############
        # scale node variables for training
        print("INFO: Scaling variables, using: ", scale_type)
        comb_jagged_arr_train_scaled = {}
        scale_values = {}

        for feat in self.node_features:
            feat_arr = comb_jagged_arr_train[feat]

            if scale_type == "min_max":
                min_val = ak.min(ak.flatten(feat_arr))
                max_val = ak.max(ak.flatten(feat_arr))
                scale_values[feat] = {
                    "min": min_val,
                    "max": max_val,
                }
                comb_jagged_arr_train_scaled[feat] = self.min_max_scale(feat_arr, min_val, max_val)

            elif scale_type == "standard":
                mean = ak.mean(ak.flatten(feat_arr))
                std = ak.std(ak.flatten(feat_arr))
                scale_values[feat] = {
                    "mean": mean,
                    "std": std,
                }
                comb_jagged_arr_train_scaled[feat] = self.standardize(feat_arr, mean, std)

        comb_jagged_arr_train_scaled = ak.zip(comb_jagged_arr_train_scaled)

        # scale node variables for validation
        comb_jagged_arr_val_scaled = {}
        for feat in self.node_features:
            feat_arr = comb_jagged_arr_val[feat]
            if scale_type == "min_max":
                comb_jagged_arr_val_scaled[feat] = self.min_max_scale(feat_arr, scale_values[feat]["min"], scale_values[feat]["max"])
            elif scale_type == "standard":
                comb_jagged_arr_val_scaled[feat] = self.standardize(feat_arr, scale_values[feat]["mean"], scale_values[feat]["std"])

        
        # scale extra features for training
        if scale_type == "min_max":
            extra_features_train_min = torch.min(comb_extra_features_train, dim=0)[0]
            extra_features_train_max = torch.max(comb_extra_features_train, dim=0)[0]
            extra_features_scale_values = {
                "min": extra_features_train_min.numpy().tolist(),
                "max": extra_features_train_max.numpy().tolist()
            }
            comb_extra_features_train_scaled = self.min_max_scale(comb_extra_features_train, extra_features_train_min, extra_features_train_max)
            comb_extra_features_val_scaled = self.min_max_scale(comb_extra_features_val, extra_features_train_min, extra_features_train_max)
        elif scale_type == "standard":
            extra_features_train_mean = torch.mean(comb_extra_features_train, dim=0)
            extra_features_train_std = torch.std(comb_extra_features_train, dim=0)
            extra_features_scale_values = {
                "mean": extra_features_train_mean.numpy().tolist(),
                "std": extra_features_train_std.numpy().tolist()
            }
            comb_extra_features_train_scaled = self.standardize(comb_extra_features_train, extra_features_train_mean, extra_features_train_std)
            comb_extra_features_val_scaled = self.standardize(comb_extra_features_val, extra_features_train_mean, extra_features_train_std)


        # get weights for training and validation
        print("INFO: Getting weights for training and validation")
        class_weights_for_training = self.get_weights_for_training(comb_class_labels_train, comb_relative_xsec_train)
        class_weights_for_val = self.get_weights_for_val_test(comb_class_labels_val, comb_relative_xsec_val)
        class_weights_for_training_no_absolute = self.get_weights_for_val_test(comb_class_labels_train, comb_relative_xsec_train)

        #return comb_jagged_arr_scaled, comb_extra_features, comb_class_labels, comb_relative_xsec, scale_values
        return comb_jagged_arr_train_scaled, comb_extra_features_train_scaled, comb_class_labels_train, class_weights_for_training, class_weights_for_training_no_absolute, comb_relative_xsec_train, scale_values, extra_features_scale_values, comb_jagged_arr_val_scaled, comb_extra_features_val_scaled, comb_class_labels_val, class_weights_for_val, comb_relative_xsec_val

    
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

    def prep_input_GNN(self, samples_path, out_path):

        self.output_path = out_path
        os.makedirs(out_path, exist_ok=True)

        if self.save_prelim_inputs:
            self.prelim_inputs(samples_path)
        
        

        #comb_jagged_arr_scaled, comb_extra_features, comb_class_labels, comb_relative_xsec, scale_values = self.merge_transform_scale(samples_path)
        comb_jagged_arr_train_scaled, comb_extra_features_train_scaled, comb_class_labels_train, class_weights_for_training, class_weights_for_training_no_absolute, comb_relative_xsec_train, scale_values, extra_features_scale_values, comb_jagged_arr_val_scaled, comb_extra_features_val_scaled, comb_class_labels_val, class_weights_for_val, comb_relative_xsec_val = self.merge_transform_scale(samples_path)

        # save scale values
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/scale_values.json", "w") as f:
            json.dump(scale_values, f)
        print(extra_features_scale_values)
        print(type(extra_features_scale_values))
        with open(f"{out_path}/extra_features_scale_values.json", "w") as f:
            json.dump(extra_features_scale_values, f)

        # prepare and save Data list for train and val
        print("INFO: Preparing Data list for training")
        data_lst_train = self.prep_Data_lst(comb_jagged_arr_train_scaled, comb_extra_features_train_scaled, comb_class_labels_train, class_weights_for_training)
        print("INFO: Saving Data list for training")
        torch.save(data_lst_train, f"{out_path}/data_lst_train.pt")

        print("INFO: Preparing Data list for validation")
        data_lst_val = self.prep_Data_lst(comb_jagged_arr_val_scaled, comb_extra_features_val_scaled, comb_class_labels_val, class_weights_for_val)
        print("INFO: Saving Data list for validation")
        torch.save(data_lst_val, f"{out_path}/data_lst_val.pt")

        # save relative cross section weights
        torch.save(comb_relative_xsec_train, f"{out_path}/relative_xsec_train.pt")
        torch.save(comb_relative_xsec_val, f"{out_path}/relative_xsec_val.pt")
        torch.save(class_weights_for_training_no_absolute, f"{out_path}/class_weights_for_training_no_absolute.pt")

        # also save the node features and extra features list
        with open(f"{out_path}/node_features.json", "w") as f:
            json.dump(self.node_features, f)
        with open(f"{out_path}/extra_features.json", "w") as f:
            json.dump(self.extra_features, f)


        # clear memory
        del comb_jagged_arr_train_scaled, comb_extra_features_train_scaled, comb_class_labels_train, class_weights_for_training, class_weights_for_training_no_absolute, comb_relative_xsec_train, scale_values, extra_features_scale_values, comb_jagged_arr_val_scaled, comb_extra_features_val_scaled, comb_class_labels_val, class_weights_for_val, comb_relative_xsec_val
        gc.collect()

        return data_lst_train, data_lst_val


if __name__ == "__main__":

    samples_path = "samples/"
    #samples_path = "samples_for_mlp_dummy"

    prep = PrepareInputs()

    data_lst_train, data_lst_val = prep.prep_input_GNN(samples_path, "gnn_inputs_20241203_new")
    print("data_lst", data_lst_train[0])
    print(data_lst_train[0].x.shape)
    print(data_lst_train[0].keys())
    print("INFO: Done with preparing inputs for GNN")