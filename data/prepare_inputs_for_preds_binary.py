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

class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = os.path.join(os.path.dirname(__file__), "input_variables.json")
        self.sample_to_class = {
            "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00": "is_GluGluToHH_sig",
            "VBFHHto2B2G_CV_1_C2V_1_C3_1": "is_VBFToHH_sig"
            }
        self.class_num = {
            "is_GluGluToHH_sig": 1,
            "is_VBFToHH_sig": 0
        }
        self.classes = ["is_GluGluToHH_sig", "is_VBFToHH_sig"]

    def load_vars(self, path):
        print(path)
        with open(path) as f:
            vars = json.load(f)
        return vars

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
#        print(f"INFO: loaded parquet files from the path {path}")
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
#
#        sample_type = os.path.basename(os.path.dirname(path))
#        #events["sample_type"] = ak.Array([sample_type] * len(events))
#
#        return events, sum_genw_beforesel

    def load_parquet(self, path, N_files=-1):

        file_list = glob.glob(path + '*.parquet')
        if N_files == -1:
            file_list_ = file_list
        else:
            file_list_ = file_list[:N_files]

        events = ak.from_parquet(file_list_)
        #print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        events = ak.from_parquet(file_list_)
        #print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        return events

    def add_var(self, events):

        # add variables
        events["diphoton_PtOverM_ggjj"] = events.pt / events.nonRes_HHbbggCandidate_mass
        events["dijet_PtOverM_ggjj"] = events.nonRes_dijet_pt / events.nonRes_HHbbggCandidate_mass

        return events

    def plot_pt_variables(self, comb_inputs, vars_for_training):
        for var in vars_for_training:
            if 'pt' in var or 'Pt' in var:
                plt.figure()
                for cls in self.classes:
                    class_events = comb_inputs[comb_inputs[cls] == 1]
                    data_to_plot = class_events[var]
                    binning = np.linspace(0, 10, 41)
                    Hist, Edges = np.histogram(ak.to_numpy(data_to_plot), bins=binning, density=True)
                    hep.histplot((Hist, Edges), histtype='step', label=f'{cls}')
                plt.xlabel(f"{var}")
                plt.ylabel("Events per bin")
                plt.xlim([0, 10])
                plt.ylim(bottom=0)
                plt.legend()
                hep.cms.label("Private work", data=False, year="2022", com=13.6)
                plt.savefig(f'/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/data/{var}_plot.png')
                plt.clf()

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

        class_weights_for_training = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            cls_bool = (y_train[:, i] == 1)
            abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
            class_weights_for_training = class_weights_for_training + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

        for i in range(y_train.shape[1]):
            print(f"(number of events: sum of class_weights_for_training) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training[y_train[:, i] == 1])})")

        return class_weights_for_training

    def get_weights_for_val_test(self, y_val, rel_w_val):

        class_weights_for_val = ak.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            cls_bool = (y_val[:, i] == 1)
            rel_xsec_weight_for_class = rel_w_val * cls_bool
            class_weights_for_val = class_weights_for_val + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

        return class_weights_for_val

    def train_test_split(self, X, Y, relative_weights, train_ratio=0.7):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, rel_w_train, rel_w_test_val = train_test_split(X, Y, relative_weights, train_size=train_ratio, shuffle=True, random_state=42)
        X_val, X_test, y_val, y_test, rel_w_val, rel_w_test = train_test_split(X_test_val, y_test_val, rel_w_test_val, train_size=0.6, shuffle=True, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test

    def standardize(self, X, mean, std):
        return (X - mean) / std

    def min_max_scale(self, X, min, max):
        return (X - min) / (max - min)

    def prep_input_for_mlp(self, samples_path, inputs_path, fill_nan = -9):

        out_path = f"{inputs_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)

        for era in ["preEE", "postEE"]:
            for samples in self.sample_to_class.keys():

                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", -1)
                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", -1)
                #if era == "preEE":
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/final_dump/{samples}/nominal/"
                #else:
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/postEE/{samples}/nominal/"
                events = self.load_parquet(f"{samples_path}/{era}/{samples}/", -1)

                # event selection
                diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
                dijet_mass_cut = ((events.nonRes_dijet_mass > 70) & (events.nonRes_dijet_mass < 190))
                nonRes = (events.nonRes_has_two_btagged_jets == True)
                events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]

                print(f"INFO: Number of events in {samples} after selection for {era}: {len(events)}")

                # add more variables
                events = self.add_var(events)

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)

                # add the bools for each class
                #for cls in self.classes:  # first intialize everything to zero
                #    events[cls] = ak.zeros_like(events.eta)
#
                #events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded

                # add era bools
                #for era_ in ["preEE", "postEE"]:
                #    events[f"{era_}"] = ak.zeros_like(events.eta)
                #events[era] = ak.ones_like(events.pt)
                events["era"] = ak.zeros_like(events.eta) if era == "preEE" else ak.ones_like(events.eta)

                

                #print(f"INFO: Number of events in {samples}: {len(events)}")
        
                comb_inputs = pd.DataFrame(ak.to_list(events))

                #for cls in self.classes:
                #    print("\n", f"INFO: Number of events in {cls}: {sum(comb_inputs[cls])}")

                # get the variables required for training
                vars_config = self.load_vars(self.input_var_json)[self.model_type]
                #vars_for_training = vars_config["vars"] + ["preEE", "postEE"]
                #vars_for_training = vars_config["vars"] + ["era"]
                with open(f"{inputs_path}/input_vars.txt", 'r') as f:
                    vars = json.load(f)
                vars_for_training = vars# + ["era"]

                vars_for_log = vars_config["vars_for_log_transform"]

                X = comb_inputs[vars_for_training]
                #Y = comb_inputs[[cls for cls in self.classes]]
                relative_weights = comb_inputs["rel_xsec_weight"]

                # perform log transformation for variables if needed
                for var in vars_for_log:
                    X[var] = np.log(X[var])


                X = X.values
                #Y = Y.values
                relative_weights = relative_weights.values


                # mask -999.0 to nan
                mask = (X < -998.0)
                X[mask] = np.nan

                #X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test = self.train_test_split(X, Y, relative_weights)

                # get mean according to training data set
                scale_file = f"{inputs_path}/mean_std_dict.pkl"
                with open(scale_file, 'rb') as f:
                    mean_std_dict = pickle.load(f)

                mean = mean_std_dict["mean"]
                std = mean_std_dict["std_dev"]

                # transform all data set
                X = self.standardize(X, mean, std)
                #X_val = self.standardize(X_val, mean, std)
                #X_test = self.standardize(X_test, mean, std)

                # replace NaN with fill_nan value
                X = np.nan_to_num(X, nan=fill_nan)
                #X_val = np.nan_to_num(X_val, nan=fill_nan)
                #X_test = np.nan_to_num(X_test, nan=fill_nan)

                #class_weights_for_training = self.get_weights_for_training(y_train, rel_w_train)
                #class_weights_for_train_no_aboslute = self.get_weights_for_val_test(y_train, rel_w_train)
                #class_weights_for_val = self.get_weights_for_val_test(y_val, rel_w_val)
                #class_weights_for_test = self.get_weights_for_val_test(y_test, rel_w_test)

                # save all the numpy arrays
                print("INFO: saving inputs for mlp")
                full_path_to_save = f"{out_path}/{era}/{samples}/"
                os.makedirs(full_path_to_save, exist_ok=True)

                np.save(f"{full_path_to_save}/X", X)
                np.save(f"{full_path_to_save}/rel_w", relative_weights)

                # also save the event
                ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                #np.save(f"{out_path}/X_val", X_val)
                #np.save(f"{out_path}/X_test", X_test)

                #np.save(f"{out_path}/y_train", y_train)
                #np.save(f"{out_path}/y_val", y_val)
                #np.save(f"{out_path}/y_test", y_test)

                #np.save(f"{out_path}/rel_w_val", rel_w_val)
                #np.save(f"{out_path}/rel_w_test", rel_w_test)

                #np.save(f"{out_path}/class_weights_for_training", class_weights_for_training)
                #np.save(f"{out_path}/class_weights_for_train_no_aboslute", class_weights_for_train_no_aboslute)
                #np.save(f"{out_path}/class_weights_for_val", class_weights_for_val)
                #np.save(f"{out_path}/class_weights_for_test", class_weights_for_test)

                # save the training mean ans std_dev. This will be used for standardizing data
                mean_std_dict = {
                    "mean": mean,
                    "std_dev": std
                }
                with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
                    pickle.dump(mean_std_dict, f)

        return 0

if __name__ == "__main__":
    out = PrepareInputs()
    # out.prep_input("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models")
    #out.prep_input_for_mlp("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples_v2/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm")
    out.prep_input_for_mlp("samples/", "../data/inputs_for_binary_MLP_20250121/")