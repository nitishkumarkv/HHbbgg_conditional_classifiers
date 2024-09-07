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

class PrepareInputs:
    def __init__(
        self
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = os.path.join(os.path.dirname(__file__), "input_variables.json")
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
        print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in file_list)

        sample_type = os.path.basename(os.path.dirname(path))
        events["sample_type"] = ak.Array([sample_type] * len(events))

        return events, sum_genw_beforesel

    def add_var(self, events):

        # add variables
        events["diphoton_PtOverM_ggjj"] = events.pt / events.HHbbggCandidate_mass
        events["dijet_PtOverM_ggjj"] = events.dijet_pt / events.HHbbggCandidate_mass

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
                
        for i in range(y_train.shape[1]):
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


    def train_test_split(self, X, Y, relative_weights, train_ratio=0.6):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, rel_w_train, rel_w_test_val = train_test_split(X, Y, relative_weights, train_size=train_ratio, shuffle=True, random_state=42)
        X_val, X_test, y_val, y_test, rel_w_val, rel_w_test = train_test_split(X_test_val, y_test_val, rel_w_test_val, train_size=0.5, shuffle=True, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test

    def standardize(self, X, mean, std):
        return (X - mean) / std

    def prep_input_for_mlp(self, samples_path, out_path, fill_nan = -999):

        #os.makedirs(out_path)

        comb_inputs = []

        for samples in self.sample_to_class.keys():

            events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{samples}/nominal/", -1)

            # get only valid events, cuts not applied in HiggsDNA
            events = events[(events.mass > 100) | (events.mass < 180)]
            events = events[(events.dijet_mass > 70) | (events.dijet_mass < 190)]

            # add more variables
            events = self.add_var(events)

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
        vars_for_log = vars_config["vars_for_log_transform"]

        X = comb_inputs[vars_for_training]
        Y = comb_inputs[[cls for cls in self.classes]]
        relative_weights = comb_inputs["rel_xsec_weight"]
        
        # perform log transformation for variables if needed
        for var in vars_for_log:
            X[var] = np.log(X[var])

        X = X.values
        Y = Y.values
        relative_weights = relative_weights.values
       
        # mask -999.0 to nan
        mask = (X < -998.0)
        X[mask] = np.nan

        X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test = self.train_test_split(X, Y, relative_weights)

        # get mean according to training data set
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0)
        
        # transform all data set
        X_train = self.standardize(X_train, mean, std)
        X_val = self.standardize(X_val, mean, std)
        X_test = self.standardize(X_test, mean, std)

        # replace NaN with fill_nan value
        X_train = np.nan_to_num(X_train, nan=fill_nan)
        X_val = np.nan_to_num(X_val, nan=fill_nan)
        X_test = np.nan_to_num(X_test, nan=fill_nan)

        #increase weights for signal events
        w_for_train = rel_w_train
        w_for_train[(y_train == [0, 0, 1, 0]).all(axis=1)] *= 10
        w_for_train[(y_train == [0, 0, 0, 1]).all(axis=1)] *= 10

        #increase weights for signal events
        w_for_train = rel_w_train
        w_for_train[(y_train == [0, 0, 1, 0]).all(axis=1)] *= 10
        w_for_train[(y_train == [0, 0, 0, 1]).all(axis=1)] *= 10

        class_weights_for_training = self.get_weights_for_training(y_train, rel_w_train)
        class_weights_for_val = self.get_weights_for_val_test(y_val, rel_w_val)
        class_weights_for_test = self.get_weights_for_val_test(y_test, rel_w_test)

        # save all the numpy arrays
        print("\n INFO: saving inputs for mlp")
        np.save(f"{out_path}/X_train", X_train)
        np.save(f"{out_path}/X_val", X_val)
        np.save(f"{out_path}/X_test", X_test)

        np.save(f"{out_path}/y_train", y_train)
        np.save(f"{out_path}/y_val", y_val)
        np.save(f"{out_path}/y_test", y_test)

        np.save(f"{out_path}/rel_w_train", rel_w_train)
        np.save(f"{out_path}/rel_w_val", rel_w_val)
        np.save(f"{out_path}/rel_w_test", rel_w_test)

        np.save(f"{out_path}/class_weights_for_training", class_weights_for_training)
        np.save(f"{out_path}/class_weights_for_val", class_weights_for_val)
        np.save(f"{out_path}/class_weights_for_test", class_weights_for_test)

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
    out.prep_input_for_mlp("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples_v2/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/training_inputs_for_mlp_wnorm")