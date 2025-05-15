import awkward as ak
import numpy as np
import yaml
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
        self,
        input_var_json: None,
        sample_to_class: None,
        class_num: None,
        classes: None,
        samples_info: None,
        random_seed: None,
        outpath: None,
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = input_var_json
        self.sample_to_class = sample_to_class
        self.class_num = class_num
        self.classes = classes
        self.samples_info = samples_info
        self.random_seed = random_seed
        self.outpath = outpath
        

    def load_vars(self, path):
        print(path)
        with open(path, 'r') as f:
            vars = yaml.safe_load(f)
        return vars

    def load_parquet(self, path, columns, N_files=-1):
        columns = columns + ["mass", "nonRes_dijet_mass", "nonRes_has_two_btagged_jets", "weight", "pt", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "Res_mjj_regressed"]

        file_list = glob.glob(path + '*.parquet')
        if N_files == -1:
            file_list_ = file_list
        else:
            file_list_ = file_list[:N_files]

        events = ak.from_parquet(file_list_, columns=columns)
        print(f"INFO: loaded parquet files from the path {path}")

        return events

    def add_var(self, events, era):

        events["nonRes_lead_bjet_pt_over_M_regressed"] = events.nonRes_lead_bjet_pt / events.dijet_mass
        events["nonRes_sublead_bjet_pt_over_M_regressed"] = events.nonRes_sublead_bjet_pt / events.dijet_mass
        events["nonRes_diphoton_PtOverM_ggjj"] = events.pt / events.nonRes_HHbbggCandidate_mass
        events["nonRes_dijet_PtOverM_ggjj"] = events.nonRes_dijet_pt / events.nonRes_HHbbggCandidate_mass
        
        events["Res_lead_bjet_pt_over_M_regressed"] = events.Res_lead_bjet_pt / events.dijet_mass
        events["Res_sublead_bjet_pt_over_M_regressed"] = events.Res_sublead_bjet_pt / events.dijet_mass
        events["Res_diphoton_PtOverM_ggjj"] = events.pt / events.Res_HHbbggCandidate_mass
        events["Res_dijet_PtOverM_ggjj"] = events.Res_dijet_pt / events.Res_HHbbggCandidate_mass

        # add deltaR between lead and sublead photon
        events["deltaR_gg"] = np.sqrt((events.lead_eta - events.sublead_eta) ** 2 + (events.lead_phi - events.sublead_phi) ** 2)
        # add deltaR between lead and sublead bjet
        if era == "preEE":
            events["era"] = 0
        elif era == "postEE":
            events["era"] = 1
        elif era == "preBPix":
            events["era"] = 2
        elif era == "postBPix":
            events["era"] = 3

        return events

    def get_relative_xsec_weight(self, events, sample_type, era):

        dict_xsec = {
            "GGJets": 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.02391e3,  # cross sectio of TTGG 0.01696, copilot: 0.502
            "ttHtoGG_M_125": 0.5700e3 * 0.00227,  # cross sectio of ttH * BR(HToGG)
            "BBHto2G_M_125": 0.4385e3 * 0.00227,  # cross sectio of BBH * BR(HToGG)
            "GluGluHToGG_M_125": 52.23e3 * 0.00227,  # cross sectio of GluGluHToGG * BR(HToGG)
            "VBFHToGG_M_125": 4.078e3 * 0.00227,
            "VHtoGG_M_125": 2.4009e3 * 0.00227,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.034e3 * 0.00227 * 0.582 * 2,#0.02964e3 * 0.00227 * 0.582 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HToGG) * 2 for two combination ### have to recheck if this is correct. 
            "VBFHHto2B2G_CV_1_C2V_1_C3_1": 0.00173e3 * 0.00227 * 0.582 * 2,  # cross sectio of VBFToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct.
            "DDQCDGJET": 1.0,
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.08373e3 * 0.00227 * 0.582 * 2,
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.06531e3 * 0.00227 * 0.582 * 2,
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.01285e3 * 0.00227 * 0.582 * 2
        }
        luminosities = {
        "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
        "postEE": 26.67,  # Integrated luminosity for postEE in fb^-1
        "preBPix": 17.794,  # Integrated luminosity for preEE in fb^-1
        "postBPix": 9.451  # Integrated luminosity for postEE in fb^-1
        }

        lumi = luminosities[era]
        if sample_type == "DDQCDGJET":
            lumi = 1.0

        events["rel_xsec_weight"] = (events.weight) * dict_xsec[sample_type] * lumi
        events["weight_tot"] = (events.weight) * dict_xsec[sample_type] * lumi

        return events

    def get_weights_for_training(self, y_train, rel_w_train):

        true_class_weights = ak.zeros_like(rel_w_train)
        class_weights_for_training_abs = ak.zeros_like(rel_w_train)
        class_weights_only_positive = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            cls_bool = (y_train[:, i] == 1)

            rel_xsec_weight_for_class = rel_w_train * cls_bool
            true_class_weights = true_class_weights + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

            abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
            class_weights_for_training_abs = class_weights_for_training_abs + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

            only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0)
            class_weights_only_positive = class_weights_only_positive + (only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class))

        for i in range(y_train.shape[1]):
            print(f"(number of events: sum of class_weights_for_training_abs) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training_abs[y_train[:, i] == 1])})")
            print(f"(number of events: sum of class_weights_only_positive) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_only_positive[y_train[:, i] == 1])})")
            print(f"(number of events: sum of true_class_weights) for class number {i+1} = ({sum(y_train[:, i])}: {sum(true_class_weights[y_train[:, i] == 1])})")
        return true_class_weights, class_weights_for_training_abs, class_weights_only_positive

    def get_weights_for_val_test(self, y_val, rel_w_val):

        class_weights_for_val = ak.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            cls_bool = (y_val[:, i] == 1)
            rel_xsec_weight_for_class = rel_w_val * cls_bool
            class_weights_for_val = class_weights_for_val + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

        return class_weights_for_val

    def train_test_split(self, X, Y, relative_weights, train_ratio=0.7, val_ratio=0.3):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, rel_w_train, rel_w_test_val = train_test_split(X, Y, relative_weights, train_size=train_ratio, shuffle=True, random_state=self.random_seed)
        if (train_ratio + val_ratio) == 1.0:
            X_val, y_val, rel_w_val = X_test_val, y_test_val, rel_w_test_val
            X_test, y_test, rel_w_test = None, None, None
        else:
            X_val, X_test, y_val, y_test, rel_w_val, rel_w_test = train_test_split(X_test_val, y_test_val, rel_w_test_val, train_size=0.5, shuffle=True, random_state=self.random_seed)

        return X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test

    def standardize(self, X, mean, std):
        return (X - mean) / std

    def min_max_scale(self, X, min, max):
        return (X - min) / (max - min)

    def corr_with_mgg_mjj(self, events, vars_for_training, out_path):

        corr_matrix = np.zeros([len(vars_for_training), 5])
        for i in range(len(vars_for_training)):
            var = vars_for_training[i]
            # calculate correlation with mgg and mjj, do not include -999 values
            mask = ((events[var] > -998.0) & (events.mass > -998.0))
            mass = events.mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 0] = np.corrcoef(mass, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.nonRes_dijet_mass > -998.0))
            nonRes_dijet_mass = events.nonRes_dijet_mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 1] = np.corrcoef(nonRes_dijet_mass, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.nonRes_mjj_regressed > -998.0))
            nonRes_mjj_regressed = events.nonRes_mjj_regressed[mask]
            var_values = events[var][mask]
            corr_matrix[i, 2] = np.corrcoef(nonRes_mjj_regressed, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.Res_dijet_mass > -998.0))
            Res_dijet_mass = events.Res_dijet_mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 3] = np.corrcoef(Res_dijet_mass, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.Res_mjj_regressed > -998.0))
            Res_mjj_regressed = events.Res_mjj_regressed[mask]
            var_values = events[var][mask]
            corr_matrix[i, 4] = np.corrcoef(Res_mjj_regressed, var_values)[0, 1]

        # plot the correlation matrix
        plt.figure(figsize=(18, len(vars_for_training)))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        # annotate the values
        for i in range(len(vars_for_training)):
            for j in range(2):
                # format the value to 2 decimal places
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='b')

        plt.xticks([0, 1], ['mass', 'nonRes_dijet_mass', 'nonRes_mjj_regressed', 'Res_dijet_mass', 'Res_mjj_regressed'])
        plt.yticks(range(len(vars_for_training)), vars_for_training)
        plt.colorbar()
        plt.savefig(f'{out_path}', dpi=300, )
        plt.clf()

    
    def preselection(self, events):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.Res_mjj_regressed > 70) & (events.Res_mjj_regressed < 190))
        events = events[mass_bool & dijet_mass_bool]

        return events
    
    def plot_variables(self, comb_inputs, vars_for_training, plot_path):

        # add color scheme for each class
        color_list = [
            "#882255",  # Deep Burgundy
            "#117733",  # Dark Green
            "#332288",  # Deep Indigo
            "#AA4499",  # Soft Magenta
            "#DDCC77",  # Light Mustard
            "#6699CC",  # Dusty Blue
            "#888888",  # Neutral Gray
            "black",  # Vivid Pink
            "#44AA77",  # Medium Teal
            "#774411",  # Earthy Brown
            "#DDDDDD",  # Pale Gray
            "#E69F00",  # Orange
            "#56B4E9",  # Sky Blue
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#0072B2",  # Royal Blue
            "#D55E00",  # Vermillion
            "#CC79A7",  # Reddish Purple
            "#000000",  # Black
            "#CC6677",  # Muted Red
            "#88CCEE",  # Light Blue
            "#44AA99",  # Teal
            "#999999",  # Medium Gray
            "#FF8A50",  # Darker Peach
            "#FFB300",  # Golden Yellow
            "#66BB6A",  # Rich Green
            "#42A5F5",  # Deeper Sky Blue
            "#AB47BC",  # Strong Lavender Purple
            "#EC407A",  # Deeper Pink
            "#C0CA33",  # Darker Lime
            "#26A69A",  # Deep Teal
            "#FB8C00",  # Vibrant Orange
            "#795548",  # Deep Brown
            "#757575",  # Medium Gray
            "#8E24AA",  # Medium-Dark Purple
        ]

        for var in vars_for_training:
            plt.figure(figsize=(10, 8))
            sample_num = 0
            data_to_plot_dict = {}
            range_list = []
            for sample in self.sample_to_class.keys():
                if sample in ["DDQCDGJET", ]:
                    continue

                sample_events = comb_inputs[comb_inputs["sample_type"] == sample]
                data_to_plot = sample_events[var]
                mask = (data_to_plot > -998.0)
                data_to_plot = data_to_plot[mask]
                data_to_plot_dict[sample] = data_to_plot

                if range_list == []:
                    range_list = [min(data_to_plot), max(data_to_plot)]
                else:
                    range_list[0] = min(range_list[0], min(data_to_plot))
                    range_list[1] = max(range_list[1], max(data_to_plot))

            for sample in self.sample_to_class.keys():
                if sample in ["DDQCDGJET", ]:
                    continue
                data_to_plot = data_to_plot_dict[sample]
                
                # use mplhep to plot the histogram
                hist_ = np.histogram(ak.to_numpy(data_to_plot), bins=30, range=range_list, density=True)
                plt.style.use(hep.style.CMS)
                hep.histplot(hist_, histtype='step', label=sample, color=color_list[sample_num], linestyle='solid', linewidth=1.5)
                sample_num += 1

            plt.xlabel(f"{var}")
            plt.ylabel("a.u.")
            plt.legend(ncols=2, fontsize=13, loc='upper right')
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}.png")
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}_log.png")
            plt.clf()


    def prep_inputs_for_training(self, fill_nan = -9):

        out_path = self.outpath
        os.makedirs(self.out_path, exist_ok=True)

        comb_inputs = []

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        vars_for_training = vars_config["vars"] 
        vars_for_log = vars_config["vars_for_log_transform"]

        for era in self.samples_info["eras"]:
            for samples in self.sample_to_class.keys():

                #events = self.load_parquet(f"data/samples/{era}/{samples}/", vars_for_training, -1)
                vars_to_load = vars_for_training + ["mass", "nonRes_dijet_mass", "nonRes_has_two_btagged_jets", "weight", "pt", "nonRes_dijet_pt", "Res_dijet_pt", "Res_lead_bjet_pt", "Res_sublead_bjet_pt", "nonRes_HHbbggCandidate_mass", "Res_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "Res_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"]

                samples_path = self.samples_info["samples_path"]
                parquet_path = self.samples_info[era][samples]
                events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)

                # add more variables
                events = self.add_var(events, era)

                events = self.preselection(events)

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)

                print(f"INFO: Number of MC events in {samples} after selection for {era}: {len(events)}")
                print(f"INFO: Sum of weight_tot in {samples} after selection for {era}: {sum(events.weight_tot)}")

                # add the bools for each class
                for cls in self.classes:  # first intialize everything to zero
                    events[cls] = ak.zeros_like(events.eta)

                events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded
                comb_inputs.append(events)
                events["sample_type"] = samples

                # plot_correlation_matrix
                os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
                corr_out_path = f"{out_path}/correlation_matrix/{samples}_{era}.pdf"
                self.corr_with_mgg_mjj(events, vars_for_training, corr_out_path)

        print("INFO: Combining all the samples")
        comb_inputs = ak.concatenate(comb_inputs, axis=0)

        comb_inputs = pd.DataFrame(ak.to_list(comb_inputs))

        plot_path = f"{out_path}/var_plots/"
        os.makedirs(plot_path, exist_ok=True)
        self.plot_variables(comb_inputs, vars_for_training, plot_path)

        for cls in self.classes:
            print("\n", f"INFO: Number of events in {cls}: {sum(comb_inputs[cls])}")



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

        # replace NaN with fill_nan value
        X_train = np.nan_to_num(X_train, nan=fill_nan)
        X_val = np.nan_to_num(X_val, nan=fill_nan)

        true_class_weights, class_weights_for_training_abs, class_weights_only_positive = self.get_weights_for_training(y_train, rel_w_train)
        class_weights_for_val = self.get_weights_for_val_test(y_val, rel_w_val)

        if X_test is not None:
            X_test = self.standardize(X_test, mean, std)
            X_test = np.nan_to_num(X_test, nan=fill_nan)
            class_weights_for_test = self.get_weights_for_val_test(y_test, rel_w_test)
        

        # save all the numpy arrays
        print("\n INFO: saving inputs for mlp")
        # save str of input variables
        with open(f"{out_path}/input_vars.txt", 'w') as f:
            json.dump(vars_for_training, f)
        
        np.save(f"{out_path}/X_train", X_train)
        np.save(f"{out_path}/X_val", X_val)
        
        np.save(f"{out_path}/y_train", y_train)
        np.save(f"{out_path}/y_val", y_val)

        np.save(f"{out_path}/rel_w_train", rel_w_train)
        np.save(f"{out_path}/rel_w_val", rel_w_val)

        np.save(f"{out_path}/true_class_weights", true_class_weights)
        np.save(f"{out_path}/class_weights_for_training_abs", class_weights_for_training_abs)
        np.save(f"{out_path}/class_weights_only_positive", class_weights_only_positive)
        np.save(f"{out_path}/class_weights_for_val", class_weights_for_val)
        
        if X_test is not None:
            np.save(f"{out_path}/X_test", X_test)
            np.save(f"{out_path}/rel_w_test", rel_w_test)
            np.save(f"{out_path}/y_test", y_test)
            np.save(f"{out_path}/class_weights_for_test", class_weights_for_test)

        # save the training mean ans std_dev. This will be used for standardizing data
        mean_std_dict = {
            "mean": mean,
            "std_dev": std
        }
        with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
            pickle.dump(mean_std_dict, f)

        return 0
    
    def prep_inputs_for_prediction_sim(self, fill_nan = -9):

        samples_info = self.samples_info
        inputs_path = self.outpath
        out_path = f"{inputs_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)
        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars

        vars_for_log = vars_config["vars_for_log_transform"]

        for era in samples_info["eras"]:
            for samples in samples_info[era].keys():
                
                samples_path = samples_info["samples_path"]
                parquet_path = samples_info[era][samples]
                events = ak.from_parquet(f"{samples_path}/{parquet_path}")
                
                print(f"INFO: Number of events in {samples} after selection for {era}: {len(events)}")

                # add more variables
                events = self.add_var(events, era)

                # add preselection
                # events = self.preselection(events)

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)
                
                comb_inputs = pd.DataFrame(ak.to_list(events))

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

                # get mean according to training data set
                scale_file = f"{inputs_path}/mean_std_dict.pkl"
                with open(scale_file, 'rb') as f:
                    mean_std_dict = pickle.load(f)

                mean = mean_std_dict["mean"]
                std = mean_std_dict["std_dev"]

                # transform all data set
                X = self.standardize(X, mean, std)

                # replace NaN with fill_nan value
                X = np.nan_to_num(X, nan=fill_nan)

                # save all the numpy arrays
                print("INFO: saving inputs for mlp")
                full_path_to_save = f"{out_path}/{era}/{samples}/"
                os.makedirs(full_path_to_save, exist_ok=True)

                np.save(f"{full_path_to_save}/X", X)
                np.save(f"{full_path_to_save}/rel_w", relative_weights)

                # also save the event
                ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                # save the training mean ans std_dev. This will be used for standardizing data
                mean_std_dict = {
                    "mean": mean,
                    "std_dev": std
                }
                with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
                    pickle.dump(mean_std_dict, f)

        return 0
    
    def prep_inputs_for_prediction_data(self, samples_path, inputs_path, fill_nan = -9):

        samples_info = self.samples_info
        inputs_path = self.outpath
        out_path = f"{inputs_path}/individual_samples_data/"
        os.makedirs(out_path, exist_ok=True)

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]
        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars

        vars_for_log = vars_config["vars_for_log_transform"]

        comb_inputs = []


        #for era in ["preEE"]:
        for data in samples_info["data"]:

            events = ak.from_parquet(f"{self.samples_path}/{data}")

            sample_to_era = {"2022_EraE": "postEE", 
                                 "2022_EraF": "postEE", 
                                 "2022_EraG": "postEE", 
                                 "2022_EraC": "preEE", 
                                 "2022_EraD": "preEE",
                                 "2023_EraCv1to3": "preBPix", 
                                 "2023_EraCv4": "preBPix", 
                                 "2023_EraD": "postBPix"}

            # add more variables
            events = self.add_var(events, sample_to_era[data])

            # add preselection
            # events = self.preselection(events)


            comb_inputs = pd.DataFrame(ak.to_list(events))

            X = comb_inputs[vars_for_training]

            # perform log transformation for variables if needed
            for var in vars_for_log:
                X[var] = np.log(X[var])


            X = X.values



            # mask -999.0 to nan
            mask = (X < -998.0)
            X[mask] = np.nan


            # get mean according to training data set
            scale_file = f"{inputs_path}/mean_std_dict.pkl"
            with open(scale_file, 'rb') as f:
                mean_std_dict = pickle.load(f)

            mean = mean_std_dict["mean"]
            std = mean_std_dict["std_dev"]

            # transform all data set
            X = self.standardize(X, mean, std)
            X = np.nan_to_num(X, nan=fill_nan)


            # save all the numpy arrays
            print("INFO: saving inputs for mlp")
            #full_path_to_save = f"{out_path}/"
            full_path_to_save = f"{out_path}/{samples}/"
            os.makedirs(full_path_to_save, exist_ok=True)

            np.save(f"{full_path_to_save}/X", X)

            ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

        return 0
    

    


    