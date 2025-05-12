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
        self,
        input_var_json: None,
        sample_to_class: None,
        class_num: None,
        classes: None
        ) -> None:
        self.model_type = "mlp"
        #self.input_var_json = os.path.join(os.path.dirname(__file__), "input_variables.json")
        #self.sample_to_class = {
        #    "GGJets": "is_non_resonant_bkg",
        #    "GJetPt20To40": "is_non_resonant_bkg",
        #    "GJetPt40": "is_non_resonant_bkg",
        #    "TTGG": "is_non_resonant_bkg",
        #    "ttHtoGG_M_125": "is_ttH_bkg",
        #    "GluGluHToGG_M_125": "is_single_H_bkg",
        #    "VBFHToGG_M_125": "is_single_H_bkg",
        #    "VHtoGG_M_125": "is_single_H_bkg",
        #    "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00": "is_GluGluToHH_sig",
        #    "VBFHHto2B2G_CV_1_C2V_1_C3_1": "is_VBFToHH_sig",
        #    }
        #self.class_num = {
        #    "is_non_resonant_bkg": 0,
        #    "is_ttH_bkg": 1,
        #    "is_single_H_bkg": 2,
        #    "is_GluGluToHH_sig": 3,
        #    "is_VBFToHH_sig": 4
        #}
        #self.classes = ["is_non_resonant_bkg", "is_ttH_bkg", "is_single_H_bkg", "is_GluGluToHH_sig", "is_VBFToHH_sig"]
        self.input_var_json = input_var_json
        self.sample_to_class = sample_to_class
        self.class_num = class_num
        self.classes = classes
        

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

    def load_parquet(self, path, columns, N_files=-1):
        columns = columns + ["mass", "nonRes_dijet_mass", "nonRes_has_two_btagged_jets", "weight", "pt", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "Res_mjj_regressed"]

        file_list = glob.glob(path + '*.parquet')
        if N_files == -1:
            file_list_ = file_list
        else:
            file_list_ = file_list[:N_files]

        events = ak.from_parquet(file_list_, columns=columns)
        #events = ak.from_parquet(file_list_)
        #print(events.fields)
        print(f"INFO: loaded parquet files from the path {path}")

        #events = ak.from_parquet(file_list_)
        #print(events.fields)
        #print(f"INFO: loaded parquet files from the path {path}")

        return events

    def add_var(self, events, era):

        collection = "nonres"
        # add variables
        if collection == "nonres":
            events["dijet_mass"] = events.nonRes_mjj_regressed
        else:
            events["dijet_mass"] = events.Res_mjj_regressed

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

    def plot_pt_variables(self, comb_inputs, vars_for_training):
        for var in vars_for_training:
            if 'pt' in var or 'Pt' in var:
                plt.figure(figsize=(10, 8))
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

    def train_test_split(self, X, Y, relative_weights, train_ratio=0.7, val_ratio=0.3):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, rel_w_train, rel_w_test_val = train_test_split(X, Y, relative_weights, train_size=train_ratio, shuffle=True, random_state=42)
        if (train_ratio + val_ratio) == 1.0:
            X_val, y_val, rel_w_val = X_test_val, y_test_val, rel_w_test_val
            X_test, y_test, rel_w_test = None, None, None
        else:
            X_val, X_test, y_val, y_test, rel_w_val, rel_w_test = train_test_split(X_test_val, y_test_val, rel_w_test_val, train_size=0.5, shuffle=True, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test

    def standardize(self, X, mean, std):
        return (X - mean) / std

    def min_max_scale(self, X, min, max):
        return (X - min) / (max - min)

    def corr_with_mgg_mjj(self, events, vars_for_training, out_path):

        corr_matrix = np.zeros([len(vars_for_training), 2])
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
        # plot the correlation matrix
        plt.figure(figsize=(18, len(vars_for_training)))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        # annotate the values
        for i in range(len(vars_for_training)):
            for j in range(2):
                # format the value to 2 decimal places
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='b')

        plt.xticks([0, 1], ['mass', 'nonRes_dijet_mass'])
        plt.yticks(range(len(vars_for_training)), vars_for_training)
        plt.colorbar()
        plt.savefig(f'{out_path}', dpi=300, )
        plt.clf()

    
    def preselection(self, events):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.dijet_mass > 70) & (events.dijet_mass < 190))
        #lead_mvaID_bool = ((events.lead_mvaID > 0.0439603) & (events.lead_isScEtaEB == True)) | ((events.lead_mvaID > -0.249526) & (events.lead_isScEtaEE == True))
        #sublead_mvaID_bool = ((events.sublead_mvaID > 0.0439603) & (events.sublead_isScEtaEB == True)) | ((events.sublead_mvaID > -0.249526) & (events.sublead_isScEtaEE == True))

        #events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]
        events = events[mass_bool & dijet_mass_bool]

        return events


            

    def prep_input_for_mlp(self, samples_path, out_path, fill_nan = -9):

        os.makedirs(out_path, exist_ok=True)

        comb_inputs = []

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]
        #vars_for_training = vars_config["vars"] + ["preEE", "postEE"]
        vars_for_training = vars_config["vars"] #+ ["era"]
        vars_for_log = vars_config["vars_for_log_transform"]

        #for era in ["preEE", "postEE"]:
        for era in ["preEE", "postEE", "preBPix", "postBPix"]:
        #for era in ["preBPix", "postBPix"]:
        #for era in ["2022preEE", "2022postEE"]:
            for samples in self.sample_to_class.keys():

                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", -1)

                #events, sum_genw_beforesel = self.load_parquet(f"{samples_path}/{era}/{samples}/nominal/", f"{samples_path}/{era}/{samples}/nominal/", -1)

                #if era == "preEE":
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/final_dump/{samples}/nominal/"
                #else:
                #    path_for_w = f"/net/scratch_cms3a/kasaraguppe/work/HDNA/HHbbgg_updates/for_production/HiggsDNA/postEE/{samples}/nominal/"
                #print(f"{samples_path}/sim/{era}/{samples}/nominal")
                sample_name_to_file_name = {
                    "GGJets": "GGJets_Rescaled_mbb_reg.parquet",
                    "DDQCDGJET": {"preEE": "DDQCDGJET_Rescaled_mbb_reg.parquet", "postEE": "DDQCDGJets_Rescaled_mbb_reg.parquet"},
                    "TTGG": "TTGG_merged_mbb_reg.parquet",
                    "ttHtoGG_M_125": "ttHtoGG_M-125_merged_mbb_reg.parquet",
                    "BBHto2G_M_125": "BBHto2G_M-125_merged_mbb_reg.parquet",
                    "GluGluHToGG_M_125": "GluGluHtoGG_M-125_merged_mbb_reg.parquet",
                    "VBFHToGG_M_125": "VBFHtoGG_M-125_merged_mbb_reg.parquet",
                    "VHtoGG_M_125": "VHtoGG_M-125_merged_mbb_reg.parquet",
                    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00_merged_mbb_reg.parquet",
                    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-5p00_kt-1p00_c2-0p00_merged_mbb_reg.parquet",
                    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-0p00_kt-1p00_c2-0p00_merged_mbb_reg.parquet",
                    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-2p45_kt-1p00_c2-0p00_merged_mbb_reg.parquet",
                }
                sample_name_to_file_name = {
                    "GGJets": "GGJets_Rescaled.parquet",
                    "DDQCDGJET": "DDQCDGJET_Rescaled.parquet",
                    "TTGG": "TTGG.parquet",
                    "ttHtoGG_M_125": "ttHtoGG.parquet",
                    "BBHto2G_M_125": "bbHToGG.parquet",
                    "GluGluHToGG_M_125": "GluGluHtoGG.parquet",
                    "VBFHToGG_M_125": "VBFHtoGG.parquet",
                    "VHtoGG_M_125": "VHtoGG.parquet",
                    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00.parquet",
                    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-5p00_kt-1p00_c2-0p00.parquet",
                    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-0p00_kt-1p00_c2-0p00.parquet",
                    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl-2p45_kt-1p00_c2-0p00.parquet",
                }

                #events = self.load_parquet(f"data/samples/{era}/{samples}/", vars_for_training, -1)
                vars_to_load = vars_for_training + ["mass", "nonRes_dijet_mass", "nonRes_has_two_btagged_jets", "weight", "pt", "nonRes_dijet_pt", "Res_dijet_pt", "Res_lead_bjet_pt", "Res_sublead_bjet_pt", "nonRes_HHbbggCandidate_mass", "Res_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "Res_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"]
                year = "2023" if "BPix" in era else "2022"
                if isinstance(sample_name_to_file_name[samples], dict):
                    events = ak.from_parquet(f"{samples_path}/Run3_{year}{era}/{sample_name_to_file_name[samples][era]}")
                else:
                    events = ak.from_parquet(f"{samples_path}/Run3_{year}{era}/{sample_name_to_file_name[samples]}")

                #if (era == "postEE") and (samples in ["TTGG", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]):
                #    print(f"data/samples/{era}/{samples}/")
                #    events = self.load_parquet(f"data/samples/{era}/{samples}/", vars_for_training, -1)
                #elif (era == "postEE") and (samples in ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"]):
                #    events = self.load_parquet(f"data/samples/{era}/GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00/", vars_for_training, -1)
                #elif (era == "postEE") and ("125" in samples):
                #    temp_samples_dict = {
                #        "ttHtoGG_M_125": "ttHToGG",
                #        "BBHto2G_M_125": "BBHto2G_M_125",
                #        "GluGluHToGG_M_125": "GluGluHToGG",
                #        "VBFHToGG_M_125": "VBFHToGG",
                #        "VHtoGG_M_125": "VHToGG"
                #    }
                #    temp_samples = temp_samples_dict[samples]
                #    print(f"{samples_path}/sim/{era}/{temp_samples}/nominal/")
                #    events = self.load_parquet(f"{samples_path}/sim/{era}/{temp_samples}/nominal/", vars_for_training, -1)
                #elif "DDQCDGJET" in samples:
                #    events = self.load_parquet(f"{samples_path}/sim/{era}/DDQCDGJets_to_be_updated/", vars_for_training, -1)
                #else: 
                #    events = self.load_parquet(f"{samples_path}/sim/{era}/{samples}/nominal/", vars_for_training, -1)

                ## event selection
                #diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
                ##dijet_mass_cut = ((events.nonRes_dijet_mass > 70) & (events.nonRes_dijet_mass < 190))
                #dijet_mass_cut = ((events.nonRes_mjj_regressed > 70) & (events.nonRes_mjj_regressed < 190))
                #nonRes = (events.nonRes_has_two_btagged_jets == True)
                #nBLoose = (events.nBLoose >= 2)
                #events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]
                ##events = events[diphoton_mass_cut & dijet_mass_cut & nonRes & nBLoose]
#
                #### we need to select only the events with lead and sublead mva score > -0.7
                #mva_score_cut = ((events.lead_mvaID > -0.7) & (events.sublead_mvaID > -0.7))
                #events = events[mva_score_cut]
                    
                # add more variables
                events = self.add_var(events, era)

                events = self.preselection(events)

                

                

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)

                print(f"INFO: Number of MC events in {samples} after selection for {era}: {len(events)}")
                print(f"INFO: Sum of weight_tot in {samples} after selection for {era}: {sum(events.weight_tot)}")

                # efficiency of slecting events with atleast two medium btagged jets
                # 'nBTight','nBMedium','nBLoose
                #print(f"INFO: {era}/{samples}/ Efficiency wrt after applying the diphoton mass cut, dijet mass cut and nonRes cut")
                #print(f"nBTight >= 2: {sum(events[events.nBTight >= 2].weight_tot) / sum(events.weight_tot)}")
                #print(f"nBMedium >= 2: {sum(events[events.nBMedium >= 2].weight_tot) / sum(events.weight_tot)}")
                #print(f"nBLoose >= 2: {sum(events[events.nBLoose >= 2].weight_tot) / sum(events.weight_tot)}")
                #print("")

                # add the bools for each class
                for cls in self.classes:  # first intialize everything to zero
                    events[cls] = ak.zeros_like(events.eta)

                #events["era"] = ak.zeros_like(events.eta) if era == "preEE" else ak.ones_like(events.eta)

                events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded
                comb_inputs.append(events)
                events["sample_type"] = samples

                # plot_correlation_matrix
                os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
                corr_out_path = f"{out_path}/correlation_matrix/{samples}_{era}.pdf"
                self.corr_with_mgg_mjj(events, vars_for_training, corr_out_path)

                # add era bools
                #for era_ in ["preEE", "postEE"]:
                #    events[f"{era_}"] = ak.zeros_like(events.eta)
                #events[era] = ak.ones_like(events.pt)
                
                
                #print(events["era"])

                #print(f"INFO: Number of events in {samples}: {len(events)}")

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

        class_weights_for_training = self.get_weights_for_training(y_train, rel_w_train)
        class_weights_for_train_no_aboslute = self.get_weights_for_val_test(y_train, rel_w_train)
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

        np.save(f"{out_path}/class_weights_for_training", class_weights_for_training)
        np.save(f"{out_path}/class_weights_for_train_no_aboslute", class_weights_for_train_no_aboslute)
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
    

    def plot_variables(self, comb_inputs, vars_for_training, plot_path):

        # add color scheme for each class
        color_list = [
            "#882255",  # Deep Burgundy
            "#117733",  # Dark Green
            "#332288",  # Deep Indigo
            "#AA4499",  # Soft Magenta
            "#DDCC77",  # Light Mustard
            "#661100",  # Dark Brown
            "#6699CC",  # Dusty Blue
            "#888888",  # Neutral Gray
            "#AA3377",  # Vivid Pink
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
            plt.legend(ncols=2, fontsize=10, loc='upper right')
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}.png")
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}_log.png")
            plt.clf()


                

                

if __name__ == "__main__":
    out = PrepareInputs()
    # out.prep_input("/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/", "/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models")
    #out.prep_input_for_mlp("samples_for_mlp_dummy/", "test_new")
    out.prep_input_for_mlp("samples/", "inputs_for_MLP_202411226/")
    