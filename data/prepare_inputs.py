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
import gc
from vector import register_awkward
register_awkward()

class PrepareInputs:
    def __init__(
        self,
        input_var_json: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        outpath: Optional[Dict[str, Any]] = None,
        predict_parquet_info: Optional[Dict[str, Any]] = None,
        ) -> None:
        self.model_type = "mlp"
        self.input_var_json = input_var_json
        self.training_info = training_info
        self.outpath = outpath
        self.predict_parquet_info = predict_parquet_info
        
        if self.training_info is not None:
            self.sample_to_class = self.training_info["sample_to_class"]
            self.classes = self.training_info["classes"]
            self.random_seed = self.training_info["random_seed"]
            self.weight_scheme_process = self.training_info["weight_scheme_process"]
        self.fill_nan = -9

        #self.extra_vars = ["mass", "nonRes_dijet_mass", "Res_dijet_mass", "nonRes_has_two_btagged_jets", "weight", "pt", "nonRes_dijet_pt", "Res_dijet_pt", "Res_lead_bjet_pt", "Res_sublead_bjet_pt", "Res_lead_bjet_ptPNetCorr", "Res_sublead_bjet_ptPNetCorr", "nonRes_HHbbggCandidate_mass", "Res_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "Res_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "jet1_mass", "jet2_mass", "jet3_mass", "jet4_mass", "jet5_mass", "jet6_mass", "Res_lead_bjet_jet_idx", "Res_sublead_bjet_jet_idx", "jet1_index", "jet2_index", "jet3_index", "jet4_index", "jet5_index", "jet6_index",
        #                   "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "jet5_pt", "jet6_pt", "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta", "jet5_eta", "jet6_eta", "jet1_phi", "jet2_phi", "jet3_phi", "jet4_phi", "jet5_phi", "jet6_phi", "lead_phi", "sublead_phi"]

        self.extra_vars = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_HHbbggCandidate_mass", "nonResReg_dijet_pt", "nonResReg_lead_bjet_pt", "nonResReg_sublead_bjet_pt", "nonResReg_lead_bjet_eta", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "nonResReg_lead_bjet_btagPNetB", "nonResReg_sublead_bjet_btagPNetB", "nonResReg_lead_bjet_btagUParTAK4B", "nonResReg_sublead_bjet_btagUParTAK4B", "weight", "pt", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi"] # "lead_genPartFlav", "sublead_genPartFlav", "weight_tot" added for sim predictions only in the dedicated functions

        self.vars_for_boosted = ['sublead_mvaID', 'fatjet3_tau2', 'fatjet3_particleNet_XbbVsQCD', 'fatjet4_subjet2_eta', 'sublead_eta', 'fatjet2_phi', 'fatjet1_mass', 'nonResReg_CosThetaStar_gg', 'fatjet4_particleNet_XbbVsQCD', 'lead_phi', 'fatjet4_pt', 'fatjet4_tau1', 'fatjet4_tau2', 'fatjet2_particleNet_XbbVsQCD', 'fatjet3_subjet1_eta', 'fatjet1_subjet1_eta', 'lead_eta', 'fatjet3_msoftdrop', 'fatjet4_mass', 'fatjet4_particleNet_massCorr', 'fatjet1_tau1', 'eta', 'fatjet2_pt', 'phi', 'fatjet1_subjet2_phi', 'fatjet3_eta', 'fatjet1_subjet2_eta', 'nonResReg_phosublead_PtOverM', 'fatjet4_subjet1_phi', 'fatjet3_subjet2_phi', 'fatjet3_subjet1_phi', 'fatjet2_tau2', 'n_jets', 'fatjet2_msoftdrop', 'fatjet2_subjet2_phi', 'fatjet3_pt', 'fatjet2_eta', 'fatjet3_tau1', 'fatjet4_eta', 'fatjet1_eta', 'fatjet3_mass', 'n_fatjets', 'fatjet1_pt', 'fatjet3_subjet2_eta', 'fatjet1_subjet1_phi', 'fatjet1_msoftdrop', 'lead_mvaID', 'fatjet4_subjet1_eta', 'nonResReg_pholead_PtOverM', 'fatjet2_tau1', 'fatjet2_mass', 'fatjet2_subjet2_eta', 'fatjet3_phi', 'n_leptons', 'fatjet1_particleNet_massCorr', 'fatjet2_subjet1_phi', 'fatjet4_subjet2_phi', 'fatjet1_tau2', 'fatjet1_phi', 'fatjet2_subjet1_eta', 'fatjet4_phi', 'fatjet1_particleNet_XbbVsQCD', 'fatjet3_particleNet_massCorr', 'fatjet4_msoftdrop', 'sublead_phi', 'fatjet2_particleNet_massCorr']
        
        # prepare process numbers for proccesses in each class
        num_process_each_class = {
            class_: 0 for class_ in self.classes
        }

        process_numbers = {}

        for sample in self.sample_to_class.keys():
            class_ = self.sample_to_class[sample]
            process_numbers[sample] = num_process_each_class[class_]
            num_process_each_class[class_] += 1

        self.num_process_each_class = num_process_each_class
        self.process_numbers = process_numbers

        self.class_idx_to_name = {i: class_ for i, class_ in enumerate(self.classes)}

        # info to save in parquet
        self.save_all_columns_sim_nominal = training_info["save_all_columns_sim_nominal"]
        self.save_all_columns_data = training_info["save_all_columns_data"]
        self.save_all_columns_sim_systematics = training_info["save_all_columns_sim_systematics"]
        

    def load_vars(self, path):
        with open(path, 'r') as f:
            vars = yaml.safe_load(f)
        return vars

    def load_and_process_sample(self, samples_path, parquet_path, samples, era, vars_to_load, preselection_func, save_all_columns=False, systematic=None, apply_xsec_weights=True):
        """
        Load parquet file(s) for a sample and apply preprocessing.
        Handles both single files and lists of files (for VH merging in 2024/2025).

        Args:
            samples_path: Base path to samples
            parquet_path: Either a string path or list of paths (for VH merging)
            samples: Sample name (e.g., 'VHtoGG_M_125')
            era: Era name (e.g., '2024')
            vars_to_load: List of variables to load from parquet
            preselection_func: Function to apply for event selection
            save_all_columns: If True, load all columns instead of vars_to_load
            systematic: If provided, replace 'nominal' with this systematic name in paths
            apply_xsec_weights: If True, apply cross-section weights (set False for data)

        Returns:
            events: Awkward array with processed events
        """
        # Handle case where parquet_path is a list (for merging processes like VH in 2024)
        if isinstance(parquet_path, list):
            print(f"INFO: Merging {len(parquet_path)} files for {samples} in {era}")
            # Process name mapping for individual VH components
            vh_component_names = {
                "WmHtoGG": "WmHtoGG_M_125",
                "WpHtoGG": "WpHtoGG_M_125",
                "ZHtoGG": "ZHtoGG_M_125"
            }
            events_list = []
            for path in parquet_path:
                # Apply systematic replacement if needed
                if systematic is not None:
                    path = path.replace("nominal", systematic)
                    # Check if file exists
                    if not os.path.exists(f"{samples_path}/{path}"):
                        print(f"WARNING: {samples} for {era} for {systematic} does not exist. Skipping.: {samples_path}/{path}")
                        continue

                if save_all_columns:
                    events_part = ak.from_parquet(f"{samples_path}/{path}")
                else:
                    events_part = ak.from_parquet(f"{samples_path}/{path}", columns=vars_to_load)

                events_part = preselection_func(events_part)
                events_part = self.add_var(events_part, era)

                # Determine which VH component this is from the path
                component_name = None
                for key in vh_component_names.keys():
                    if key in path:
                        component_name = vh_component_names[key]
                        break

                # Apply correct cross-section for each component
                if apply_xsec_weights:
                    events_part = self.get_relative_xsec_weight(events_part, component_name, era)
                events_list.append(events_part)

            # Only concatenate if we have events (some systematics might not exist)
            if len(events_list) > 0:
                events = ak.concatenate(events_list, axis=0)
                # Clean up intermediate list
                del events_list
            else:
                events = None
        else:
            # Apply systematic replacement if needed
            if systematic is not None:
                parquet_path = parquet_path.replace("nominal", systematic)
                # Check if file exists
                if not os.path.exists(f"{samples_path}/{parquet_path}"):
                    print(f"WARNING: {samples} for {era} for {systematic} does not exist. Skipping.: {samples_path}/{parquet_path}")
                    return None

            if save_all_columns:
                events = ak.from_parquet(f"{samples_path}/{parquet_path}")
            else:
                events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)

            events = preselection_func(events)
            events = self.add_var(events, era)
            if apply_xsec_weights:
                events = self.get_relative_xsec_weight(events, samples, era)

        # Convert float64 fields to float32 to save memory
        if events is not None:
            for field in events.fields:
                if hasattr(events[field], 'dtype'):
                    if events[field].dtype == np.float64:
                        events[field] = ak.values_astype(events[field], np.float32)

        return events
    
    def deltaR(self, eta1, phi1, eta2, phi2, fill_none=True):
        eta1 = ak.mask(eta1, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        phi1 = ak.mask(phi1, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        eta2 = ak.mask(eta2, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        phi2 = ak.mask(phi2, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))

        dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
        deta = eta1 - eta2
        delta_r = np.sqrt(deta**2 + dphi**2)

        if fill_none:
            return ak.fill_none(delta_r, -999.0)
        else:
            return delta_r

    def add_var(self, events, era):

        events["diphoton_PtOverM_ggjj"] = events.pt / events.nonResReg_HHbbggCandidate_mass
        events["nonResReg_dijet_PtOverM_ggjj"] = events.nonResReg_dijet_pt / events.nonResReg_HHbbggCandidate_mass

        events["nonResReg_lead_bjet_over_M_regressed"] = events.nonResReg_lead_bjet_pt / events.nonResReg_dijet_mass_DNNreg
        events["nonResReg_sublead_bjet_over_M_regressed"] = events.nonResReg_sublead_bjet_pt / events.nonResReg_dijet_mass_DNNreg

        # add deltaR between lead and sublead photon
        events["deltaR_gg"] = self.deltaR(events.lead_eta, events.lead_phi, events.sublead_eta, events.sublead_phi)

        # add b-tagging working points
        btagVariable = "btag"
        if era == "preEE":
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.961, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.961, int)
        elif era == "postEE":
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.9664, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.9664, int)
        elif era == "preBPix":
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.9659, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.9659, int)
        elif era == "postBPix":
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_lead_bjet_btagPNetB"] > 0.9688, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagPNetB"] > 0.9688, int)
        elif era == "2024" or era == "2025":
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_lead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_lead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_lead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_lead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_lead_bjet_btagUParTAK4B"] > 0.9739, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_sublead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_sublead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_sublead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_sublead_bjet_btagUParTAK4B"] > 0.9739, int)

        # add jet related mass
            
        # # Build awkward array of jets
        # jets = ak.zip({
        #     "pt": ak.concatenate([events[f"jet{i}_pt"][:, None] for i in range(1, 7)], axis=1),
        #     "eta": ak.concatenate([events[f"jet{i}_eta"][:, None] for i in range(1, 7)], axis=1),
        #     "phi": ak.concatenate([events[f"jet{i}_phi"][:, None] for i in range(1, 7)], axis=1),
        #     "mass": ak.concatenate([events[f"jet{i}_mass"][:, None] for i in range(1, 7)], axis=1),
        #     "index": ak.concatenate([events[f"jet{i}_index"][:, None] for i in range(1, 7)], axis=1),
        # }, with_name="Momentum4D")
        
        # # Mask out jets that are b-jets
        # is_not_bjet = (jets.index != events.Res_lead_bjet_jet_idx[:, None]) & \
        #               (jets.index != events.Res_sublead_bjet_jet_idx[:, None])
        # jets_clean = jets[is_not_bjet]

        # # Select up to 4 jets
        # selected_jets = jets_clean[:, :4]
        
        # # ΔR to objects
        # def min_deltaR_to(obj_eta, obj_phi):
        #     min = ak.min(self.deltaR(selected_jets.eta, selected_jets.phi, obj_eta[:, None], obj_phi[:, None], fill_none=False), axis=1)
        #     return ak.fill_none(min, -999.0)

        # events["min_deltaR_jet_b1"] = min_deltaR_to(events.Res_lead_bjet_eta, events.Res_lead_bjet_phi)
        # events["min_deltaR_jet_b2"] = min_deltaR_to(events.Res_sublead_bjet_eta, events.Res_sublead_bjet_phi)
        # events["min_deltaR_jet_g1"] = min_deltaR_to(events.lead_eta, events.lead_phi)
        # events["min_deltaR_jet_g2"] = min_deltaR_to(events.sublead_eta, events.sublead_phi)

        # # deltaR betwreen the jets anf photons, bjets
        # events["deltaR_g1_j1"] = self.deltaR(events.lead_eta, events.lead_phi, selected_jets.eta[:, 0], selected_jets.phi[:, 0])
        # events["deltaR_g1_j2"] = self.deltaR(events.lead_eta, events.lead_phi, selected_jets.eta[:, 1], selected_jets.phi[:, 1])
        # events["deltaR_g1_j3"] = self.deltaR(events.lead_eta, events.lead_phi, selected_jets.eta[:, 2], selected_jets.phi[:, 2])
        # events["deltaR_g1_j4"] = self.deltaR(events.lead_eta, events.lead_phi, selected_jets.eta[:, 3], selected_jets.phi[:, 3])
        # events["deltaR_g2_j1"] = self.deltaR(events.sublead_eta, events.sublead_phi, selected_jets.eta[:, 0], selected_jets.phi[:, 0])
        # events["deltaR_g2_j2"] = self.deltaR(events.sublead_eta, events.sublead_phi, selected_jets.eta[:, 1], selected_jets.phi[:, 1])
        # events["deltaR_g2_j3"] = self.deltaR(events.sublead_eta, events.sublead_phi, selected_jets.eta[:, 2], selected_jets.phi[:, 2])
        # events["deltaR_g2_j4"] = self.deltaR(events.sublead_eta, events.sublead_phi, selected_jets.eta[:, 3], selected_jets.phi[:, 3])
        # events["deltaR_b1_j1"] = self.deltaR(events.Res_lead_bjet_eta, events.Res_lead_bjet_phi, selected_jets.eta[:, 0], selected_jets.phi[:, 0])
        # events["deltaR_b1_j2"] = self.deltaR(events.Res_lead_bjet_eta, events.Res_lead_bjet_phi, selected_jets.eta[:, 1], selected_jets.phi[:, 1])
        # events["deltaR_b1_j3"] = self.deltaR(events.Res_lead_bjet_eta, events.Res_lead_bjet_phi, selected_jets.eta[:, 2], selected_jets.phi[:, 2])
        # events["deltaR_b1_j4"] = self.deltaR(events.Res_lead_bjet_eta, events.Res_lead_bjet_phi, selected_jets.eta[:, 3], selected_jets.phi[:, 3])
        # events["deltaR_b2_j1"] = self.deltaR(events.Res_sublead_bjet_eta, events.Res_sublead_bjet_phi, selected_jets.eta[:, 0], selected_jets.phi[:, 0])
        # events["deltaR_b2_j2"] = self.deltaR(events.Res_sublead_bjet_eta, events.Res_sublead_bjet_phi, selected_jets.eta[:, 1], selected_jets.phi[:, 1])
        # events["deltaR_b2_j3"] = self.deltaR(events.Res_sublead_bjet_eta, events.Res_sublead_bjet_phi, selected_jets.eta[:, 2], selected_jets.phi[:, 2])
        # events["deltaR_b2_j4"] = self.deltaR(events.Res_sublead_bjet_eta, events.Res_sublead_bjet_phi, selected_jets.eta[:, 3], selected_jets.phi[:, 3])

        # # add jet pt, eta, phi
        # events["j1_pt"] = selected_jets.pt[:, 0]
        # events["j2_pt"] = selected_jets.pt[:, 1]
        # events["j3_pt"] = selected_jets.pt[:, 2]
        # events["j4_pt"] = selected_jets.pt[:, 3]
        # events["j1_eta"] = selected_jets.eta[:, 0]
        # events["j2_eta"] = selected_jets.eta[:, 1]
        # events["j3_eta"] = selected_jets.eta[:, 2]
        # events["j4_eta"] = selected_jets.eta[:, 3]
        # events["j1_phi"] = selected_jets.phi[:, 0]
        # events["j2_phi"] = selected_jets.phi[:, 1]
        # events["j3_phi"] = selected_jets.phi[:, 2]
        # events["j4_phi"] = selected_jets.phi[:, 3]
       


        # # Build Lorentz vectors from selected_jets
        # jets_vec = selected_jets

        # # Pair indices for 4 jets
        # pair_indices = [(0, 1), (0, 2), (0, 3),
        #                 (1, 2), (1, 3),
        #                 (2, 3)]

        # pair_names = ["j1_j2", "j1_j3", "j1_j4", "j2_j3", "j2_j4", "j3_j4"]

        #for (i, j), name in zip(pair_indices, pair_names):
        #    # Mask if either jet is invalid (pt == -999)
        #    valid = (selected_jets.pt[:, i] != -999) & (selected_jets.pt[:, j] != -999)

        #    # Sum vectors and get invariant mass
        #    m_pair = (jets_vec[:, i] + jets_vec[:, j]).mass

        #    # Set to -999 if invalid
        #    events[f"mass_{name}"] = ak.where(valid, m_pair, -999.0)

        #    # Compute ΔR for the pair using your self.deltaR
        #    delta_r = self.deltaR(
        #        selected_jets.eta[:, i], selected_jets.phi[:, i],
        #        selected_jets.eta[:, j], selected_jets.phi[:, j]
        #    )
        #    events[f"deltaR_{name}"] = ak.where(valid, delta_r, -999.0)

        return events


    def get_relative_xsec_weight(self, events, sample_type, era):

        dict_xsec = {
            # Accounting for the template fit factor which is not included in the parquet weight in 2024 and 2025
            "GGJets": (88.75e3 * 1.59) if (era=="2024" or era=="2025") else 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.02391e3,  # cross sectio of TTGG 0.01696, copilot: 0.502
            "ttHtoGG_M_125": 0.5700e3 * 0.00227,  # cross sectio of ttH * BR(HToGG)
            "BBHto2G_M_125": 0.4385e3 * 0.00227,  # cross sectio of BBH * BR(HToGG)
            "GluGluHToGG_M_125": 52.23e3 * 0.00227,  # cross sectio of GluGluHToGG * BR(HToGG)
            "VBFHToGG_M_125": 4.078e3 * 0.00227,
            "VHtoGG_M_125": 2.4009e3 * 0.00227,
            "WpHtoGG_M_125": 0.880114e3 * 0.00227,
            "WmHtoGG_M_125": 0.562032e3 * 0.00227,
            "ZHtoGG_M_125": 0.9361e3 * 0.00227,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.034e3 * 0.00227 * 0.582 * 2,#0.02964e3 * 0.00227 * 0.582 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HToGG) * 2 for two combination ### have to recheck if this is correct. 
            "VBFHH_CV_1p000_C2V_1p000_C3_1p000": 0.00173e3 * 0.00227 * 0.582 * 2,  # cross sectio of VBFToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct.
            "DDQCDGJET": 1.0,
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.09965e3 * 0.00227 * 0.582 * 2,
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.07575e3 * 0.00227 * 0.582 * 2,
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.01477e3 * 0.00227 * 0.582 * 2,
            "TTG_10_100": 4.334e3,
            "TTG_100_200": 0.44e3,
            "TTG_200": 0.12e3,
            "TT": 730e3,
        }
        luminosities = {
        "preEE": 7.98,  # Integrated luminosity in fb^-1
        "postEE": 26.67,  # Integrated luminosity in fb^-1
        "preBPix": 17.794,  # Integrated luminosity in fb^-1
        "postBPix": 9.451,  # Integrated luminosity in fb^-1
        "2024": 109.08,  # Integrated luminosity in fb^-1
        "2025": 115.65  # Integrated luminosity in fb^-1
        }

        lumi = luminosities[era]
        if sample_type == "DDQCDGJET":
            lumi = 1.0

        events["rel_xsec_weight"] = (events.weight) * dict_xsec[sample_type] * lumi
        events["weight_tot"] = (events.weight) * dict_xsec[sample_type] * lumi

        return events

    def get_weights_for_training(self, y_train, rel_w_train, proc_num_train):

        true_class_weights = ak.zeros_like(rel_w_train)
        class_weights_for_training_abs = ak.zeros_like(rel_w_train)
        class_weights_only_positive = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                cls_bool = (y_train[:, i] == 1)

                true_class_weights_ = ak.zeros_like(rel_w_train)
                class_weights_for_training_abs_ = ak.zeros_like(rel_w_train)
                class_weights_only_positive_ = ak.zeros_like(rel_w_train)

                for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
                    rel_xsec_weight_for_class = rel_w_train * cls_bool * (proc_num_train == proc)
                    true_class_weights_ = true_class_weights_ + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

                    abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool * (proc_num_train == proc)
                    class_weights_for_training_abs_ = class_weights_for_training_abs_ + (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

                    only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0) * (proc_num_train == proc)
                    class_weights_only_positive_ = class_weights_only_positive_ + (only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class))

                # normalize the weights for this class to be one
                true_class_weights = true_class_weights + (true_class_weights_ / np.sum(true_class_weights_))
                class_weights_for_training_abs = class_weights_for_training_abs + (class_weights_for_training_abs_ / np.sum(class_weights_for_training_abs_))
                class_weights_only_positive = class_weights_only_positive + (class_weights_only_positive_ / np.sum(class_weights_only_positive_))
                
            else:
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

            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                print("\n")
                for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
                    print(f"(number of events: sum of class_weights_for_training_abs) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(class_weights_for_training_abs * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                    print(f"(number of events: sum of class_weights_only_positive) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(class_weights_only_positive * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                    print(f"(number of events: sum of true_class_weights) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(true_class_weights * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                print("\n")

        return true_class_weights, class_weights_for_training_abs, class_weights_only_positive

    def get_weights_for_val_test(self, y_val, rel_w_val, proc_num_val):

        class_weights_for_val = ak.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            
            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                cls_bool = (y_val[:, i] == 1)
                class_weights_for_val_ = ak.zeros_like(rel_w_val)

                for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
                    rel_xsec_weight_for_class = rel_w_val * cls_bool * (proc_num_val == proc)
                    class_weights_for_val_ = class_weights_for_val_ + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

                # normalize the weights for this class to be one
                class_weights_for_val = class_weights_for_val + (class_weights_for_val_ / np.sum(class_weights_for_val_))

            else:
                cls_bool = (y_val[:, i] == 1)
                rel_xsec_weight_for_class = rel_w_val * cls_bool
                class_weights_for_val = class_weights_for_val + (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                print("\n")
                for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
                    print(f"(number of events: sum of class_weights_for_val) for class number {i+1} and process number {proc} = ({sum(y_val[:, i] * (proc_num_val == proc))}: {sum(class_weights_for_val * (y_val[:, i] == 1) * (proc_num_val == proc))})")
                print("\n")

        return class_weights_for_val

    def train_test_split(self, X, Y, relative_weights, proc_num, train_ratio=0.7, val_ratio=0.3):

        from sklearn.model_selection import train_test_split

        X_train, X_test_val, y_train, y_test_val, rel_w_train, rel_w_test_val, proc_num_train, proc_num_val = train_test_split(X, Y, relative_weights, proc_num, train_size=train_ratio, shuffle=True, random_state=self.random_seed)
        if (train_ratio + val_ratio) == 1.0:
            X_val, y_val, rel_w_val = X_test_val, y_test_val, rel_w_test_val
            X_test, y_test, rel_w_test, proc_num_test = None, None, None, None
        else:
            X_val, X_test, y_val, y_test, rel_w_val, rel_w_test, proc_num_val, proc_num_test = train_test_split(X_test_val, y_test_val, rel_w_test_val, proc_num_val, train_size=0.5, shuffle=True, random_state=self.random_seed)

        return X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test, proc_num_train, proc_num_val, proc_num_test


    def standardize(self, X, mean, std):
        return (X - mean) / std

    def min_max_scale(self, X, min, max):
        return (X - min) / (max - min)

    def corr_with_mgg_mjj(self, events, vars_for_training, out_path):

        "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg"

        corr_matrix = np.zeros([len(vars_for_training), 4])
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

            mask = ((events[var] > -998.0) & (events.nonResReg_dijet_mass > -998.0))
            nonResReg_dijet_mass = events.nonResReg_dijet_mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 2] = np.corrcoef(nonResReg_dijet_mass, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.nonResReg_dijet_mass_DNNreg > -998.0))
            nonResReg_dijet_mass_DNNreg = events.nonResReg_dijet_mass_DNNreg[mask]
            var_values = events[var][mask]
            corr_matrix[i, 3] = np.corrcoef(nonResReg_dijet_mass_DNNreg, var_values)[0, 1]

        # plot the correlation matrix
        plt.figure(figsize=(18, len(vars_for_training)))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        # annotate the values
        for i in range(len(vars_for_training)):
            for j in range(4):
                # format the value to 2 decimal places
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='b')

        plt.xticks([0, 1, 2, 3], ['mass', 'nonRes_dijet_mass', 'nonResReg_dijet_mass', 'nonResReg_dijet_mass_DNNreg'], rotation=90)
        plt.yticks(range(len(vars_for_training)), vars_for_training)
        plt.colorbar()
        plt.savefig(f'{out_path}', dpi=300, )
        plt.clf()

    
    def preselection(self, events):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.nonResReg_dijet_mass_DNNreg > 70) & (events.nonResReg_dijet_mass_DNNreg < 190))

        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

        return events
    
    def preselection_for_pred(self, events):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))

        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        events = events[mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

        return events
    
    def plot_variables(self, comb_inputs, vars_for_training, plot_path):
        """
        Plot variables directly from awkward array without pandas conversion.

        Args:
            comb_inputs: Awkward array containing all combined inputs
            vars_for_training: List of variable names to plot
            plot_path: Path to save plots
        """
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
            plt.figure(figsize=(10, 9))
            sample_num = 0
            data_to_plot_dict = {}
            range_list = []

            for sample in self.sample_to_class.keys():
                # Filter events for this sample using awkward operations
                sample_mask = comb_inputs["sample_type"] == sample
                sample_events_var = comb_inputs[var][sample_mask]

                # Apply value mask
                mask = (sample_events_var > -998.0)
                data_to_plot = sample_events_var[mask]
                data_to_plot_dict[sample] = data_to_plot

                if len(data_to_plot) > 0:
                    data_min = float(ak.min(data_to_plot))
                    data_max = float(ak.max(data_to_plot))

                    if range_list == []:
                        range_list = [data_min, data_max]
                    else:
                        range_list[0] = min(range_list[0], data_min)
                        range_list[1] = max(range_list[1], data_max)

            for sample in self.sample_to_class.keys():
                #if sample in ["DDQCDGJET", ]:
                #    continue
                data_to_plot = data_to_plot_dict[sample]

                if len(data_to_plot) > 0:
                    # use mplhep to plot the histogram
                    hist_ = np.histogram(ak.to_numpy(data_to_plot), bins=30, range=range_list, density=True)
                    plt.style.use(hep.style.CMS)
                    hep.histplot(hist_, histtype='step', label=sample, color=color_list[sample_num], linestyle='solid', linewidth=1.5)
                sample_num += 1

            plt.xlabel(f"{var}")
            plt.ylabel("a.u.")
            plt.legend(ncols=2, fontsize=13, loc='upper right')
            plt.yscale('log')
            #plt.ylim(0.001, plt.ylim()[1] * 6)
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}_log.png")

            plt.yscale('linear')
            yrange = plt.ylim()
            plt.ylim(0, yrange[1] * 1.3)
            plt.tight_layout()
            plt.savefig(f"{plot_path}/{var}.png")
            plt.clf()
            plt.close()  # Explicitly close figure to free memory
            


    def prep_inputs_for_training(self):

        fill_nan = self.fill_nan

        out_path = self.outpath
        os.makedirs(out_path, exist_ok=True)

        comb_inputs = []

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        vars_for_training = vars_config["vars"] 
        # vars_for_log = vars_config["vars_for_log_transform"]

        vars_to_load = vars_for_training + self.extra_vars

        for era in self.training_info["samples_info"]["eras"]:
            for samples in self.sample_to_class.keys():
                print(samples)

                samples_path = self.training_info["samples_info"]["samples_path"]
                parquet_path = self.training_info["samples_info"][era][samples]

                # Load and process sample
                events = self.load_and_process_sample(
                    samples_path=samples_path,
                    parquet_path=parquet_path,
                    samples=samples,
                    era=era,
                    vars_to_load=vars_to_load,
                    preselection_func=self.preselection,
                    save_all_columns=False
                )

                print(f"INFO: Number of MC events in {samples} after selection for {era}: {len(events)}")
                print(f"INFO: Sum of weight_tot in {samples} after selection for {era}: {sum(events.weight_tot)}")

                # add the bools for each class
                for cls in self.classes:  # first intialize everything to zero
                    events[cls] = ak.zeros_like(events.eta)

                events[self.sample_to_class[samples]] = ak.ones_like(events.pt) # one-hot encoded
                comb_inputs.append(events)
                events["sample_type"] = samples

                # add process number which is specific for each class
                events["process_number"] = self.process_numbers[samples]

                # plot_correlation_matrix
                os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
                corr_out_path = f"{out_path}/correlation_matrix/{samples}_{era}.pdf"
                self.corr_with_mgg_mjj(events, vars_for_training, corr_out_path)

        print("INFO: Combining all the samples")
        comb_inputs = ak.concatenate(comb_inputs, axis=0)

        plot_path = f"{out_path}/var_plots/"
        os.makedirs(plot_path, exist_ok=True)
        self.plot_variables(comb_inputs, vars_for_training, plot_path)

        for cls in self.classes:
            print("\n", f"INFO: Number of events in {cls}: {ak.sum(comb_inputs[cls])}")

        # Build X array directly from awkward array
        print("INFO: Converting training variables to numpy arrays")
        X_list = []
        for var in vars_for_training:
            var_data = ak.to_numpy(ak.fill_none(comb_inputs[var], -999.0))
            X_list.append(var_data)
        X = np.column_stack(X_list).astype(np.float32)  # Use float32

        # Extract Y (class labels) directly
        print("INFO: Converting class labels to numpy arrays")
        Y_list = []
        for cls in self.classes:
            # Fill None with 0 for class labels (0 = not in this class, prevents MaskedArray)
            cls_data = ak.to_numpy(ak.fill_none(comb_inputs[cls], 0))
            Y_list.append(cls_data)
        Y = np.column_stack(Y_list).astype(np.int8)  # Use int8 for labels

        # Extract weights and process numbers directly
        # Fill None with NaN for weights
        relative_weights = ak.to_numpy(ak.fill_none(comb_inputs["rel_xsec_weight"], np.nan)).astype(np.float32)
        # Fill None with -1 for process_number (int8 cannot represent NaN)
        process_number = ak.to_numpy(ak.fill_none(comb_inputs["process_number"], -1)).astype(np.int8)

        # Clean up temporary lists and awkward array
        del X_list, Y_list, comb_inputs
        import gc
        gc.collect()

        # mask -999.0 to nan
        mask = (X < -998.0)
        X[mask] = np.nan

        X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test, proc_num_train, proc_num_val, proc_num_test = self.train_test_split(X, Y, relative_weights, process_number)

        # Clean up original arrays after split
        del X, Y, relative_weights, process_number, mask
        gc.collect()

        # get mean according to training data set
        mean = np.nanmean(X_train, axis=0)
        std = np.nanstd(X_train, axis=0)

        # transform all data set
        X_train = self.standardize(X_train, mean, std)
        X_val = self.standardize(X_val, mean, std)

        # replace NaN with fill_nan value
        X_train = np.nan_to_num(X_train, nan=fill_nan)
        X_val = np.nan_to_num(X_val, nan=fill_nan)

        true_class_weights, class_weights_for_training_abs, class_weights_only_positive = self.get_weights_for_training(y_train, rel_w_train, proc_num_train)
        class_weights_for_val = self.get_weights_for_val_test(y_val, rel_w_val, proc_num_val)

        if X_test is not None:
            X_test = self.standardize(X_test, mean, std)
            X_test = np.nan_to_num(X_test, nan=fill_nan)
            class_weights_for_test = self.get_weights_for_val_test(y_test, rel_w_test, proc_num_test)
        
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
    
    def prep_inputs_for_prediction_sim(self):

        fill_nan = self.fill_nan
        training_info = self.training_info
        inputs_path = self.outpath
        out_path = f"{inputs_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars

        # vars_for_log = vars_config["vars_for_log_transform"]

        vars_to_load = vars_for_training + self.extra_vars + self.vars_for_boosted + ["lead_genPartFlav", "sublead_genPartFlav", "weight_tot"]

        samples_path = training_info["samples_info"]["samples_path"]

        # Load mean/std once for all samples
        scale_file = f"{inputs_path}/mean_std_dict.pkl"
        with open(scale_file, 'rb') as f:
            mean_std_dict = pickle.load(f)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std_dev"]

        for era in training_info["samples_info"]["eras"]:
            for samples in training_info["samples_info"][era].keys():

                parquet_path = training_info["samples_info"][era][samples]

                # Load and process sample
                events = self.load_and_process_sample(
                    samples_path=samples_path,
                    parquet_path=parquet_path,
                    samples=samples,
                    era=era,
                    vars_to_load=vars_to_load,
                    preselection_func=self.preselection_for_pred,
                    save_all_columns=self.save_all_columns_sim_nominal
                )

                print(f"INFO: Number of events in {samples} for {era}: {len(events)}")

                # Build X array directly from awkward array
                X_list = []
                for var in vars_for_training:
                    var_data = ak.to_numpy(ak.fill_none(events[var], -999.0))
                    X_list.append(var_data)
                X = np.column_stack(X_list).astype(np.float32)  # Use float32

                # Fill None with NaN for weights
                relative_weights = ak.to_numpy(ak.fill_none(events["rel_xsec_weight"], np.nan)).astype(np.float32)

                # Clean up temporary list
                del X_list

                # mask -999.0 to nan
                mask = (X < -998.0)
                X[mask] = np.nan

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

                # Explicitly free memory
                del X, relative_weights, events, mask

        # save the training mean ans std_dev. This will be used for standardizing data
        # (save only once at the end)
        mean_std_dict = {
            "mean": mean,
            "std_dev": std
        }
        with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
            pickle.dump(mean_std_dict, f)

        return 0

    def prep_inputs_for_prediction_sim_sys(self):

        fill_nan = self.fill_nan
        training_info = self.training_info
        inputs_path = self.outpath
        out_path = f"{inputs_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars

        # vars_for_log = vars_config["vars_for_log_transform"]

        vars_to_load = vars_for_training + self.extra_vars + self.vars_for_boosted + ["lead_genPartFlav", "sublead_genPartFlav", "weight_tot"]

        samples_path = training_info["samples_info"]["samples_path"]

        # Load mean/std once for all samples
        scale_file = f"{inputs_path}/mean_std_dict.pkl"
        with open(scale_file, 'rb') as f:
            mean_std_dict = pickle.load(f)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std_dev"]

        for era in training_info["samples_info"]["eras"]:
            for samples in training_info["samples_info"][era].keys():
                for sys in training_info["systematics"]:

                    if samples in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
                        continue

                    parquet_path = training_info["samples_info"][era][samples]

                    # Load and process sample with systematic variation
                    events = self.load_and_process_sample(
                        samples_path=samples_path,
                        parquet_path=parquet_path,
                        samples=samples,
                        era=era,
                        vars_to_load=vars_to_load,
                        preselection_func=self.preselection_for_pred,
                        save_all_columns=self.save_all_columns_sim_systematics,
                        systematic=sys
                    )

                    # Skip if events is None (file didn't exist)
                    if events is None:
                        continue

                    print(f"INFO: Number of events in {samples} for {era} for {sys}: {len(events)}")

                    # Build X array directly from awkward array
                    X_list = []
                    for var in vars_for_training:
                        var_data = ak.to_numpy(ak.fill_none(events[var], -999.0))
                        X_list.append(var_data)
                    X = np.column_stack(X_list).astype(np.float32)  # Use float32

                    # Fill None with NaN for weights
                    relative_weights = ak.to_numpy(ak.fill_none(events["rel_xsec_weight"], np.nan)).astype(np.float32)

                    # Clean up temporary list
                    del X_list

                    # mask -999.0 to nan
                    mask = (X < -998.0)
                    X[mask] = np.nan

                    # transform all data set
                    X = self.standardize(X, mean, std)

                    # replace NaN with fill_nan value
                    X = np.nan_to_num(X, nan=fill_nan)

                    # save all the numpy arrays
                    full_path_to_save = f"{out_path}/{era}/{samples}/{sys}/"
                    os.makedirs(full_path_to_save, exist_ok=True)

                    np.save(f"{full_path_to_save}/X", X)
                    np.save(f"{full_path_to_save}/rel_w", relative_weights)

                    # also save the event
                    ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                    # Explicitly free memory
                    del X, relative_weights, events, mask

        # save the training mean ans std_dev. This will be used for standardizing data
        # (save only once at the end)
        mean_std_dict = {
            "mean": mean,
            "std_dev": std
        }
        with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
            pickle.dump(mean_std_dict, f)

        return 0
    
    def prep_inputs_for_prediction_data(self):

        fill_nan = self.fill_nan
        training_info = self.training_info
        inputs_path = self.outpath
        out_path = f"{inputs_path}/individual_samples_data/"
        os.makedirs(out_path, exist_ok=True)

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]
        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars

        # vars_for_log = vars_config["vars_for_log_transform"]

        vars_to_load = vars_for_training + self.extra_vars + self.vars_for_boosted

        samples_path = training_info["samples_info"]["samples_path"]
        datas = training_info["samples_info"]["data"]

        # Load mean/std once for all data samples
        scale_file = f"{inputs_path}/mean_std_dict.pkl"
        with open(scale_file, 'rb') as f:
            mean_std_dict = pickle.load(f)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std_dev"]

        sample_to_era = {"2022_EraE": "postEE",
                         "2022_EraF": "postEE",
                         "2022_EraG": "postEE",
                         "2022_EraC": "preEE",
                         "2022_EraD": "preEE",
                         "2023_EraCv1to3": "preBPix",
                         "2023_EraCv4": "preBPix",
                         "2023_EraC": "preBPix",
                         "2023_EraD": "postBPix",
                         "2024": "2024",
                         "2025": "2025"}

        for data in datas:
            # Load and process sample with apply_xsec_weights=False for data
            events = self.load_and_process_sample(
                samples_path=samples_path,
                parquet_path=datas[data],
                samples=data,
                era=sample_to_era[data],
                vars_to_load=vars_to_load,
                preselection_func=self.preselection_for_pred,
                save_all_columns=self.save_all_columns_data,
                apply_xsec_weights=False  # Data doesn't need MC cross-section weights
            )

            # Build X array directly from awkward array
            X_list = []
            for var in vars_for_training:
                var_data = ak.to_numpy(ak.fill_none(events[var], -999.0))
                X_list.append(var_data)
            X = np.column_stack(X_list).astype(np.float32)  # Use float32

            # Clean up temporary list
            del X_list

            # mask -999.0 to nan
            mask = (X < -998.0)
            X[mask] = np.nan

            # transform all data set
            X = self.standardize(X, mean, std)
            X = np.nan_to_num(X, nan=fill_nan)

            # save all the numpy arrays
            print(f"INFO: saving inputs for {data}")
            full_path_to_save = f"{out_path}/{data}/"
            os.makedirs(full_path_to_save, exist_ok=True)

            np.save(f"{full_path_to_save}/X", X)

            ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

            # Explicitly free memory
            del X, events, mask

        # save the training mean ans std_dev. This will be used for standardizing data
        # (save only once at the end)
        mean_std_dict = {
            "mean": mean,
            "std_dev": std
        }
        with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
            pickle.dump(mean_std_dict, f)

        return
