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
from vector import register_awkward
register_awkward()

# Compatibility for pyarrow>=21 where PyExtensionType was removed.
# Older awkward versions still reference pa.lib.PyExtensionType.
if not hasattr(pa.lib, "PyExtensionType") and hasattr(pa.lib, "ExtensionType"):
    pa.lib.PyExtensionType = pa.lib.ExtensionType

class PrepareInputs:
    def __init__(
        self,
        input_var_json: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        outpath: Optional[Dict[str, Any]] = None,
        predict_parquet_info: Optional[Dict[str, Any]] = None,
        mhh_var: Optional[str] = None,
        mhh_range: Optional[List[float]] = None,
        max_input_files: Optional[int] = None,
        max_rows_per_file: Optional[int] = None,
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
            self.class_weight_scale = {class_name: float(self.training_info.get("class_weight_scale", {}).get(class_name, 1.0)) for class_name in self.classes}
            self.sample_weight_scale = {sample_name: float(self.training_info.get("sample_weight_scale", {}).get(sample_name, 1.0)) for sample_name in self.sample_to_class.keys()}
            self.write_chunk = self.training_info["write_chunk"]
        self.fill_nan = -9
        self.max_input_files = max_input_files
        self.max_rows_per_file = max_rows_per_file
        self._files_processed = 0


        self.extra_vars_train = ["weight", "mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_vbfpair_dijet_mass", "nonResReg_vbfpair_HHbbggCandidate_mass", "nonResReg_vbfpair_M_X", "nonResReg_vbfpair_dijet_pt", "nonResReg_vbfpair_lead_bjet_pt", "nonResReg_vbfpair_sublead_bjet_pt", "pt", "nonResReg_vbfpair_lead_bjet_btagPNetB", "nonResReg_vbfpair_sublead_bjet_btagPNetB",  "nonResReg_vbfpair_lead_bjet_btagUParTAK4B", "nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"]

        self.extra_vars_out = ["lead_genPartFlav", "sublead_genPartFlav","n_electrons", "n_muons", "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "jet5_pt", "jet6_pt", "jet7_pt", "jet8_pt", "jet9_pt", "jet10_pt", "nBTight", "run", "event", "lumi"] #for ttH category: njets already included as a training var

        fatjet_props = ["msoftdrop", "pt", "eta", "phi", "tau1", "tau2", "tau3", "subjet1_eta", "subjet1_phi", "subjet2_eta", "subjet2_phi", "particleNet_XbbVsQCD", "globalParT3_Xbb", "globalParT3_QCD", "mass_raw", "globalParT3_massCorrX2p"]
        vars_boosted = [f"fatjet{i}_{prop}" for i in range(1, 5) for prop in fatjet_props]
        vars_boosted += ["lead_eta", "phi", "n_fatjets"]

        self.extra_vars_out = self.extra_vars_out + vars_boosted

        vars_VBFHH_MVA = ["nonResReg_vbfpair_pholead_PtOverM", "nonResReg_vbfpair_phosublead_PtOverM", "lead_mvaID", "sublead_mvaID", "nonResReg_vbfpair_FirstJet_PtOverM", "nonResReg_vbfpair_SecondJet_PtOverM", "nonResReg_vbfpair_lead_bjet_btagPNetB", "nonResReg_vbfpair_sublead_bjet_btagPNetB", "nonResReg_vbfpair_DeltaR_jg_min", "nonResReg_vbfpair_CosThetaStar_CS", "nonResReg_vbfpair_CosThetaStar_gg", "nonResReg_vbfpair_CosThetaStar_jj", "nonResReg_vbfpair_VBF_first_jet_btagPNetQvG", "nonResReg_vbfpair_VBF_second_jet_btagPNetQvG","nonResReg_vbfpair_VBF_jet_eta_prod", "nonResReg_vbfpair_VBF_jet_eta_diff", "nonResReg_vbfpair_VBF_DeltaR_jb_min", "nonResReg_vbfpair_VBF_DeltaR_jg_min", "nonResReg_vbfpair_VBF_Cgg", "nonResReg_vbfpair_VBF_Cbb", "nonResReg_vbfpair_VBF_first_jet_PtOverM", "nonResReg_vbfpair_VBF_second_jet_PtOverM", "nonResReg_vbfpair_VBF_dijet_mass", "nonResReg_vbfpair_VBF_dijet_vbfpair_Score_jj", "nonResReg_vbfpair_HHbbggCandidate_pt"]
        vars_VBFHH_MVA += ["nonResReg_vbfpair_dijet_mass"]

        self.extra_vars_out = self.extra_vars_out + vars_VBFHH_MVA

        # Each configured sample type is one process. Use globally unique process
        # numbers so sample-level diagnostics and weights are never class-ambiguous.
        process_numbers = {
            sample: proc_num
            for proc_num, sample in enumerate(self.sample_to_class.keys())
        }
        samples_in_class = {class_: [] for class_ in self.classes}
        for sample, class_ in self.sample_to_class.items():
            samples_in_class[class_].append(process_numbers[sample])

        self.samples_in_class = samples_in_class
        self.num_process_each_class = {
            class_: len(samples_in_class[class_])
            for class_ in self.classes
        }
        self.process_numbers = process_numbers
        self.sample_weight_scale_by_process = np.ones(len(process_numbers), dtype=np.float32)
        for sample, proc_num in process_numbers.items():
            self.sample_weight_scale_by_process[proc_num] = self.sample_weight_scale[sample]

        self.class_idx_to_name = {i: class_ for i, class_ in enumerate(self.classes)}

        # info to save in parquet
        self.save_all_columns_sim_nominal = training_info["save_all_columns_sim_nominal"]
        self.save_all_columns_data = training_info["save_all_columns_data"]
        self.save_all_columns_sim_systematics = training_info["save_all_columns_sim_systematics"]

        # mHH binning options (variable name and [min,max])
        self.mhh_var = mhh_var
        # store as tuple (min, max) where max can be math.inf
        if mhh_range is not None:
            import math as _math
            lo = float(mhh_range[0])
            hi = float(mhh_range[1]) if len(mhh_range) > 1 else _math.inf
            if hi == float('inf'):
                hi = _math.inf
            self.mhh_range = (lo, hi)
        else:
            self.mhh_range = None

    def _apply_input_caps(self, events):
        if self.max_rows_per_file is None:
            return events

        try:
            max_rows = int(self.max_rows_per_file)
        except (TypeError, ValueError):
            return events

        if max_rows <= 0:
            return events[:0]

        return events[:max_rows]

    def _can_process_more_input_files(self) -> bool:
        if self.max_input_files is None:
            return True
        if int(self.max_input_files) <= 0:
            return False
        return self._files_processed < self.max_input_files

    def _record_input_file(self) -> bool:
        if self.max_input_files is None:
            return False
        self._files_processed += 1
        return self._files_processed >= self.max_input_files

    def _reset_input_file_counter(self) -> None:
        self._files_processed = 0
        

    def load_vars(self, path):
        with open(path, 'r') as f:
            vars = yaml.safe_load(f)
        return vars
    
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

        # events["diphoton_PtOverM_ggjj"] = events.pt / events.nonResReg_HHbbggCandidate_mass
        # events["nonResReg_dijet_PtOverM_ggjj"] = events.nonResReg_dijet_pt / events.nonResReg_HHbbggCandidate_mass

        events["diphoton_PtOverM_X"] = events.pt / events.nonResReg_vbfpair_M_X
        # events["nonResReg_dijet_PtOverM_X"] = events.nonResReg_dijet_pt / events.nonResReg_vbfpair_M_X

        events["nonResReg_lead_bjet_over_M_regressed"] = events.nonResReg_vbfpair_lead_bjet_pt / events.nonResReg_vbfpair_dijet_mass
        events["nonResReg_sublead_bjet_over_M_regressed"] = events.nonResReg_vbfpair_sublead_bjet_pt / events.nonResReg_vbfpair_dijet_mass

        # add deltaR between lead and sublead photon
        # events["deltaR_gg"] = self.deltaR(events.lead_eta, events.lead_phi, events.sublead_eta, events.sublead_phi)

        btagVariable = "btag"
        # Use PNetB for NanoAODv12/v13 
        if era == "preEE":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.961, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.961, int)
        elif era == "postEE":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9664, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9664, int)
        elif era == "preBPix":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9659, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9659, int)
        elif era == "postBPix":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9688, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9688, int)
        # Use UParT for NanoAODv15
        elif era == "2024" or era == "2025":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9739, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9739, int)
        elif era == "2016preVFP":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0387, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1847, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5467, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6777, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9218, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0387, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1847, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5467, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6777, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9218, int)
        elif era == "2016postVFP":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0400, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1898, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5538, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6872, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9353, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0400, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1898, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5538, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6872, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9353, int)
        elif era == "2017":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0331, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1776, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5755, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.7274, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9666, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0331, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1776, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5755, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.7274, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9666, int)
        elif era == "2018":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0308, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1610, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5405, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6992, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9655, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0308, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1610, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5405, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6992, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9655, int)
        else:
            raise ValueError(f"Era '{era}' not recognized for b-tagging WP assignment")

        if ("2016" in era) | ("VFP" in era):
            events["year"] = 0
        elif era == "2017":
            events["year"] = 1
        elif era == "2018":
            events["year"] = 2
        elif (era == "preEE") | ("2022" in era):
            events["year"] = 3
        elif (era == "postEE") | ("2022" in era):
            events["year"] = 3
        elif (era == "preBPix") | ("2023" in era):
            events["year"] = 4
        elif (era == "postBPix") | ("2023" in era):
            events["year"] = 4
        elif era == "2024":
            events["year"] = 5
        elif era == "2025":
            events["year"] = 6

        return events

    def _apply_mhh_filter(self, events):
        """Apply mHH variable range filter to events if configured.

        Keeps events where mhh_var is > -998 and within [lo, hi) (hi may be inf).
        If the configured variable is missing, raises a ValueError.
        """
        if (self.mhh_var is None) or (self.mhh_range is None):
            return events

        var = self.mhh_var
        lo, hi = self.mhh_range

        # ensure variable exists in the events
        try:
            vals = events[var]
        except Exception:
            raise ValueError(f"mHH binning variable '{var}' not found in events")

        # build mask excluding sentinel values
        if np.isfinite(hi):
            mask = (vals > -998.0) & (vals >= lo) & (vals < hi)
        else:
            mask = (vals > -998.0) & (vals >= lo)

        events = events[mask]

        return events


    def get_relative_xsec_weight(self, events, sample_type, era):
        # Using mH = 125.4 GeV
        # for kl samples and H BRs: https://gitlab.cern.ch/hh/recommendations/-/blob/master/CrossSections.md?ref_type=heads
        # for singleH at 13.6 TeV: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap
        # for singleH at 13 TeV: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#gluon_gluon_Fusion_Process
        
        dict_xsec_13TeV = {
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.030649e3 * 0.00227 * 0.576 * 2,#0.033969e3 * 0.00227 * 0.576 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct. 
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.068317e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.013422e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.090488e3 * 0.00227 * 0.576 * 2,

            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": 0.132486e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": 0.016068e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": 0.009427e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": 2.617158e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": 1.791638e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": 1.752648e3 * 0.00227 * 0.576 * 2, 

            "VBFHH_CV_1_C2V_1_C3_1": 0.0017260e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_1_C2V_0_C3_1": 0.0270800e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_1p74_C2V_1p37_C3_14p4": 0.3777832e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": 0.6322811e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": 0.0000120e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": 0.3340766e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": 0.0009976e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": 0.0033739e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": 0.0105109e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": 0.0149850e3 * 0.00227 * 0.576 * 2,

            # For singleH, XS(process) * BR(HtoGG)
            # Using mH = 125.4 unless specified otherwise
            "ttHtoGG_M_125": 0.5033e3 * 0.00227,
            "BBHto2G_M_125": 0.5223e3 * 0.00227,
            "GluGluHToGG_M_125": 48.30e3 * 0.00227,
            "VBFHToGG_M_125": 3.770e3 * 0.00227,
            "VHtoGG_M_125": 2.2347e3 * 0.00227, # XS is sum of WH and ZH

            "DDQCDGJET": 1.0,
            "TTG_10_100": 4.216e3,
            "TTG_100_200": 0.4114e3,
            "TTG_200": 0.1284e3,
            "TT": 762.3e3,
            "GGJets": 88.75e3,
            "TTGG": 0.02391e3,
        }
        dict_xsec_13p6TeV = {
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.033969e3 * 0.00227 * 0.576 * 2,#0.033969e3 * 0.00227 * 0.576 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct. 
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.075495e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.014864e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.099298e3 * 0.00227 * 0.576 * 2,

            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": 2.900686e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": 0.010448e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": 0.146839e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": 0.017809e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": 1.985733e3 * 0.00227 * 0.576 * 2,
            "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": 1.752648e3 * 1.108268907 * 0.00227 * 0.576 * 2, 

            "VBFHH_CV_1_C2V_1_C3_1": 0.0019292e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_1_C2V_0_C3_1": 0.0296772e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_1p74_C2V_1p37_C3_14p4": 0.4002163e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": 0.6800842e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": 0.0000127e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": 0.3593242e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": 0.0011275e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": 0.0037987e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": 0.0117008e3 * 0.00227 * 0.576 * 2,
            "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": 0.0168528e3 * 0.00227 * 0.576 * 2,

            # For singleH, XS(process) * BR(HtoGG)
            # Using mH = 125.38 unless specified otherwise
            "ttHtoGG_M_125": 0.5638e3 * 0.00227,
            "BBHto2G_M_125": 0.5251e3 * 0.00227, # mH = 125.09
            "GluGluHToGG_M_125": 51.96e3 * 0.00227,
            "VBFHToGG_M_125": 4.067e3 * 0.00227,
            "VHtoGG_M_125": 2.3781e3 * 0.00227, # XS is sum of WH and ZH
            "WmHtoGG": 0.8801e3 * 0.00227,
            "WpHtoGG": 0.5620e3 * 0.00227,
            "ZHtoGG": 0.9361e3 * 0.00227,

            "DDQCDGJET": 1.0,
            "TTG_10_100": 4.216e3,
            "TTG_100_200": 0.4114e3,
            "TTG_200": 0.1284e3,
            "TT": 762.3e3,
            "GGJets": 87.51e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.02391e3,
        }

        luminosities = {
        "2016preVFP": 19.5,
        "2016postVFP": 16.8,
        "2017": 42.07,
        "2018": 59.56,
        "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
        "postEE": 26.67,  # Integrated luminosity for postEE in fb^-1
        "preBPix": 18.06,  # Integrated luminosity for preEE in fb^-1
        "postBPix": 9.89,  # Integrated luminosity for postEE in fb^-1
        "2024": 108.82,
        "2025": 110.58,
        }

        lumi = luminosities[era]
        if sample_type == "DDQCDGJET":
            lumi = 1.0

        if era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            events["weight_tot"] = (events.weight) * dict_xsec_13TeV[sample_type] * lumi
        
        elif era in ["preEE", "postEE", "preBPix", "postBPix", "2024", "2025"]:
            events["weight_tot"] = (events.weight) * dict_xsec_13p6TeV[sample_type] * lumi
        else:
            raise ValueError(f"Unknown era: {era}")

        return events

    def get_weights_for_training(self, y_train, rel_w_train, proc_num_train):

        true_class_weights = ak.zeros_like(rel_w_train)
        class_weights_for_training_abs = ak.zeros_like(rel_w_train)
        class_weights_only_positive = ak.zeros_like(rel_w_train)

        for i in range(y_train.shape[1]):

            class_name = self.class_idx_to_name[i]
            class_scale = self.class_weight_scale.get(class_name, 1.0)
            sample_weight_scale_by_process = self.sample_weight_scale_by_process

            if self.weight_scheme_process[class_name] == "equal_weight":
                cls_bool = (y_train[:, i] == 1)

                true_class_weights_ = ak.zeros_like(rel_w_train)
                class_weights_for_training_abs_ = ak.zeros_like(rel_w_train)
                class_weights_only_positive_ = ak.zeros_like(rel_w_train)

                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    rel_xsec_weight_for_class = rel_w_train * cls_bool * (proc_num_train == proc)
                    true_class_weights_ = true_class_weights_ + sample_scale * (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

                    abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool * (proc_num_train == proc)
                    class_weights_for_training_abs_ = class_weights_for_training_abs_ + sample_scale * (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

                    only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0) * (proc_num_train == proc)
                    class_weights_only_positive_ = class_weights_only_positive_ + sample_scale * (only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class))

                # normalize the weights for this class to be class_scale
                true_class_weights = true_class_weights + class_scale * (true_class_weights_ / np.sum(true_class_weights_))
                class_weights_for_training_abs = class_weights_for_training_abs + class_scale * (class_weights_for_training_abs_ / np.sum(class_weights_for_training_abs_))
                class_weights_only_positive = class_weights_only_positive + class_scale * (class_weights_only_positive_ / np.sum(class_weights_only_positive_))
                
            else:
                cls_bool = (y_train[:, i] == 1)

                rel_xsec_weight_for_class = rel_w_train * cls_bool
                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    if sample_scale != 1.0:
                        rel_xsec_weight_for_class[proc_num_train == proc] *= sample_scale
                true_class_weights = true_class_weights + class_scale * (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

                abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    if sample_scale != 1.0:
                        abs_rel_xsec_weight_for_class[proc_num_train == proc] *= sample_scale
                class_weights_for_training_abs = class_weights_for_training_abs + class_scale * (abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class))

                only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0)
                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    if sample_scale != 1.0:
                        only_positive_rel_xsec_weight_for_class[proc_num_train == proc] *= sample_scale
                class_weights_only_positive = class_weights_only_positive + class_scale * (only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class))

        for i in range(y_train.shape[1]):
            print(f"(number of events: sum of class_weights_for_training_abs) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_for_training_abs[y_train[:, i] == 1])})")
            print(f"(number of events: sum of class_weights_only_positive) for class number {i+1} = ({sum(y_train[:, i])}: {sum(class_weights_only_positive[y_train[:, i] == 1])})")
            print(f"(number of events: sum of true_class_weights) for class number {i+1} = ({sum(y_train[:, i])}: {sum(true_class_weights[y_train[:, i] == 1])})")

            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                print("\n")
                for proc in self.samples_in_class[self.class_idx_to_name[i]]:
                    print(f"(number of events: sum of class_weights_for_training_abs) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(class_weights_for_training_abs * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                    print(f"(number of events: sum of class_weights_only_positive) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(class_weights_only_positive * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                    print(f"(number of events: sum of true_class_weights) for class number {i+1} and process number {proc} = ({sum(y_train[:, i] * (proc_num_train == proc))}: {sum(true_class_weights * (y_train[:, i] == 1) * (proc_num_train == proc))})")
                print("\n")

        return true_class_weights, class_weights_for_training_abs, class_weights_only_positive

    def get_weights_for_val_test(self, y_val, rel_w_val, proc_num_val):

        class_weights_for_val = ak.zeros_like(rel_w_val)

        for i in range(y_val.shape[1]):
            class_name = self.class_idx_to_name[i]
            class_scale = self.class_weight_scale.get(class_name, 1.0)
            sample_weight_scale_by_process = self.sample_weight_scale_by_process
            
            if self.weight_scheme_process[class_name] == "equal_weight":
                cls_bool = (y_val[:, i] == 1)
                class_weights_for_val_ = ak.zeros_like(rel_w_val)

                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    rel_xsec_weight_for_class = rel_w_val * cls_bool * (proc_num_val == proc)
                    class_weights_for_val_ = class_weights_for_val_ + sample_scale * (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

                # normalize the weights for this class to be class_scale
                class_weights_for_val = class_weights_for_val + class_scale * (class_weights_for_val_ / np.sum(class_weights_for_val_))

            else:
                cls_bool = (y_val[:, i] == 1)
                rel_xsec_weight_for_class = rel_w_val * cls_bool
                for proc in self.samples_in_class[class_name]:
                    sample_scale = sample_weight_scale_by_process[proc]
                    if sample_scale != 1.0:
                        rel_xsec_weight_for_class[proc_num_val == proc] *= sample_scale
                class_weights_for_val = class_weights_for_val + class_scale * (rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class))

        for i in range(y_val.shape[1]):
            print(f"(number of events: sum of class_weights_for_val) for class number {i+1} = ({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})")

            if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
                print("\n")
                for proc in self.samples_in_class[self.class_idx_to_name[i]]:
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

            mask = ((events[var] > -998.0) & (events.nonResReg_dijet_mass > -998.0))
            nonResReg_dijet_mass = events.nonResReg_dijet_mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 2] = np.corrcoef(nonResReg_dijet_mass, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.nonResReg_dijet_mass_DNNreg > -998.0))
            nonResReg_dijet_mass_DNNreg = events.nonResReg_dijet_mass_DNNreg[mask]
            var_values = events[var][mask]
            corr_matrix[i, 3] = np.corrcoef(nonResReg_dijet_mass_DNNreg, var_values)[0, 1]

            mask = ((events[var] > -998.0) & (events.nonResReg_vbfpair_dijet_mass > -998.0))
            nonResReg_vbfpair_dijet_mass = events.nonResReg_vbfpair_dijet_mass[mask]
            var_values = events[var][mask]
            corr_matrix[i, 4] = np.corrcoef(nonResReg_vbfpair_dijet_mass, var_values)[0, 1]

        # plot the correlation matrix
        plt.figure(figsize=(18, len(vars_for_training)))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        # annotate the values
        for i in range(len(vars_for_training)):
            for j in range(5):
                # format the value to 2 decimal places
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='b')

        plt.xticks([0, 1, 2, 3, 4], ['mass', 'nonRes_dijet_mass', 'nonResReg_dijet_mass', 'nonResReg_dijet_mass_DNNreg', 'nonResReg_vbfpair_dijet_mass'], rotation=90)
        plt.yticks(range(len(vars_for_training)), vars_for_training)
        plt.colorbar()
        plt.savefig(f'{out_path}', dpi=300, )
        plt.clf()
        plt.close()

    def plot_correlation_matrix(self, events, vars_for_training, out_path):
        corr_matrix = np.zeros([len(vars_for_training), len(vars_for_training)])

        for i in range(len(vars_for_training)):
            var_i = vars_for_training[i]
            for j in range(len(vars_for_training)):
                var_j = vars_for_training[j]

                # calculate pair-wise correlation, do not include -999 values
                mask = ((events[var_i] > -998.0) & (events[var_j] > -998.0))
                values_i = events[var_i][mask]
                values_j = events[var_j][mask]

                if len(values_i) > 1:
                    corr_matrix[i, j] = np.corrcoef(values_i, values_j)[0, 1]
                else:
                    corr_matrix[i, j] = np.nan

        # plot the correlation matrix
        plt.figure(figsize=(len(vars_for_training), len(vars_for_training)))
        plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')

        # annotate the values
        for i in range(len(vars_for_training)):
            for j in range(len(vars_for_training)):
                plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha='center', va='center', color='b')

        plt.xticks(range(len(vars_for_training)), vars_for_training, rotation=90)
        plt.yticks(range(len(vars_for_training)), vars_for_training)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{out_path}', dpi=300)
        plt.clf()
        plt.close()
    
    def preselection(self, events):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.nonResReg_vbfpair_dijet_mass > 70) & (events.nonResReg_vbfpair_dijet_mass < 190))

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
            print(f"Plotting variable: {var}")

            plt.figure(figsize=(10, 9))
            sample_num = 0
            data_to_plot_dict = {}
            range_list = []
            for sample in self.sample_to_class.keys():
                
                #if sample in ["DDQCDGJET", ]:
                #    continue
                print(var)
                print(sample)

                sample_events = comb_inputs[comb_inputs["sample_type"] == sample]
                print(len(sample_events))
                data_to_plot = sample_events[var]
                mask = (data_to_plot > -998.0)
                print(len(data_to_plot))
                data_to_plot = data_to_plot[mask]
                data_to_plot_dict[sample] = data_to_plot

                if len(data_to_plot) == 0:
                    print(f"No entries for {var}, {sample} when plotting variables.")
                else:
                    if range_list == []:
                        range_list = [min(data_to_plot), max(data_to_plot)]
                    else:
                        range_list[0] = min(range_list[0], min(data_to_plot))
                        range_list[1] = max(range_list[1], max(data_to_plot))

            for sample in self.sample_to_class.keys():
                #if sample in ["DDQCDGJET", ]:
                #    continue
                data_to_plot = data_to_plot_dict[sample]
                
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
            plt.close()


    def prep_inputs_for_training(self):
        self._reset_input_file_counter()

        fill_nan = self.fill_nan

        out_path = self.outpath
        os.makedirs(out_path, exist_ok=True)

        comb_inputs = pd.DataFrame()

        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        vars_for_training = vars_config["vars"] 
        # vars_for_log = vars_config["vars_for_log_transform"]

        # Dictionary to accumulate events by sample (across all eras)
        sample_events_dict = {sample: [] for sample in self.sample_to_class.keys()}

        stop_processing = False

        for era in self.training_info["samples_info"]["eras"]:

            vars_to_load = vars_for_training + self.extra_vars_train

            # remove the UParTAK4B if NanoAODv12/v13
            if any(x in era for x in ["preEE", "postEE", "preBPix", "postBPix"]):
                vars_to_load.remove("nonResReg_vbfpair_lead_bjet_btagUParTAK4B")
                vars_to_load.remove("nonResReg_vbfpair_sublead_bjet_btagUParTAK4B")
    
            # for samples in self.sample_to_class.keys(): 
            for samples in self.training_info["samples_info"][era].keys():                
                if stop_processing or not self._can_process_more_input_files():
                    stop_processing = True
                    break

                samples_path = self.training_info["samples_info"]["samples_path"]
                parquet_path = self.training_info["samples_info"][era][samples]
                events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)
                events = self._apply_input_caps(events)
                stop_processing = self._record_input_file()

                events = self.preselection(events)

                # add more variables
                events = self.add_var(events, era)

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)

                events = events[vars_to_load + ["weight_tot"]]

                # apply mHH bin filter (if configured) and skip sample if empty
                if self.mhh_var is not None and self.mhh_range is not None:
                    events = self._apply_mhh_filter(events)
                    if len(events) == 0:
                        print(f"WARNING: No events left in sample {samples} for era {era} after mHH filter {self.mhh_range}. Skipping.")
                        continue

                print(f"INFO: Number of MC events in {samples} after selection for {era}: {len(events)}")
                print(f"INFO: Sum of weight_tot in {samples} after selection for {era}: {sum(events.weight_tot)}")

                # add the bools for each class
                for cls in self.classes:  # first intialize everything to zero
                    events[cls] = ak.zeros_like(events.eta)

                events[self.sample_to_class[samples]] = ak.ones_like(events.eta) # one-hot encoded

                # comb_inputs.append(events)
                events["sample_type"] = samples

                # add globally unique process number
                events["process_number"] = self.process_numbers[samples]

                # # store events for plot of correlation combining all eras)
                # sample_events_dict[samples].append(events)

                # plot_correlation_matrix
                os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
                corr_out_path = f"{out_path}/correlation_matrix/corr_mgg_mjj_{samples}_{era}.pdf"
                # self.corr_with_mgg_mjj(events, vars_for_training, corr_out_path)
                # self.plot_correlation_matrix(events, vars_for_training, f"{out_path}/correlation_matrix/corr_matrix_{samples}_{era}.pdf")

                print("INFO: Appending process samples to whole dataframe")

                i = 0
                while len(events) > 0:
                    events_intermediate = events[:self.write_chunk]
                    events = events[self.write_chunk:]
                    comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                    i+=1

                # events = pd.DataFrame(ak.to_list(events))
                # comb_inputs = pd.concat([comb_inputs, events])
                if stop_processing:
                    print("INFO: Maximum input file limit reached for lightweight prepare mode. Stopping early.")
                    break
            if stop_processing:
                break

        # # Plot correlation matrices (one per sample, combining all eras)
        # print("\nINFO: Computing correlation matrices for each sample (combined across all eras)")
        # os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
        
        # for sample in self.sample_to_class.keys():
        #     if len(sample_events_dict[sample]) == 0:
        #         continue
            
        #     # Concatenate all eras for this sample
        #     sample_combined = ak.concatenate(sample_events_dict[sample])
            
        #     print(f"INFO: Plotting correlation matrices for {sample} ({len(sample_combined)} events from all eras)")
            
        #     # Plot corr_mgg_mjj
        #     self.corr_with_mgg_mjj(sample_combined, vars_for_training, f"{out_path}/correlation_matrix/corr_mgg_mjj_{sample}.pdf")
            
        #     # Plot full correlation matrix
        #     self.plot_correlation_matrix(sample_combined, vars_for_training, f"{out_path}/correlation_matrix/corr_matrix_{sample}.pdf")

        print("INFO: Plotting variables")
        plot_path = f"{out_path}/var_plots/"
        if len(comb_inputs) == 0:
            raise ValueError(
                "No events remained after lightweight caps and selections. "
                "Increase --max_input_files or --max_rows_per_file."
            )

        os.makedirs(plot_path, exist_ok=True)
        self.plot_variables(comb_inputs, vars_for_training, plot_path)
        for cls in self.classes:
            print("\n", f"INFO: Number of events in {cls}: {sum(comb_inputs[cls])}")

        print("\n=== Sanity check: per-sample feature missingness ===")
        problematic = []
        for sample in self.sample_to_class.keys():
            sample_mask = comb_inputs["sample_type"] == sample
            n = sample_mask.sum()
            if n == 0:
                print(f"  [WARN] {sample}: 0 events after preselection")
                continue
            for var in vars_for_training:
                n_valid = ((comb_inputs.loc[sample_mask, var] > -998.0)).sum()
                frac_missing = 1 - n_valid / n
                if frac_missing > 0.99:   # >99% missing
                    problematic.append((sample, var, frac_missing, n))
                    print(f"  [LEAK?] {sample} / {var}: "
                        f"{frac_missing*100:.1f}% missing ({n} events) — "
                        f"will become constant -9 → label leakage risk")

        X = comb_inputs[vars_for_training]
        Y = comb_inputs[[cls for cls in self.classes]]
        relative_weights = comb_inputs["weight_tot"]

        # perform log transformation for variables if needed
        # for var in vars_for_log:
        #     X[var] = np.log(X[var])
        process_number = comb_inputs["process_number"].values
        del comb_inputs
        X = X.values
        Y = Y.values
        relative_weights = relative_weights.values
        
        # mask -999.0 to nan
        mask = (X < -998.0)
        X[mask] = np.nan

        X_train, X_val, X_test, y_train, y_val, y_test, rel_w_train, rel_w_val, rel_w_test, proc_num_train, proc_num_val, proc_num_test = self.train_test_split(X, Y, relative_weights, process_number)
        del process_number
        del X
        del Y
        del relative_weights

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

        with open(f"{out_path}/training_info.txt", 'w') as f:
            json.dump(self.training_info, f)

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
        
        # save globally unique process numbers
        np.save(f"{out_path}/proc_num_train", proc_num_train)
        np.save(f"{out_path}/proc_num_val", proc_num_val)
        
        if X_test is not None:
            np.save(f"{out_path}/X_test", X_test)
            np.save(f"{out_path}/rel_w_test", rel_w_test)
            np.save(f"{out_path}/y_test", y_test)
            np.save(f"{out_path}/class_weights_for_test", class_weights_for_test)
            np.save(f"{out_path}/proc_num_test", proc_num_test)

        # save process number mapping (sample name -> globally unique process number)
        with open(f"{out_path}/process_numbers_mapping.json", 'w') as f:
            # Convert to int for JSON serialization
            mapping = {sample: int(proc_num) for sample, proc_num in self.process_numbers.items()}
            json.dump(mapping, f, indent=2)
        
        # also save sample_to_class mapping for reference
        with open(f"{out_path}/sample_to_class_mapping.json", 'w') as f:
            json.dump(self.sample_to_class, f, indent=2)

        # save the training mean ans std_dev. This will be used for standardizing data
        mean_std_dict = {
            "mean": mean,
            "std_dev": std
        }
        with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
            pickle.dump(mean_std_dict, f)

        return 0
    
    def prep_inputs_for_prediction_sim(self):
        self._reset_input_file_counter()

        fill_nan = self.fill_nan
        training_info = self.training_info
        inputs_path = self.outpath
        # print(f"DEBUG: inputs_path = {inputs_path}")
        out_path = f"{inputs_path}/individual_samples/"
        os.makedirs(out_path, exist_ok=True)
        # get the variables required for training
        vars_config = self.load_vars(self.input_var_json)[self.model_type]

        with open(f"{inputs_path}/input_vars.txt", 'r') as f:
            vars = json.load(f)
        vars_for_training = vars
        reported_x_features = False

        # vars_for_log = vars_config["vars_for_log_transform"]
        vars_gen = ["gen_mHH_hardProc", "gen_pT_HH_hardProc", "gen_CosThetaStar_HH_hardProc"]

        samples_path = training_info["samples_info"]["samples_path"]

        stop_processing = False

        for era in training_info["samples_info"]["eras"]:

            vars_to_load = vars_for_training + self.extra_vars_train + self.extra_vars_out

            # remove the UParTAK4B if NanoAODv12/v13
            if any(x in era for x in ["preEE", "postEE", "preBPix", "postBPix"]):
                vars_to_load.remove("nonResReg_vbfpair_lead_bjet_btagUParTAK4B")
                vars_to_load.remove("nonResReg_vbfpair_sublead_bjet_btagUParTAK4B")
    
            for samples in training_info["samples_info"][era].keys():
                if stop_processing or not self._can_process_more_input_files():
                    stop_processing = True
                    break
                
                parquet_path = training_info["samples_info"][era][samples]
                if self.save_all_columns_sim_nominal:
                    events = ak.from_parquet(f"{samples_path}/{parquet_path}")
                elif ("GluGlutoHHto2B2G_kl" in samples):
                    vars_to_load = vars_to_load + vars_gen
                    events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)
                elif ("EFTReweighted" in samples):
                    vars_to_load = vars_to_load + vars_EFTReweighted + vars_gen
                    events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)
                else:
                    events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)
                events = self._apply_input_caps(events)
                stop_processing = self._record_input_file()

                print(f"INFO: Number of events in {samples} for {era}: {len(events)}")

                # add preselection
                events = self.preselection_for_pred(events)

                # add more variables
                events = self.add_var(events, era)

                # get relative weights according to cross section of the process
                events = self.get_relative_xsec_weight(events, samples, era)

                # apply mHH bin filter (if configured) and skip sample if empty
                if self.mhh_var is not None and self.mhh_range is not None:
                    events = self._apply_mhh_filter(events)
                    if len(events) == 0:
                        print(f"WARNING: No events left in sample {samples} for era {era} after mHH filter {self.mhh_range}. Skipping.")
                        continue

                # also save the event
                full_path_to_save = f"{out_path}/{era}/{samples}/"
                os.makedirs(full_path_to_save, exist_ok=True)
                ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                events = events[vars_for_training + ["weight_tot"]]

                comb_inputs = pd.DataFrame()
                i = 0
                while len(events) > 0:
                    events_intermediate = events[:self.write_chunk]
                    events = events[self.write_chunk:]
                    comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                    i+=1

                X = comb_inputs[vars_for_training]
                #Y = comb_inputs[[cls for cls in self.classes]]
                relative_weights = comb_inputs["weight_tot"]

                # perform log transformation for variables if needed
                # for var in vars_for_log:
                #     X[var] = np.log(X[var])

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
                if not reported_x_features:
                    print(f"INFO: Final prediction-X feature count: {X.shape[1]}")
                    reported_x_features = True
                print("INFO: saving inputs for mlp")
                np.save(f"{full_path_to_save}/X", X)
                np.save(f"{full_path_to_save}/rel_w", relative_weights)

                # save the training mean ans std_dev. This will be used for standardizing data
                mean_std_dict = {
                    "mean": mean,
                    "std_dev": std
                }
                with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
                    pickle.dump(mean_std_dict, f)

                if stop_processing:
                    print("INFO: Maximum input file limit reached for lightweight prepare mode. Stopping early.")
                    break
            if stop_processing:
                break

        return 0

    def prep_inputs_for_prediction_sim_sys(self):
        self._reset_input_file_counter()

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
        reported_x_features = False

        # vars_for_log = vars_config["vars_for_log_transform"]

        samples_path = training_info["samples_info"]["samples_path"]

        stop_processing = False

        for era in training_info["samples_info"]["eras"]:

            vars_to_load = vars_for_training + self.extra_vars_train + self.extra_vars_out

            # remove the UParTAK4B if NanoAODv12/v13
            if any(x in era for x in ["preEE", "postEE", "preBPix", "postBPix"]):
                vars_to_load.remove("nonResReg_vbfpair_lead_bjet_btagUParTAK4B")
                vars_to_load.remove("nonResReg_vbfpair_sublead_bjet_btagUParTAK4B")
    
            for samples in training_info["samples_info"][era].keys():
                if stop_processing or not self._can_process_more_input_files():
                    stop_processing = True
                    break

                for sys in training_info["systematics"]:
                    if stop_processing or not self._can_process_more_input_files():
                        stop_processing = True
                        break

                    if samples in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
                        continue
                
                    parquet_path = (training_info["samples_info"][era][samples]).replace("nominal", sys)
                    print(f"DEBUG: Checking existence of {samples} for {era} for {sys} at path: {samples_path}/{parquet_path}")
                    if not os.path.exists(f"{samples_path}/{parquet_path}"):
                        print(f"WARNING: {samples} for {era} for {sys} does not exist. Skipping.: {samples_path}/{parquet_path}")
                        continue
                    if self.save_all_columns_sim_systematics:
                        events = ak.from_parquet(f"{samples_path}/{parquet_path}")
                    else:
                        events = ak.from_parquet(f"{samples_path}/{parquet_path}", columns=vars_to_load)
                    events = self._apply_input_caps(events)
                    stop_processing = self._record_input_file()

                    print(f"INFO: Number of events in {samples} for {era} for {sys}: {len(events)}")

                    # add preselection
                    events = self.preselection_for_pred(events)

                    # add more variables
                    events = self.add_var(events, era)

                    # get relative weights according to cross section of the process
                    events = self.get_relative_xsec_weight(events, samples, era)

                    # apply mHH bin filter (if configured) and skip sample if empty
                    if self.mhh_var is not None and self.mhh_range is not None:
                        events = self._apply_mhh_filter(events)
                        if len(events) == 0:
                            print(f"WARNING: No events left in sample {samples} for era {era} after mHH filter {self.mhh_range}. Skipping.")
                            continue

                    # also save the event
                    full_path_to_save = f"{out_path}/{era}/{samples}/{sys}/"
                    os.makedirs(full_path_to_save, exist_ok=True)
                    ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

                    events = events[vars_for_training + ["weight_tot"]]

                    comb_inputs = pd.DataFrame()
                    i = 0
                    while len(events) > 0:
                        events_intermediate = events[:self.write_chunk]
                        events = events[self.write_chunk:]
                        comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                        i+=1

                    X = comb_inputs[vars_for_training]
                    #Y = comb_inputs[[cls for cls in self.classes]]
                    relative_weights = comb_inputs["weight_tot"]

                    # perform log transformation for variables if needed
                    # for var in vars_for_log:
                    #     X[var] = np.log(X[var])

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
                    if not reported_x_features:
                        print(f"INFO: Final prediction-X feature count: {X.shape[1]}")
                        reported_x_features = True
                    #print("INFO: saving inputs for mlp")
                    np.save(f"{full_path_to_save}/X", X)
                    np.save(f"{full_path_to_save}/rel_w", relative_weights)

                    # save the training mean ans std_dev. This will be used for standardizing data
                    mean_std_dict = {
                        "mean": mean,
                        "std_dev": std
                    }
                    with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
                        pickle.dump(mean_std_dict, f)

                    if stop_processing:
                        print("INFO: Maximum input file limit reached for lightweight prepare mode. Stopping early.")
                        break
            if stop_processing:
                break

        return 0
    
    def prep_inputs_for_prediction_data(self):
        self._reset_input_file_counter()

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
        reported_x_features = False

        # vars_for_log = vars_config["vars_for_log_transform"]

        samples_path = training_info["samples_info"]["samples_path"]
        datas = training_info["samples_info"]["data"]

        stop_processing = False

        sample_to_era = {
            "2016preVFP_EraBv1": "2016preVFP",
            "2016preVFP_EraBv2": "2016preVFP",
            "2016preVFP_EraC": "2016preVFP",
            "2016preVFP_EraD": "2016preVFP",
            "2016preVFP_EraE": "2016preVFP",
            "2016preVFP_EraF": "2016preVFP",
            "2016postVFP_EraF": "2016postVFP",
            "2016postVFP_EraG": "2016postVFP",
            "2016postVFP_EraH": "2016postVFP",
            "2017_EraB": "2017",
            "2017_EraC": "2017",
            "2017_EraD": "2017",
            "2017_EraE": "2017",
            "2017_EraF": "2017",
            "2018_EraA": "2018",
            "2018_EraB": "2018",
            "2018_EraC": "2018",
            "2018_EraD": "2018",
            "2017": "2017",
            "2018": "2018",
            "2022_EraE": "postEE", 
            "2022_EraF": "postEE", 
            "2022_EraG": "postEE", 
            "2022_EraC": "preEE", 
            "2022_EraD": "preEE",
            "2023_EraC": "preBPix",
            "2023_EraD": "postBPix",
            "2024_EraC_EG0": "2024",
            "2024_EraC_EG1": "2024",
            "2024_EraD_EG0": "2024",
            "2024_EraD_EG1": "2024",
            "2024_EraE_EG0": "2024",
            "2024_EraE_EG1": "2024",
            "2024_EraF_EG0": "2024",
            "2024_EraF_EG1": "2024",
            "2024_EraG_EG0": "2024",
            "2024_EraG_EG1": "2024",
            "2024_EraH_EG0": "2024",
            "2024_EraH_EG1": "2024",
            "2024_EraIv1_EG0": "2024",
            "2024_EraIv1_EG1": "2024",
            "2024_EraIv2_EG0": "2024",
            "2024_EraIv2_EG1": "2024",
            "2025_EraCv1_EG0": "2025",
            "2025_EraCv1_EG1": "2025",
            "2025_EraCv1_EG2": "2025",
            "2025_EraCv1_EG3": "2025",
            "2025_EraCv2_EG0": "2025",
            "2025_EraCv2_EG1": "2025",
            "2025_EraCv2_EG2": "2025",
            "2025_EraCv2_EG3": "2025",
            "2025_EraDv1_EG0": "2025",
            "2025_EraDv1_EG1": "2025",
            "2025_EraDv1_EG2": "2025",
            "2025_EraDv1_EG3": "2025",
            "2025_EraEv1_EG0": "2025",
            "2025_EraEv1_EG1": "2025",
            "2025_EraEv1_EG2": "2025",
            "2025_EraEv1_EG3": "2025",
            "2025_EraFv1_EG0": "2025",
            "2025_EraFv1_EG1": "2025",
            "2025_EraFv1_EG2": "2025",
            "2025_EraFv1_EG3": "2025",
            "2025_EraFv2_EG0": "2025",
            "2025_EraFv2_EG1": "2025",
            "2025_EraFv2_EG2": "2025",
            "2025_EraFv2_EG3": "2025",
            "2025_EraGv1_EG0": "2025",
            "2025_EraGv1_EG1": "2025",
            "2025_EraGv1_EG2": "2025",
            "2025_EraGv1_EG3": "2025",
        }

        for data in datas:

            vars_to_load = vars_for_training + self.extra_vars_train + self.extra_vars_out

            # remove the UParTAK4B if NanoAODv12/v13
            if any(x in sample_to_era.get(data, "") for x in ["preEE", "postEE", "preBPix", "postBPix"]):
                vars_to_load.remove("nonResReg_vbfpair_lead_bjet_btagUParTAK4B")
                vars_to_load.remove("nonResReg_vbfpair_sublead_bjet_btagUParTAK4B")
    
            if stop_processing or not self._can_process_more_input_files():
                stop_processing = True
                break

            if self.save_all_columns_data:
                events = ak.from_parquet(f"{samples_path}/{datas[data]}")
            else:
                events = ak.from_parquet(f"{samples_path}/{datas[data]}", columns=vars_to_load)
            events = self._apply_input_caps(events)
            stop_processing = self._record_input_file()

            # add preselection
            events = self.preselection_for_pred(events)

            # apply mHH bin filter (if configured) and skip sample if empty
            if self.mhh_var is not None and self.mhh_range is not None:
                events = self._apply_mhh_filter(events)
                if len(events) == 0:
                    print(f"WARNING: No events left in sample {data} after mHH filter {self.mhh_range}. Skipping.")
                    continue

            # add more variables
            events = self.add_var(events, sample_to_era.get(data, "2024"))

            # also save the event
            full_path_to_save = f"{out_path}/{data}/"
            os.makedirs(full_path_to_save, exist_ok=True)
            ak.to_parquet(events, f"{full_path_to_save}/events.parquet")

            events = events[vars_for_training]

            comb_inputs = pd.DataFrame()
            i = 0
            while len(events) > 0:
                events_intermediate = events[:self.write_chunk]
                events = events[self.write_chunk:]
                comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                i += 1

            X = comb_inputs[vars_for_training]

            # perform log transformation for variables if needed
            # for var in vars_for_log:
            #     X[var] = np.log(X[var])

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
            if not reported_x_features:
                print(f"INFO: Final prediction-X feature count: {X.shape[1]}")
                reported_x_features = True
            print(f"INFO: saving inputs for {data}")
            #full_path_to_save = f"{out_path}/"
            np.save(f"{full_path_to_save}/X", X)

            # ak.to_parquet(events, f"{full_path_to_save}/events.parquet")
            mean_std_dict = {
                "mean": mean,
                "std_dev": std
            }
            with open(f"{out_path}/mean_std_dict.pkl", 'wb') as f:
                pickle.dump(mean_std_dict, f)

            if stop_processing:
                print("INFO: Maximum input file limit reached for lightweight prepare mode. Stopping early.")
                break

        return
    

    


    
