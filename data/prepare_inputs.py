import awkward as ak
import numpy as np
import yaml
import json
import os
import hashlib
from typing import Any, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import mplhep as hep
import matplotlib.pyplot as plt
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
    split_select: Optional[str] = None,
  ) -> None:
    self.model_type = "mlp"
    self.input_var_json = input_var_json
    self.training_info = training_info
    self.outpath = outpath
    self.predict_parquet_info = predict_parquet_info

    self.split_select = split_select

    if self.split_select not in [None, "train", "final"]:
      raise ValueError(
        f"split_select must be None, 'train', or 'final'. Got: {self.split_select}"
      )

    self.split_data = False
    self.train_sample_fraction = 0.5
    self.split_seed = 12345
    self.train_split_dir_name = "Train_split"
    self.final_split_dir_name = "Final_split"

    if self.training_info is not None:
      self.sample_to_class = self.training_info["sample_to_class"]
      self.classes = self.training_info["classes"]
      self.random_seed = self.training_info["random_seed"]
      self.weight_scheme_process = self.training_info["weight_scheme_process"]
      self.class_weight_scale = self.training_info.get("class_weight_scale", None)
      self.var_prefix = self.training_info.get("var_prefix", "nonResReg")

      # split_data applies only to real data; MC is always split.
      self.split_data = self.training_info.get("split_data", False)
      self.train_sample_fraction = self.training_info.get("train_sample_fraction", 0.5)
      self.split_seed = self.training_info.get("split_seed", 12345)

      self.train_split_dir_name = self.training_info.get(
        "train_split_dir_name",
        "Train_split",
      )
      self.final_split_dir_name = self.training_info.get(
        "final_split_dir_name",
        "Final_split",
      )

      self.save_all_columns_sim_nominal = training_info["save_all_columns_sim_nominal"]
      self.save_all_columns_data = training_info["save_all_columns_data"]
      self.save_all_columns_sim_systematics = training_info["save_all_columns_sim_systematics"]
    else:
      self.var_prefix = "nonResReg"

    self.mc_split_weight_scale = 1.0
    if self.split_select is not None:
      if not 0.0 < self.train_sample_fraction < 1.0:
        raise ValueError("train_sample_fraction must be greater than 0 and less than 1")

      split_fraction = (
        self.train_sample_fraction
        if self.split_select == "train"
        else 1.0 - self.train_sample_fraction
      )
      self.mc_split_weight_scale = 1.0 / split_fraction
      print(
        f"INFO: MC weights for {self.split_select} split are scaled by "
        f"{self.mc_split_weight_scale:g} to represent the full MC yield"
      )

    self.fill_nan = -9

    self.extra_vars = [
      "mass",
      "nonRes_dijet_mass",
      f"{self.var_prefix}_dijet_mass",
      f"{self.var_prefix}_dijet_mass_DNNreg",
      f"{self.var_prefix}_HHbbggCandidate_mass",
      f"{self.var_prefix}_dijet_pt",
      f"{self.var_prefix}_lead_bjet_pt",
      f"{self.var_prefix}_sublead_bjet_pt",
      f"{self.var_prefix}_lead_bjet_eta",
      f"{self.var_prefix}_DNNpair_dijet_mass",
      f"{self.var_prefix}_DNNpair_dijet_mass_DNNreg",
      f"{self.var_prefix}_lead_bjet_btagPNetB",
      f"{self.var_prefix}_sublead_bjet_btagPNetB",
      f"{self.var_prefix}_lead_bjet_btagUParTAK4B",
      f"{self.var_prefix}_sublead_bjet_btagUParTAK4B",
      "weight",
      "pt",
      "nonRes_dijet_pt",
      "nonRes_HHbbggCandidate_mass",
      "eta",
      "nBTight",
      "nBMedium",
      "nBLoose",
      "nonRes_lead_bjet_pt",
      "nonRes_sublead_bjet_pt",
      "lead_isScEtaEB",
      "lead_isScEtaEE",
      "sublead_isScEtaEB",
      "sublead_isScEtaEE",
      "lead_mvaID",
      "sublead_mvaID",
      "lead_eta",
      "lead_phi",
      "sublead_eta",
      "sublead_phi",
    ]

    self.vars_for_boosted = [
      "sublead_mvaID",
      "fatjet3_tau2",
      "fatjet3_particleNet_XbbVsQCD",
      "fatjet4_subjet2_eta",
      "sublead_eta",
      "fatjet2_phi",
      "fatjet1_mass",
      f"{self.var_prefix}_CosThetaStar_gg",
      "fatjet4_particleNet_XbbVsQCD",
      "lead_phi",
      "fatjet4_pt",
      "fatjet4_tau1",
      "fatjet4_tau2",
      "fatjet2_particleNet_XbbVsQCD",
      "fatjet3_subjet1_eta",
      "fatjet1_subjet1_eta",
      "lead_eta",
      "fatjet3_msoftdrop",
      "fatjet4_mass",
      "fatjet4_particleNet_massCorr",
      "fatjet1_tau1",
      "eta",
      "fatjet2_pt",
      "phi",
      "fatjet1_subjet2_phi",
      "fatjet3_eta",
      "fatjet1_subjet2_eta",
      f"{self.var_prefix}_phosublead_PtOverM",
      "fatjet4_subjet1_phi",
      "fatjet3_subjet2_phi",
      "fatjet3_subjet1_phi",
      "fatjet2_tau2",
      "n_jets",
      "fatjet2_msoftdrop",
      "fatjet2_subjet2_phi",
      "fatjet3_pt",
      "fatjet2_eta",
      "fatjet3_tau1",
      "fatjet4_eta",
      "fatjet1_eta",
      "fatjet3_mass",
      "n_fatjets",
      "fatjet1_pt",
      "fatjet3_subjet2_eta",
      "fatjet1_subjet1_phi",
      "fatjet1_msoftdrop",
      "lead_mvaID",
      "fatjet4_subjet1_eta",
      f"{self.var_prefix}_pholead_PtOverM",
      "fatjet2_tau1",
      "fatjet2_mass",
      "fatjet2_subjet2_eta",
      "fatjet3_phi",
      "n_leptons",
      "fatjet1_particleNet_massCorr",
      "fatjet2_subjet1_phi",
      "fatjet4_subjet2_phi",
      "fatjet1_tau2",
      "fatjet1_phi",
      "fatjet2_subjet1_eta",
      "fatjet4_phi",
      "fatjet1_particleNet_XbbVsQCD",
      "fatjet3_particleNet_massCorr",
      "fatjet4_msoftdrop",
      "sublead_phi",
      "fatjet2_particleNet_massCorr",
    ]

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

  def load_vars(self, path):
    with open(path, "r") as f:
      vars = yaml.safe_load(f)
    return vars

  def substitute_var_prefix(self, var_list):
    return [
      var.replace("regcol_", f"{self.var_prefix}_")
      if "regcol_" in var
      else var
      for var in var_list
    ]

  @staticmethod
  def normalize_era(era):
    era_aliases = {
      "2022preEE": "preEE",
      "2022postEE": "postEE",
      "2023preBPix": "preBPix",
      "2023postBPix": "postBPix",
    }
    return era_aliases.get(era, era)

  def _mem_mb(self):
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) / 1024
    return -1

  def _stable_file_seed(self, file_path):
    digest = hashlib.md5(file_path.encode("utf-8")).hexdigest()
    file_int = int(digest[:8], 16)
    return int(self.split_seed + file_int) % (2**32 - 1)

  def _get_split_outpath(self, base_outpath):
    if self.split_select == "train":
      return f"{base_outpath}/{self.train_split_dir_name}"

    if self.split_select == "final":
      return f"{base_outpath}/{self.final_split_dir_name}"

    return base_outpath
    
  def _fast_row_split_mask(self, file_path, row_start, batch_len):
    row_indices = np.arange(row_start, row_start + batch_len, dtype=np.uint64)

    file_seed = np.uint64(self._stable_file_seed(file_path))
    split_seed = np.uint64(self.split_seed)

    hashed = row_indices
    hashed = hashed ^ file_seed
    hashed = hashed ^ (split_seed << np.uint64(16))

    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    hashed = hashed ^ (hashed >> np.uint64(31))

    random_values = hashed.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)

    return random_values < self.train_sample_fraction

  def _source_row_split_mask(
    self,
    file_path,
    n_rows,
    row_start,
    batch_len,
    force_split=True,
  ):
    if self.split_select is None:
      return np.ones(batch_len, dtype=bool)

    # For real data only:
    # force_split=False means:
    #   Train_split gets no data events.
    #   Final_split gets all data events.
    if not force_split:
      if self.split_select == "final":
        return np.ones(batch_len, dtype=bool)

      if self.split_select == "train":
        return np.zeros(batch_len, dtype=bool)

    # For MC:
    # always split into Train_split and Final_split.
    is_train = self._fast_row_split_mask(
      file_path=file_path,
      row_start=row_start,
      batch_len=batch_len,
    )

    if self.split_select == "train":
      return is_train

    if self.split_select == "final":
      return ~is_train

    return np.ones(batch_len, dtype=bool)

  def _iter_file_batched(
    self,
    file_path,
    vars_to_load,
    save_all_columns,
    era,
    preselection_func,
    xsec_sample_name,
    apply_xsec_weights,
    force_split=True,
  ):
    pf = pq.ParquetFile(file_path)
    n_rows_total = pf.metadata.num_rows

    batch_size = self.training_info.get("parquet_batch_size", 50000)
    batch_size = max(1, min(batch_size, n_rows_total))

    columns = None if save_all_columns else vars_to_load

    row_start = 0

    for record_batch in pf.iter_batches(batch_size=batch_size, columns=columns):
      batch_len = len(record_batch)

      split_mask = self._source_row_split_mask(
        file_path=file_path,
        n_rows=n_rows_total,
        row_start=row_start,
        batch_len=batch_len,
        force_split=force_split,
      )

      row_start += batch_len

      if not np.any(split_mask):
        del record_batch
        continue

      batch = ak.from_arrow(record_batch)
      del record_batch

      batch = batch[split_mask]

      batch = preselection_func(batch)

      if len(batch) > 0:
        batch = self.add_var(batch, era)

        if apply_xsec_weights:
          batch = self.get_relative_xsec_weight(batch, xsec_sample_name, era)

        for field in batch.fields:
          if hasattr(batch[field], "dtype") and batch[field].dtype == np.float64:
            batch[field] = ak.values_astype(batch[field], np.float32)

        yield batch

  def load_and_process_sample(
    self,
    samples_path,
    parquet_path,
    samples,
    era,
    vars_to_load,
    preselection_func,
    save_all_columns=False,
    systematic=None,
    apply_xsec_weights=True,
    force_split=True,
  ):
    if isinstance(parquet_path, list):
      print(f"INFO: Merging {len(parquet_path)} files for {samples} in {era}")

      vh_component_names = {
        "WmHtoGG": "WmHtoGG_M_125",
        "WpHtoGG": "WpHtoGG_M_125",
        "ZHtoGG": "ZHtoGG_M_125",
      }

      events_list = []

      for path in parquet_path:
        if systematic is not None:
          path = path.replace("nominal", systematic)
          if not os.path.exists(f"{samples_path}/{path}"):
            print(
              f"WARNING: {samples} for {era} for {systematic} does not exist. "
              f"Skipping.: {samples_path}/{path}"
            )
            continue

        component_name = None
        for key in vh_component_names.keys():
          if key in path:
            component_name = vh_component_names[key]
            break

        for batch in self._iter_file_batched(
          file_path=f"{samples_path}/{path}",
          vars_to_load=vars_to_load,
          save_all_columns=save_all_columns,
          era=era,
          preselection_func=preselection_func,
          xsec_sample_name=component_name,
          apply_xsec_weights=apply_xsec_weights,
          force_split=force_split,
        ):
          events_list.append(batch)

      if len(events_list) > 0:
        events = ak.concatenate(events_list, axis=0)
        del events_list
      else:
        events = None

    else:
      if systematic is not None:
        parquet_path = parquet_path.replace("nominal", systematic)
        if not os.path.exists(f"{samples_path}/{parquet_path}"):
          print(
            f"WARNING: {samples} for {era} for {systematic} does not exist. "
            f"Skipping.: {samples_path}/{parquet_path}"
          )
          return None

      batches = list(self._iter_file_batched(
        file_path=f"{samples_path}/{parquet_path}",
        vars_to_load=vars_to_load,
        save_all_columns=save_all_columns,
        era=era,
        preselection_func=preselection_func,
        xsec_sample_name=samples,
        apply_xsec_weights=apply_xsec_weights,
        force_split=force_split,
      ))

      events = ak.concatenate(batches, axis=0) if batches else None
      del batches

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

    return delta_r

  def add_var(self, events, era):
    era = self.normalize_era(era)

    events["diphoton_PtOverM_ggjj"] = events.pt / events[f"{self.var_prefix}_HHbbggCandidate_mass"]
    events[f"{self.var_prefix}_dijet_PtOverM_ggjj"] = events[f"{self.var_prefix}_dijet_pt"] / events[f"{self.var_prefix}_HHbbggCandidate_mass"]

    events[f"{self.var_prefix}_lead_bjet_over_M_regressed"] = events[f"{self.var_prefix}_lead_bjet_pt"] / events[f"{self.var_prefix}_dijet_mass_DNNreg"]
    events[f"{self.var_prefix}_sublead_bjet_over_M_regressed"] = events[f"{self.var_prefix}_sublead_bjet_pt"] / events[f"{self.var_prefix}_dijet_mass_DNNreg"]

    events["deltaR_gg"] = self.deltaR(events.lead_eta, events.lead_phi, events.sublead_eta, events.sublead_phi)

    btag_wp_config = {
      "2016preVFP": ("btagUParTAK4B", [0.0387, 0.1847, 0.5467, 0.6777, 0.9218]),
      "2016postVFP": ("btagUParTAK4B", [0.0400, 0.1898, 0.5538, 0.6872, 0.9353]),
      "2017": ("btagUParTAK4B", [0.0331, 0.1776, 0.5755, 0.7274, 0.9666]),
      "2018": ("btagUParTAK4B", [0.0308, 0.1610, 0.5405, 0.6992, 0.9655]),
      "preEE": ("btagPNetB", [0.0470, 0.2450, 0.6734, 0.7862, 0.9610]),
      "postEE": ("btagPNetB", [0.0499, 0.2605, 0.6915, 0.8033, 0.9664]),
      "preBPix": ("btagPNetB", [0.0358, 0.1917, 0.6172, 0.7515, 0.9659]),
      "postBPix": ("btagPNetB", [0.0359, 0.1919, 0.6133, 0.7544, 0.9688]),
      "2024": ("btagUParTAK4B", [0.0246, 0.1272, 0.4648, 0.6298, 0.9739]),
      "2025": ("btagUParTAK4B", [0.0246, 0.1272, 0.4648, 0.6298, 0.9739]),
    }
    if era not in btag_wp_config:
      raise ValueError(f"No b-tagging working points configured for era: {era}")

    discriminator, wps = btag_wp_config[era]
    btag_var = f"{self.var_prefix}_lead_bjet_{discriminator}"
    sub_btag_var = f"{self.var_prefix}_sublead_bjet_{discriminator}"

    wp_names = ["L", "M", "T", "XT", "XXT"]

    for wp_name, wp_value in zip(wp_names, wps):
      events[f"{self.var_prefix}_lead_bjet_btag_WP_{wp_name}"] = ak.values_astype(
        events[btag_var] > wp_value,
        int,
      )
      events[f"{self.var_prefix}_sublead_bjet_btag_WP_{wp_name}"] = ak.values_astype(
        events[sub_btag_var] > wp_value,
        int,
      )

    return events

  def get_relative_xsec_weight(self, events, sample_type, era):
    era = self.normalize_era(era)

    dict_xsec = {
            "GGJets": 86.96e3 if ("201" in era) else 88.75e3,
            "GJetPt20To40": 242.5e3,
            "GJetPt40": 919.1e3,
            "TTGG": 0.01696e3 if ("201" in era) else 0.02391e3,
            "ttHtoGG_M_125": 0.0011e3 if ("201" in era) else (0.5700e3 * 0.00227),
            "BBHto2G_M_125": 0.4385e3 * 0.00227,
            "GluGluHToGG_M_125": 0.1103e3 if ("201" in era) else (52.23e3 * 0.00227),
            "VBFHToGG_M_125": 0.00855e3 if ("201" in era) else (4.078e3 * 0.00227),
            "VHtoGG_M_125": 0.00508e3 if ("201" in era) else (2.4009e3 * 0.00227),
            "WpHtoGG_M_125": 0.880114e3 * 0.00227,
            "WmHtoGG_M_125": 0.562032e3 * 0.00227,
            "ZHtoGG_M_125": 0.9361e3 * 0.00227,
            # xs(HH) * BR(HToGG) * BR(HToGG) * 2 from https://gitlab.cern.ch/hh/recommendations/-/blob/master/CrossSections.md
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": (0.030922e3 * 0.00227 * 0.576 * 2) if ("201" in era) else (0.034170e3 * 0.00227 * 0.576 * 2),
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": (0.068321e3 * 0.00227 * 0.576 * 2) if ("201" in era) else (0.075766e3 * 0.00227 * 0.576 * 2),
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": (0.013405e3 * 0.00227 * 0.576 * 2) if ("201" in era) else (0.014814e3 * 0.00227 * 0.576 * 2),
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": (0.088012e3 * 0.00227 * 0.576 * 2) if ("201" in era) else (0.097259e3 * 0.00227 * 0.576 * 2),
            "VBFHH_CV_1p000_C2V_1p000_C3_1p000": (0.0017260e3 * 0.00227 * 0.576 * 2) if ("201" in era) else (0.0019292e3 * 0.00227 * 0.576 * 2),
            "DDQCDGJET": 1.0,
            "TTG_10_100": 4.334e3,
            "TTG_100_200": 0.44e3,
            "TTG_200": 0.12e3,
            "TT": 730e3,
        }

    luminosities = {
      "2016preVFP": 19.5,
      "2016postVFP": 16.8,
      "2017": 42.07,
      "2018": 59.56,
      "preEE": 7.99,
      "postEE": 26.68,
      "preBPix": 17.96,
      "postBPix": 9.68,
      "2024": 109.95,
      "2025": 110.84,
    }

    lumi = luminosities[era]

    if sample_type == "DDQCDGJET":
      lumi = 1.0

    full_weight = (
      events.weight
      * dict_xsec[sample_type]
      * lumi
      * self.mc_split_weight_scale
    )
    events["rel_xsec_weight"] = full_weight
    events["weight_tot"] = full_weight

    return events

  def get_weights_for_training(self, y_train, rel_w_train, proc_num_train):
    true_class_weights = ak.zeros_like(rel_w_train)
    class_weights_for_training_abs = ak.zeros_like(rel_w_train)
    class_weights_only_positive = ak.zeros_like(rel_w_train)

    for i in range(y_train.shape[1]):
      if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
        cls_bool = y_train[:, i] == 1

        true_class_weights_ = ak.zeros_like(rel_w_train)
        class_weights_for_training_abs_ = ak.zeros_like(rel_w_train)
        class_weights_only_positive_ = ak.zeros_like(rel_w_train)

        for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
          rel_xsec_weight_for_class = rel_w_train * cls_bool * (proc_num_train == proc)
          true_class_weights_ = true_class_weights_ + (
            rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class)
          )

          abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool * (proc_num_train == proc)
          class_weights_for_training_abs_ = class_weights_for_training_abs_ + (
            abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class)
          )

          only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0) * (proc_num_train == proc)
          class_weights_only_positive_ = class_weights_only_positive_ + (
            only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class)
          )

        true_class_weights = true_class_weights + (
          true_class_weights_ / np.sum(true_class_weights_)
        )
        class_weights_for_training_abs = class_weights_for_training_abs + (
          class_weights_for_training_abs_ / np.sum(class_weights_for_training_abs_)
        )
        class_weights_only_positive = class_weights_only_positive + (
          class_weights_only_positive_ / np.sum(class_weights_only_positive_)
        )

      else:
        cls_bool = y_train[:, i] == 1

        rel_xsec_weight_for_class = rel_w_train * cls_bool
        true_class_weights = true_class_weights + (
          rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class)
        )

        abs_rel_xsec_weight_for_class = abs(rel_w_train) * cls_bool
        class_weights_for_training_abs = class_weights_for_training_abs + (
          abs_rel_xsec_weight_for_class / np.sum(abs_rel_xsec_weight_for_class)
        )

        only_positive_rel_xsec_weight_for_class = rel_w_train * cls_bool * (rel_w_train > 0)
        class_weights_only_positive = class_weights_only_positive + (
          only_positive_rel_xsec_weight_for_class / np.sum(only_positive_rel_xsec_weight_for_class)
        )

    if self.class_weight_scale is not None:
      print("\nINFO: Applying class_weight_scale factors:")
      for i in range(y_train.shape[1]):
        class_name = self.class_idx_to_name[i]
        scale = self.class_weight_scale.get(class_name, 1.0)
        print(f"  {class_name}: scale = {scale}")
        cls_bool = y_train[:, i] == 1
        true_class_weights = ak.where(cls_bool, true_class_weights * scale, true_class_weights)
        class_weights_for_training_abs = ak.where(cls_bool, class_weights_for_training_abs * scale, class_weights_for_training_abs)
        class_weights_only_positive = ak.where(cls_bool, class_weights_only_positive * scale, class_weights_only_positive)

    for i in range(y_train.shape[1]):
      print(
        f"(number of events: sum of class_weights_for_training_abs) "
        f"for class number {i + 1} = "
        f"({sum(y_train[:, i])}: {sum(class_weights_for_training_abs[y_train[:, i] == 1])})"
      )
      print(
        f"(number of events: sum of class_weights_only_positive) "
        f"for class number {i + 1} = "
        f"({sum(y_train[:, i])}: {sum(class_weights_only_positive[y_train[:, i] == 1])})"
      )
      print(
        f"(number of events: sum of true_class_weights) "
        f"for class number {i + 1} = "
        f"({sum(y_train[:, i])}: {sum(true_class_weights[y_train[:, i] == 1])})"
      )

      if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
        print("\n")
        for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
          print(
            f"(number of events: sum of class_weights_for_training_abs) "
            f"for class number {i + 1} and process number {proc} = "
            f"({sum(y_train[:, i] * (proc_num_train == proc))}: "
            f"{sum(class_weights_for_training_abs * (y_train[:, i] == 1) * (proc_num_train == proc))})"
          )
          print(
            f"(number of events: sum of class_weights_only_positive) "
            f"for class number {i + 1} and process number {proc} = "
            f"({sum(y_train[:, i] * (proc_num_train == proc))}: "
            f"{sum(class_weights_only_positive * (y_train[:, i] == 1) * (proc_num_train == proc))})"
          )
          print(
            f"(number of events: sum of true_class_weights) "
            f"for class number {i + 1} and process number {proc} = "
            f"({sum(y_train[:, i] * (proc_num_train == proc))}: "
            f"{sum(true_class_weights * (y_train[:, i] == 1) * (proc_num_train == proc))})"
          )
        print("\n")

    return true_class_weights, class_weights_for_training_abs, class_weights_only_positive

  def get_weights_for_val_test(self, y_val, rel_w_val, proc_num_val):
    class_weights_for_val = ak.zeros_like(rel_w_val)

    for i in range(y_val.shape[1]):
      if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
        cls_bool = y_val[:, i] == 1
        class_weights_for_val_ = ak.zeros_like(rel_w_val)

        for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
          rel_xsec_weight_for_class = rel_w_val * cls_bool * (proc_num_val == proc)
          class_weights_for_val_ = class_weights_for_val_ + (
            rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class)
          )

        class_weights_for_val = class_weights_for_val + (
          class_weights_for_val_ / np.sum(class_weights_for_val_)
        )

      else:
        cls_bool = y_val[:, i] == 1
        rel_xsec_weight_for_class = rel_w_val * cls_bool
        class_weights_for_val = class_weights_for_val + (
          rel_xsec_weight_for_class / np.sum(rel_xsec_weight_for_class)
        )

    if self.class_weight_scale is not None:
      print("\nINFO: Applying class_weight_scale factors to validation weights:")
      for i in range(y_val.shape[1]):
        class_name = self.class_idx_to_name[i]
        scale = self.class_weight_scale.get(class_name, 1.0)
        print(f"  {class_name}: scale = {scale}")
        cls_bool = y_val[:, i] == 1
        class_weights_for_val = ak.where(cls_bool, class_weights_for_val * scale, class_weights_for_val)

    for i in range(y_val.shape[1]):
      print(
        f"(number of events: sum of class_weights_for_val) "
        f"for class number {i + 1} = "
        f"({sum(y_val[:, i])}: {sum(class_weights_for_val[y_val[:, i] == 1])})"
      )

      if self.weight_scheme_process[self.class_idx_to_name[i]] == "equal_weight":
        print("\n")
        for proc in range(self.num_process_each_class[self.class_idx_to_name[i]]):
          print(
            f"(number of events: sum of class_weights_for_val) "
            f"for class number {i + 1} and process number {proc} = "
            f"({sum(y_val[:, i] * (proc_num_val == proc))}: "
            f"{sum(class_weights_for_val * (y_val[:, i] == 1) * (proc_num_val == proc))})"
          )
        print("\n")

    return class_weights_for_val

  def train_test_split(self, X, Y, relative_weights, proc_num, train_ratio=0.7, val_ratio=0.3):
    from sklearn.model_selection import train_test_split

    (
      X_train,
      X_test_val,
      y_train,
      y_test_val,
      rel_w_train,
      rel_w_test_val,
      proc_num_train,
      proc_num_val,
    ) = train_test_split(
      X,
      Y,
      relative_weights,
      proc_num,
      train_size=train_ratio,
      shuffle=True,
      random_state=self.random_seed,
    )

    if (train_ratio + val_ratio) == 1.0:
      X_val = X_test_val
      y_val = y_test_val
      rel_w_val = rel_w_test_val
      X_test = None
      y_test = None
      rel_w_test = None
      proc_num_test = None
    else:
      (
        X_val,
        X_test,
        y_val,
        y_test,
        rel_w_val,
        rel_w_test,
        proc_num_val,
        proc_num_test,
      ) = train_test_split(
        X_test_val,
        y_test_val,
        rel_w_test_val,
        proc_num_val,
        train_size=0.5,
        shuffle=True,
        random_state=self.random_seed,
      )

    return (
      X_train,
      X_val,
      X_test,
      y_train,
      y_val,
      y_test,
      rel_w_train,
      rel_w_val,
      rel_w_test,
      proc_num_train,
      proc_num_val,
      proc_num_test,
    )

  def standardize(self, X, mean, std):
    return (X - mean) / std

  def min_max_scale(self, X, min, max):
    return (X - min) / (max - min)

  def corr_with_mgg_mjj(self, events, vars_for_training, out_path):
    corr_matrix = np.zeros([len(vars_for_training), 4])

    for i in range(len(vars_for_training)):
      var = vars_for_training[i]

      mask = (events[var] > -998.0) & (events.mass > -998.0)
      mass = events.mass[mask]
      var_values = events[var][mask]
      corr_matrix[i, 0] = np.corrcoef(mass, var_values)[0, 1]

      mask = (events[var] > -998.0) & (events.nonRes_dijet_mass > -998.0)
      nonRes_dijet_mass = events.nonRes_dijet_mass[mask]
      var_values = events[var][mask]
      corr_matrix[i, 1] = np.corrcoef(nonRes_dijet_mass, var_values)[0, 1]

      mask = (events[var] > -998.0) & (events[f"{self.var_prefix}_dijet_mass"] > -998.0)
      reg_dijet_mass = events[f"{self.var_prefix}_dijet_mass"][mask]
      var_values = events[var][mask]
      corr_matrix[i, 2] = np.corrcoef(reg_dijet_mass, var_values)[0, 1]

      mask = (events[var] > -998.0) & (events[f"{self.var_prefix}_dijet_mass_DNNreg"] > -998.0)
      reg_dijet_mass_DNNreg = events[f"{self.var_prefix}_dijet_mass_DNNreg"][mask]
      var_values = events[var][mask]
      corr_matrix[i, 3] = np.corrcoef(reg_dijet_mass_DNNreg, var_values)[0, 1]

    plt.figure(figsize=(18, len(vars_for_training)))
    plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")

    for i in range(len(vars_for_training)):
      for j in range(4):
        plt.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="b")

    plt.xticks(
      [0, 1, 2, 3],
      [
        "mass",
        "nonRes_dijet_mass",
        f"{self.var_prefix}_dijet_mass",
        f"{self.var_prefix}_dijet_mass_DNNreg",
      ],
      rotation=90,
    )
    plt.yticks(range(len(vars_for_training)), vars_for_training)
    plt.colorbar()
    plt.savefig(out_path, dpi=300)
    plt.clf()
    plt.close()

  def preselection(self, events):
    mass_bool = (events.mass > 100) & (events.mass < 180)
    dijet_mass_bool = (
      (events[f"{self.var_prefix}_dijet_mass_DNNreg"] > 70)
      & (events[f"{self.var_prefix}_dijet_mass_DNNreg"] < 190)
    )

    lead_mvaID_bool = events.lead_mvaID > -0.7
    sublead_mvaID_bool = events.sublead_mvaID > -0.7

    events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    return events

  def preselection_for_pred(self, events):
    mass_bool = (events.mass > 100) & (events.mass < 180)
    lead_mvaID_bool = events.lead_mvaID > -0.7
    sublead_mvaID_bool = events.sublead_mvaID > -0.7

    events = events[mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    return events

  def plot_variables(self, comb_inputs, vars_for_training, plot_path):
    color_list = [
      "#882255",
      "#117733",
      "#332288",
      "#AA4499",
      "#DDCC77",
      "#6699CC",
      "#888888",
      "black",
      "#44AA77",
      "#774411",
      "#DDDDDD",
      "#E69F00",
      "#56B4E9",
      "#009E73",
      "#F0E442",
      "#0072B2",
      "#D55E00",
      "#CC79A7",
      "#000000",
      "#CC6677",
      "#88CCEE",
      "#44AA99",
      "#999999",
      "#FF8A50",
      "#FFB300",
      "#66BB6A",
      "#42A5F5",
      "#AB47BC",
      "#EC407A",
      "#C0CA33",
      "#26A69A",
      "#FB8C00",
      "#795548",
      "#757575",
      "#8E24AA",
    ]

    for var in vars_for_training:
      plt.figure(figsize=(10, 9))
      sample_num = 0
      data_to_plot_dict = {}
      range_list = []

      for sample in self.sample_to_class.keys():
        sample_mask = comb_inputs["sample_type"] == sample
        sample_events_var = comb_inputs[var][sample_mask]

        mask = sample_events_var > -998.0
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
        data_to_plot = data_to_plot_dict[sample]

        if len(data_to_plot) > 0:
          hist_ = np.histogram(
            ak.to_numpy(data_to_plot),
            bins=30,
            range=range_list,
            density=True,
          )
          plt.style.use(hep.style.CMS)
          hep.histplot(
            hist_,
            histtype="step",
            label=sample,
            color=color_list[sample_num],
            linestyle="solid",
            linewidth=1.5,
          )

        sample_num += 1

      plt.xlabel(var)
      plt.ylabel("a.u.")
      plt.legend(ncols=2, fontsize=13, loc="upper right")
      plt.yscale("log")
      plt.tight_layout()
      plt.savefig(f"{plot_path}/{var}_log.png")

      plt.yscale("linear")
      yrange = plt.ylim()
      plt.ylim(0, yrange[1] * 1.3)
      plt.tight_layout()
      plt.savefig(f"{plot_path}/{var}.png")
      plt.clf()
      plt.close()

  def prep_inputs_for_training(self):
    fill_nan = self.fill_nan
    out_path = self._get_split_outpath(self.outpath)
    os.makedirs(out_path, exist_ok=True)

    comb_inputs = []

    for era in self.training_info["samples_info"]["eras"]:
      vars_config = self.load_vars(self.input_var_json)[self.model_type]
      vars_for_training = self.substitute_var_prefix(vars_config["vars"])
      vars_to_load = vars_for_training + self.extra_vars

      for samples in self.sample_to_class.keys():
        print(samples)

        samples_path = self.training_info["samples_info"]["samples_path"]
        parquet_path = self.training_info["samples_info"][era][samples]

        events = self.load_and_process_sample(
          samples_path=samples_path,
          parquet_path=parquet_path,
          samples=samples,
          era=era,
          vars_to_load=vars_to_load,
          preselection_func=self.preselection,
          save_all_columns=False,
          force_split=True,
        )

        if events is None or len(events) == 0:
          print(f"WARNING: No MC events in {samples} after selection for {era}. Skipping.")
          continue

        print(f"INFO: Number of MC events in {samples} after selection for {era}: {len(events)}")
        print(f"INFO: Sum of weight_tot in {samples} after selection for {era}: {sum(events.weight_tot)}")

        for cls in self.classes:
          events[cls] = ak.zeros_like(events.eta)

        events[self.sample_to_class[samples]] = ak.ones_like(events.pt)
        events["sample_type"] = samples
        events["process_number"] = self.process_numbers[samples]

        comb_inputs.append(events)

        if self.training_info.get("make_correlation_plots", False):
          os.makedirs(f"{out_path}/correlation_matrix/", exist_ok=True)
          corr_out_path = f"{out_path}/correlation_matrix/{samples}_{era}.pdf"
          self.corr_with_mgg_mjj(events, vars_for_training, corr_out_path)

    if len(comb_inputs) == 0:
      raise RuntimeError("No events survived selection. Cannot prepare training inputs.")

    print("INFO: Combining all the samples")
    comb_inputs = ak.concatenate(comb_inputs, axis=0)

    if self.training_info.get("make_variable_plots", False):
      plot_path = f"{out_path}/var_plots/"
      os.makedirs(plot_path, exist_ok=True)
      self.plot_variables(comb_inputs, vars_for_training, plot_path)

    for cls in self.classes:
      print("\n", f"INFO: Number of events in {cls}: {ak.sum(comb_inputs[cls])}")

    print("INFO: Converting training variables to numpy arrays")
    X_list = []

    for var in vars_for_training:
      var_data = ak.to_numpy(ak.fill_none(comb_inputs[var], -999.0))
      X_list.append(var_data)

    X = np.column_stack(X_list).astype(np.float32)

    print("INFO: Converting class labels to numpy arrays")
    Y_list = []

    for cls in self.classes:
      cls_data = ak.to_numpy(ak.fill_none(comb_inputs[cls], 0))
      Y_list.append(cls_data)

    Y = np.column_stack(Y_list).astype(np.int8)

    relative_weights = ak.to_numpy(
      ak.fill_none(comb_inputs["rel_xsec_weight"], np.nan)
    ).astype(np.float32)

    process_number = ak.to_numpy(
      ak.fill_none(comb_inputs["process_number"], -1)
    ).astype(np.int8)

    del X_list, Y_list, comb_inputs
    gc.collect()

    mask = X < -998.0
    X[mask] = np.nan

    training_fraction = self.training_info.get(
      "training_fraction",
      self.training_info.get("train_fraction", 1.0),
    )

    if training_fraction < 1.0:
      rng = np.random.default_rng(self.random_seed)
      n_total = len(X)
      n_keep = int(n_total * training_fraction)
      idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))

      X = X[idx]
      Y = Y[idx]
      relative_weights = relative_weights[idx]
      process_number = process_number[idx]

      print(f"INFO: training_fraction={training_fraction}: keeping {n_keep}/{n_total} events")

    (
      X_train,
      X_val,
      X_test,
      y_train,
      y_val,
      y_test,
      rel_w_train,
      rel_w_val,
      rel_w_test,
      proc_num_train,
      proc_num_val,
      proc_num_test,
    ) = self.train_test_split(X, Y, relative_weights, process_number)

    del X, Y, relative_weights, process_number, mask
    gc.collect()

    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)

    X_train = self.standardize(X_train, mean, std)
    X_val = self.standardize(X_val, mean, std)

    X_train = np.nan_to_num(X_train, nan=fill_nan)
    X_val = np.nan_to_num(X_val, nan=fill_nan)

    (
      true_class_weights,
      class_weights_for_training_abs,
      class_weights_only_positive,
    ) = self.get_weights_for_training(y_train, rel_w_train, proc_num_train)

    class_weights_for_val = self.get_weights_for_val_test(
      y_val,
      rel_w_val,
      proc_num_val,
    )

    if X_test is not None:
      X_test = self.standardize(X_test, mean, std)
      X_test = np.nan_to_num(X_test, nan=fill_nan)
      class_weights_for_test = self.get_weights_for_val_test(
        y_test,
        rel_w_test,
        proc_num_test,
      )

    print("\nINFO: saving inputs for mlp")

    with open(f"{out_path}/input_vars.txt", "w") as f:
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

    mean_std_dict = {
      "mean": mean,
      "std_dev": std,
    }

    with open(f"{out_path}/mean_std_dict.pkl", "wb") as f:
      pickle.dump(mean_std_dict, f)

    return 0

  def prep_inputs_for_prediction_sim(self):
    fill_nan = self.fill_nan
    training_info = self.training_info
    inputs_path = self._get_split_outpath(self.outpath)
    out_path = f"{inputs_path}/individual_samples/"
    os.makedirs(out_path, exist_ok=True)

    samples_path = training_info["samples_info"]["samples_path"]

    scale_file = f"{inputs_path}/mean_std_dict.pkl"
    with open(scale_file, "rb") as f:
      mean_std_dict = pickle.load(f)

    mean = mean_std_dict["mean"]
    std = mean_std_dict["std_dev"]

    vh_component_names = {
      "WmHtoGG": "WmHtoGG_M_125",
      "WpHtoGG": "WpHtoGG_M_125",
      "ZHtoGG": "ZHtoGG_M_125",
    }

    for era in training_info["samples_info"]["eras"]:
      with open(f"{inputs_path}/input_vars.txt", "r") as f:
        vars_for_training = json.load(f)

      vars_to_load = (
        vars_for_training
        + self.extra_vars
        + self.vars_for_boosted
        + ["lead_genPartFlav", "sublead_genPartFlav", "weight_tot"]
      )

      for samples in training_info["samples_info"][era].keys():
        parquet_path = training_info["samples_info"][era][samples]

        if isinstance(parquet_path, list):
          print(f"INFO: Merging {len(parquet_path)} files for {samples} in {era}")
          file_xsec_pairs = []

          for path in parquet_path:
            component_name = None
            for key in vh_component_names:
              if key in path:
                component_name = vh_component_names[key]
                break

            file_xsec_pairs.append((f"{samples_path}/{path}", component_name))
        else:
          file_xsec_pairs = [(f"{samples_path}/{parquet_path}", samples)]

        full_path_to_save = f"{out_path}/{era}/{samples}/"
        os.makedirs(full_path_to_save, exist_ok=True)

        is_composite = len(file_xsec_pairs) > 1
        X_chunks = []
        w_chunks = []
        parquet_writer = None
        parquet_chunks = []

        for file_path, xsec_name in file_xsec_pairs:
          for batch in self._iter_file_batched(
            file_path=file_path,
            vars_to_load=vars_to_load,
            save_all_columns=self.save_all_columns_sim_nominal,
            era=era,
            preselection_func=self.preselection_for_pred,
            xsec_sample_name=xsec_name,
            apply_xsec_weights=True,
            force_split=True,
          ):
            X_chunks.append(np.column_stack([
              ak.to_numpy(ak.fill_none(batch[var], -999.0))
              for var in vars_for_training
            ]).astype(np.float32))

            w_chunks.append(
              ak.to_numpy(
                ak.fill_none(batch["rel_xsec_weight"], np.nan)
              ).astype(np.float32)
            )

            arrow_table = pa.table({
              field: ak.to_arrow(batch[field])
              for field in batch.fields
            })

            if is_composite:
              parquet_chunks.append(arrow_table)
            else:
              if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                  f"{full_path_to_save}/events.parquet",
                  arrow_table.schema,
                )
              parquet_writer.write_table(arrow_table)

            del batch, arrow_table

        if parquet_writer is not None:
          parquet_writer.close()

        if parquet_chunks:
          pq.write_table(
            pa.concat_tables(parquet_chunks, promote_options="default"),
            f"{full_path_to_save}/events.parquet",
          )
          del parquet_chunks

        if not X_chunks:
          print(f"WARNING: No events survived selection for {samples} in {era}. Skipping.")
          continue

        X = np.concatenate(X_chunks)
        del X_chunks

        relative_weights = np.concatenate(w_chunks)
        del w_chunks

        print(f"INFO: Number of events in {samples} for {era}: {len(X)}")

        mask = X < -998.0
        X[mask] = np.nan
        X = self.standardize(X, mean, std)
        X = np.nan_to_num(X, nan=fill_nan)

        print("INFO: saving inputs for mlp")
        np.save(f"{full_path_to_save}/X", X)
        np.save(f"{full_path_to_save}/rel_w", relative_weights)

        del X, relative_weights, mask
        gc.collect()

    mean_std_dict = {
      "mean": mean,
      "std_dev": std,
    }

    with open(f"{out_path}/mean_std_dict.pkl", "wb") as f:
      pickle.dump(mean_std_dict, f)

    return 0

  def prep_inputs_for_prediction_sim_sys(self):
    fill_nan = self.fill_nan
    training_info = self.training_info
    inputs_path = self._get_split_outpath(self.outpath)
    out_path = f"{inputs_path}/individual_samples/"
    os.makedirs(out_path, exist_ok=True)

    samples_path = training_info["samples_info"]["samples_path"]

    scale_file = f"{inputs_path}/mean_std_dict.pkl"
    with open(scale_file, "rb") as f:
      mean_std_dict = pickle.load(f)

    mean = mean_std_dict["mean"]
    std = mean_std_dict["std_dev"]

    vh_component_names = {
      "WmHtoGG": "WmHtoGG_M_125",
      "WpHtoGG": "WpHtoGG_M_125",
      "ZHtoGG": "ZHtoGG_M_125",
    }

    for era in training_info["samples_info"]["eras"]:
      for samples in training_info["samples_info"][era].keys():
        with open(f"{inputs_path}/input_vars.txt", "r") as f:
          vars_for_training = json.load(f)

        vars_to_load = (
          vars_for_training
          + self.extra_vars
          + self.vars_for_boosted
          + ["lead_genPartFlav", "sublead_genPartFlav", "weight_tot"]
        )

        for sys in training_info["systematics"]:
          if samples in [
            "GGJets",
            "DDQCDGJET",
            "TTG_10_100",
            "TTG_100_200",
            "TTG_200",
            "TT",
            "TTGG",
          ]:
            continue

          parquet_path = training_info["samples_info"][era][samples]

          if isinstance(parquet_path, list):
            file_xsec_pairs = []

            for path in parquet_path:
              sys_path = path.replace("nominal", sys)

              if not os.path.exists(f"{samples_path}/{sys_path}"):
                print(
                  f"WARNING: {samples} for {era} for {sys} does not exist. "
                  f"Skipping.: {samples_path}/{sys_path}"
                )
                continue

              component_name = None
              for key in vh_component_names:
                if key in path:
                  component_name = vh_component_names[key]
                  break

              file_xsec_pairs.append((f"{samples_path}/{sys_path}", component_name))

          else:
            sys_path = parquet_path.replace("nominal", sys)

            if not os.path.exists(f"{samples_path}/{sys_path}"):
              print(
                f"WARNING: {samples} for {era} for {sys} does not exist. "
                f"Skipping.: {samples_path}/{sys_path}"
              )
              continue

            file_xsec_pairs = [(f"{samples_path}/{sys_path}", samples)]

          if not file_xsec_pairs:
            continue

          full_path_to_save = f"{out_path}/{era}/{samples}/{sys}/"
          os.makedirs(full_path_to_save, exist_ok=True)

          is_composite = len(file_xsec_pairs) > 1
          X_chunks = []
          w_chunks = []
          parquet_writer = None
          parquet_chunks = []

          for file_path, xsec_name in file_xsec_pairs:
            for batch in self._iter_file_batched(
              file_path=file_path,
              vars_to_load=vars_to_load,
              save_all_columns=self.save_all_columns_sim_systematics,
              era=era,
              preselection_func=self.preselection_for_pred,
              xsec_sample_name=xsec_name,
              apply_xsec_weights=True,
              force_split=True,
            ):
              X_chunks.append(np.column_stack([
                ak.to_numpy(ak.fill_none(batch[var], -999.0))
                for var in vars_for_training
              ]).astype(np.float32))

              w_chunks.append(
                ak.to_numpy(
                  ak.fill_none(batch["rel_xsec_weight"], np.nan)
                ).astype(np.float32)
              )

              arrow_table = pa.table({
                field: ak.to_arrow(batch[field])
                for field in batch.fields
              })

              if is_composite:
                parquet_chunks.append(arrow_table)
              else:
                if parquet_writer is None:
                  parquet_writer = pq.ParquetWriter(
                    f"{full_path_to_save}/events.parquet",
                    arrow_table.schema,
                  )
                parquet_writer.write_table(arrow_table)

              del batch, arrow_table

          if parquet_writer is not None:
            parquet_writer.close()

          if parquet_chunks:
            pq.write_table(
              pa.concat_tables(parquet_chunks, promote_options="default"),
              f"{full_path_to_save}/events.parquet",
            )
            del parquet_chunks

          if not X_chunks:
            print(f"WARNING: No events survived selection for {samples} in {era} for {sys}. Skipping.")
            continue

          X = np.concatenate(X_chunks)
          del X_chunks

          relative_weights = np.concatenate(w_chunks)
          del w_chunks

          print(f"INFO: Number of events in {samples} for {era} for {sys}: {len(X)}")

          mask = X < -998.0
          X[mask] = np.nan
          X = self.standardize(X, mean, std)
          X = np.nan_to_num(X, nan=fill_nan)

          np.save(f"{full_path_to_save}/X", X)
          np.save(f"{full_path_to_save}/rel_w", relative_weights)

          del X, relative_weights, mask
          gc.collect()

    mean_std_dict = {
      "mean": mean,
      "std_dev": std,
    }

    with open(f"{out_path}/mean_std_dict.pkl", "wb") as f:
      pickle.dump(mean_std_dict, f)

    return 0

  def prep_inputs_for_prediction_data(self):
    fill_nan = self.fill_nan
    training_info = self.training_info
    inputs_path = self._get_split_outpath(self.outpath)
    out_path = f"{inputs_path}/individual_samples_data/"
    os.makedirs(out_path, exist_ok=True)

    samples_path = training_info["samples_info"]["samples_path"]
    datas = training_info["samples_info"]["data"]

    scale_file = f"{inputs_path}/mean_std_dict.pkl"
    with open(scale_file, "rb") as f:
      mean_std_dict = pickle.load(f)

    mean = mean_std_dict["mean"]
    std = mean_std_dict["std_dev"]

    sample_to_era = {
      "2016preVFP": "2016preVFP",
      "2016postVFP": "2016postVFP",
      "2017": "2017",
      "2018": "2018",
      "2022preEE": "preEE",
      "2022postEE": "postEE",
      "2023preBPix": "preBPix",
      "2023postBPix": "postBPix",
      "2022_EraE": "postEE",
      "2022_EraF": "postEE",
      "2022_EraG": "postEE",
      "2022_EraC": "preEE",
      "2022_EraD": "preEE",
      "2023_EraCv1to3": "preBPix",
      "2023_EraCv4": "preBPix",
      "2023_EraC": "preBPix",
      "2023_EraD": "postBPix",
      "2024": "2024",
      "2025": "2025",
    }

    # With split_data disabled, all real data goes to Final_split.
    force_split_data = self.training_info.get("split_data", False)

    for data in datas:
      with open(f"{inputs_path}/input_vars.txt", "r") as f:
        vars_for_training = json.load(f)

      vars_to_load = vars_for_training + self.extra_vars + self.vars_for_boosted

      era = sample_to_era[data]
      parquet_path = datas[data]

      if isinstance(parquet_path, list):
        file_paths = [f"{samples_path}/{path}" for path in parquet_path]
      else:
        file_paths = [f"{samples_path}/{parquet_path}"]

      full_path_to_save = f"{out_path}/{data}/"
      os.makedirs(full_path_to_save, exist_ok=True)

      X_chunks = []
      parquet_writer = None

      for file_path in file_paths:
        for batch in self._iter_file_batched(
          file_path=file_path,
          vars_to_load=vars_to_load,
          save_all_columns=self.save_all_columns_data,
          era=era,
          preselection_func=self.preselection_for_pred,
          xsec_sample_name=data,
          apply_xsec_weights=False,
          force_split=force_split_data,
        ):
          X_chunks.append(np.column_stack([
            ak.to_numpy(ak.fill_none(batch[var], -999.0))
            for var in vars_for_training
          ]).astype(np.float32))

          arrow_table = pa.table({
            field: ak.to_arrow(batch[field])
            for field in batch.fields
          })

          if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
              f"{full_path_to_save}/events.parquet",
              arrow_table.schema,
            )

          parquet_writer.write_table(arrow_table)

          del batch, arrow_table

      if parquet_writer is not None:
        parquet_writer.close()

      if not X_chunks:
        print(f"WARNING: No data events survived selection for {data}. Skipping.")
        continue

      X = np.concatenate(X_chunks)
      del X_chunks

      print(f"INFO: saving inputs for {data} ({len(X)} events)")

      mask = X < -998.0
      X[mask] = np.nan
      X = self.standardize(X, mean, std)
      X = np.nan_to_num(X, nan=fill_nan)

      np.save(f"{full_path_to_save}/X", X)

      del X, mask
      gc.collect()

    mean_std_dict = {
      "mean": mean,
      "std_dev": std,
    }

    with open(f"{out_path}/mean_std_dict.pkl", "wb") as f:
      pickle.dump(mean_std_dict, f)

    return 0
