# VERSION: clean process-yield output; debug summaries retained
import argparse
import json
import os
import contextlib
import io

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

plt.style.use(hep.style.CMS)


class YieldTableMaker:
  def __init__(
    self,
    base_path,
    opt,
    output_dir=None,
    signal_samples=None,
    vbfhh_samples=None,
    samples_list=None,
    interp_samples=None,
    apply_boosted_veto=False,
    mgg_sr_low=120.0,
    mgg_sr_high=130.0,
    cms_data=False,
    cms_label="Private Work",
    cms_lumi=419.5,
    cms_com="13 / 13.6",
  ):
    self.base_path = base_path
    self.opt = opt
    self.best_cut_json = os.path.join(base_path, opt, "best_cut_params.json")
    self.vbfhh_info_json = os.path.join(base_path, opt, "vbfhh_sr_info.json")
    self.output_dir = output_dir if output_dir is not None else os.path.join(base_path, opt)

    if signal_samples is None:
      raise ValueError(
        "signal_samples must be provided as bracket-enclosed list, "
        "e.g. '[GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00]'"
      )

    self.signal_samples = self._parse_bracket_list(signal_samples, "signal_samples")

    if vbfhh_samples is not None:
      self.vbfhh_samples = self._parse_bracket_list(vbfhh_samples, "vbfhh_samples")
    else:
      self.vbfhh_samples = []

    self.ggHH_signal_samples = [s for s in self.signal_samples if s not in self.vbfhh_samples]

    self.samples_list = samples_list if samples_list is not None else [
      "VBFHToGG_M_125",
      "VHtoGG_M_125",
      "ttHtoGG_M_125",
      "BBHto2G_M_125",
      "GluGluHToGG_M_125",
      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
      "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
      "TTGG",
      "GGJets",
      "DDQCDGJET",
      "TTG_100_200",
      "TTG_200",
    ]

    self.interp_samples = set(interp_samples) if interp_samples is not None else {
      "TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"
    }

    self.plot_samples = {"GGJets", "TTGG"}

    self.apply_preselection = True
    self.apply_boosted_veto = apply_boosted_veto

    self.dijet_mass_key = "nonResReg_vbfpair_dijet_mass_DNNreg"

    self.mass_sr = self._validate_and_set_mgg_sr(mgg_sr_low, mgg_sr_high)
    self.mass_left_sb = (100.0, self.mass_sr[0])
    self.mass_right_sb = (self.mass_sr[1], 180.0)

    self.scores_all = None
    self.mass_all = None
    self.dijet_mass_all = None
    self.weights_all = None
    self.labels_all = None
    self.samples_all = None
    self.eras_all = None

    self.data_scores_all = None
    self.data_mass_all = None
    self.data_dijet_mass_all = None
    self.data_weights_all = None
    self.data_samples_all = None
    self.data_eras_all = None

    # Debug-only containers. These do not change the analysis logic.
    self.debug_data_files_seen = []
    self.debug_data_files_loaded = []
    self.debug_data_files_missing_scores = []
    self.debug_data_files_mismatched_scores = []
    self.debug_data_files_failed = []
    self.debug_data_era_dirs_on_disk = []
    self.debug_data_era_dirs_configured = []

    # MC/process debug containers. These are only for bookkeeping and do not
    # change the analysis logic or any event selection. They help diagnose
    # whether a process such as TTGG is missing in one or more eras.
    self.debug_mc_era_dirs_on_disk = []
    self.debug_mc_era_dirs_configured = []
    self.debug_mc_process_records = []
    self.debug_mc_files_seen = []
    self.debug_mc_files_loaded = []
    self.debug_mc_files_missing = []
    self.debug_mc_files_failed = []

    self.cms_data = cms_data
    self.cms_label = cms_label
    self.cms_lumi = cms_lumi
    self.cms_com = cms_com

    self.region_colors = {
      "Preselection": "black",
      "vbfhh_cat1": "#E69F00",
      "cat1": "#D55E00",
      "cat2": "#0072B2",
      "cat3": "#009E73",
    }

    self.sample_row_colors = {
      "Actual Data": "black",
      "All (GGJets + TTGG)": "black",
      "GGJets": "#88CCEE",
      "TTGG": "#CC6677",
    }

    self._set_cms_style()

  def _set_cms_style(self):
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({
      "figure.dpi": 120,
      "savefig.dpi": 300,
      "axes.labelsize": 22,
      "axes.titlesize": 18,
      "legend.fontsize": 13,
      "xtick.labelsize": 16,
      "ytick.labelsize": 16,
      "axes.linewidth": 1.6,
      "xtick.major.size": 9,
      "ytick.major.size": 9,
      "xtick.minor.size": 4.5,
      "ytick.minor.size": 4.5,
      "xtick.direction": "in",
      "ytick.direction": "in",
      "xtick.top": True,
      "ytick.right": True,
      "legend.frameon": False,
    })

  def _apply_cms_axes(
    self,
    ax,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    logy=False,
    add_label=False,
  ):
    if xlabel is not None:
      ax.set_xlabel(xlabel)
    if ylabel is not None:
      ax.set_ylabel(ylabel)
    if xlim is not None:
      ax.set_xlim(*xlim)
    if ylim is not None:
      ax.set_ylim(*ylim)
    if logy:
      ax.set_yscale("log")

    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    if add_label:
      hep.cms.label(
        ax=ax,
        data=self.cms_data,
        label=self.cms_label,
        lumi=self.cms_lumi,
        com=self.cms_com,
        loc=0,
        fontsize=18,
      )

  def _add_panel_text(self, ax, text, x=0.04, y=0.93, fontsize=14):
    ax.text(
      x,
      y,
      text,
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=fontsize,
    )

  def _draw_norm_hist_with_err(
    self,
    ax,
    values,
    weights,
    bins,
    label,
    color=None,
    linewidth=2.0,
    markersize=3,
  ):
    centers, hist, err = self._hist_norm_with_err(values, weights, bins)

    line, = ax.step(
      centers,
      hist,
      where="mid",
      label=label,
      linewidth=linewidth,
      color=color,
    )
    draw_color = color if color is not None else line.get_color()

    ax.errorbar(
      centers,
      hist,
      yerr=err,
      fmt="o",
      markersize=markersize,
      capsize=2,
      linestyle="none",
      color=draw_color,
      ecolor=draw_color,
    )

  def _parse_bracket_list(self, raw_string, name):
    raw = raw_string.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
      raise ValueError(f"{name} must be bracket-enclosed, e.g. '[A,B,C]'")
    inner = raw[1:-1]
    parsed = [s.strip() for s in inner.split(",") if s.strip()]
    if not parsed:
      raise ValueError(f"{name} is empty inside brackets.")
    return parsed

  def _validate_and_set_mgg_sr(self, mgg_sr_low, mgg_sr_high):
    sr_low = float(mgg_sr_low)
    sr_high = float(mgg_sr_high)

    if sr_low <= 100.0 or sr_high >= 180.0:
      raise ValueError(
        "mgg SR boundaries must lie inside the full mgg range: "
        "100 < mgg_sr_low < mgg_sr_high < 180."
      )

    if sr_low >= sr_high:
      raise ValueError("mgg_sr_low must be smaller than mgg_sr_high.")

    return (sr_low, sr_high)

  def _discover_subdirs(self, base_dir, context_label):
    """Return sorted immediate subdirectories under base_dir.

    This is used to auto-detect available eras instead of relying on a
    hard-coded era list. It does not inspect or modify event content.
    """
    if not os.path.exists(base_dir):
      print(f"[{context_label}] Directory does not exist: {base_dir}")
      return []

    subdirs = [
      d for d in sorted(os.listdir(base_dir))
      if os.path.isdir(os.path.join(base_dir, d))
    ]

    if len(subdirs) == 0:
      print(f"[{context_label}] No era directories found under: {base_dir}")
    else:
      print(f"[{context_label}] Auto-detected eras under {base_dir}:")
      for era in subdirs:
        print(f"  - {era}")

    return subdirs

  def _resolve_data_base_dir(self):
    """Return the intended actual-data directory.

    Only individual_samples_data is supported. No fallback typo directory is used.
                                                                           
                                                 
    """
                  
    return os.path.join(self.base_path, "individual_samples_data")
                                                              
     

                           
                              
                                                    
                
                                                                 
                                                                            
                                                     
           
                   

                                                                                   
                        

  def _is_hh_signal_sample_array(self, samples):
    samples_str = np.asarray(samples).astype(str)
    is_gghh = np.char.startswith(samples_str, "GluGlutoHHto2B2G_kl_")
    is_vbfhh = np.char.startswith(samples_str, "VBFHH_CV_")
    return is_gghh | is_vbfhh

  def _make_signal_and_background_masks(self, samples, signal_samples_this_region):
    signal_mask = np.isin(samples, signal_samples_this_region)
    any_hh_mask = self._is_hh_signal_sample_array(samples)

    # Background is strictly non-HH. This excludes all ggHH and VBFHH samples,
    # including HH samples that are not the target signal for the current region.
    background_mask = ~any_hh_mask

    return signal_mask, background_mask

  def preselection(self, events, scores):
    mass_bool = (events.mass > 100) & (events.mass < 180)
    dijet_mass_bool = (events[self.dijet_mass_key] > 80) & (
      events[self.dijet_mass_key] < 190
    )
    lead_mvaID_bool = events.lead_mvaID > -0.7
    sublead_mvaID_bool = events.sublead_mvaID > -0.7

    mask = mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool
    if self.apply_boosted_veto:
      mask = mask & (events.is_boosted == False)

    events = events[mask]
    scores = scores[mask]
    return events, scores

  def load_samples(self):
    print("[load_samples] Start loading MC samples...")
    data = {
      k: []
      for k in (
        "score",
        "diphoton_mass",
        "dijet_mass",
        "weights",
        "labels",
        "sample",
        "era",
      )
    }

    mc_base = os.path.join(self.base_path, "individual_samples")
    eras = self._discover_subdirs(mc_base, "load_samples")
    self.debug_mc_era_dirs_on_disk = list(eras)
    self.debug_mc_era_dirs_configured = list(eras)

    if len(eras) == 0:
      raise RuntimeError(f"[load_samples] No MC era directories found under: {mc_base}")

    for era in eras:
      print(f"[load_samples] Era: {era}")
      for sample in self.samples_list:
        samp_dir = os.path.join(self.base_path, "individual_samples", era, sample)
        y_file = os.path.join(samp_dir, "y.npy")
        evt_file = os.path.join(
          samp_dir,
          "events" + ("_boostedCat" if self.apply_boosted_veto else "") + ".parquet"
        )

        record = {
          "Era": era,
          "Process": sample,
          "SampleDir": samp_dir,
          "YFile": y_file,
          "EventFile": evt_file,
          "HasY": os.path.exists(y_file),
          "HasEvents": os.path.exists(evt_file),
          "Status": "not_checked",
          "N_before": 0,
          "N_after_preselection": 0,
          "N_after_prompt": 0,
          "SumW_after_prompt": 0.0,
        }
        self.debug_mc_process_records.append(record)

        if not (record["HasY"] and record["HasEvents"]):
          record["Status"] = "missing"
          self.debug_mc_files_missing.append(
            f"{era} / {sample} | has_y={record['HasY']} | has_events={record['HasEvents']} | {samp_dir}"
          )
          continue

        self.debug_mc_files_seen.append(evt_file)

        try:
          y = np.load(y_file)
          columns = [
            "mass",
            self.dijet_mass_key,
            "lead_genPartFlav",
            "sublead_genPartFlav",
            "weight_tot",
            "lead_mvaID",
            "sublead_mvaID",
          ]
          if self.apply_boosted_veto:
            columns.append("is_boosted")

          print(f"  [load_samples] Reading parquet: {evt_file}", flush=True)
          events = ak.from_parquet(evt_file, columns=columns)
          n_before = len(y)
          record["N_before"] = int(n_before)

          if len(y) != len(events):
            print(
              f"  [load_samples] WARNING: score/event length mismatch for {era} / {sample}. "
              f"len(y)={len(y)}, len(events)={len(events)}"
            )

          if self.apply_preselection:
            events, y = self.preselection(events, y)

          n_after = len(y)
          record["N_after_preselection"] = int(n_after)
          print(f"  [load_samples] {sample}: {n_before} -> {n_after} after preselection")

        except Exception as exc:
          record["Status"] = "failed"
          self.debug_mc_files_failed.append(f"{era} / {sample} | {samp_dir} | {exc}")
          print(f"[load_samples] ERROR while reading {samp_dir}: {exc}")
          continue

        if sample.startswith("TTG_") or sample in {"TT", "TTGG"}:
          sel = (events["lead_genPartFlav"] == 1) & (events["sublead_genPartFlav"] == 1)
          events = events[sel]
          y = y[sel]
          print(f"  [load_samples] {sample}: {len(y)} after prompt-photon selection")

        record["N_after_prompt"] = int(len(y))
        if len(y) > 0:
          record["SumW_after_prompt"] = float(np.asarray(events["weight_tot"]).sum())

        if len(y) == 0:
          record["Status"] = "empty_after_selection"
          continue

        record["Status"] = "loaded"
        self.debug_mc_files_loaded.append(evt_file)

        data["score"].append(y)
        data["diphoton_mass"].append(np.asarray(events["mass"]))
        data["dijet_mass"].append(np.asarray(events[self.dijet_mass_key]))
        data["weights"].append(np.asarray(events["weight_tot"]))

        if sample in self.signal_samples:
          label_val = 1
        elif "GluGlutoHHto2B2G" in sample:
          label_val = -1
        else:
          label_val = 0

        data["labels"].append(np.full(len(y), label_val, dtype=int))
        data["sample"].append(np.repeat(sample, len(y)))
        data["era"].append(np.repeat(era, len(y)))

    if not data["score"]:
      raise RuntimeError("[load_samples] No MC events found in any input sample.")

    print("[load_samples] Concatenating MC arrays...")
    for key in data:
      data[key] = np.concatenate(data[key], axis=0)

    self.scores_all = data["score"]
    self.mass_all = data["diphoton_mass"]
    self.dijet_mass_all = data["dijet_mass"]
    self.weights_all = data["weights"]
    self.labels_all = data["labels"]
    self.samples_all = data["sample"]
    self.eras_all = data["era"]

    print(f"[load_samples] Loaded {len(self.mass_all)} MC events total")
    self.print_mc_debug_summary()

  def _find_data_parquet_files(self, era_dir):
    parquet_files = []

    for root, _, files in os.walk(era_dir):
      for fname in files:
        if not fname.endswith(".parquet"):
          continue
        if self.apply_boosted_veto and "boostedCat" not in fname:
          continue
        if (not self.apply_boosted_veto) and "boostedCat" in fname:
          continue
        parquet_files.append(os.path.join(root, fname))

    parquet_files = sorted(parquet_files)
    return parquet_files

  def _find_matching_y_file_for_data(self, evt_file):
    evt_dir = os.path.dirname(evt_file)
    direct_y = os.path.join(evt_dir, "y.npy")

    if os.path.exists(direct_y):
      return direct_y

    base_name = os.path.basename(evt_file).replace(".parquet", "")
    candidates = [
      os.path.join(evt_dir, f"{base_name}.npy"),
      os.path.join(evt_dir, f"{base_name}_y.npy"),
      os.path.join(evt_dir, "scores.npy"),
      os.path.join(evt_dir, "predictions.npy"),
    ]

    for cand in candidates:
      if os.path.exists(cand):
        return cand

    return None

  def _normalize_data_era_label(self, era):
    if era in {"2022_EraC", "2022_EraD"}:
      return "preEE"

    if era in {"2022_EraE", "2022_EraF", "2022_EraG"}:
      return "postEE"

    if era == "2023_EraC":
      return "preBPix"

    if era == "2023_EraD":
      return "postBPix"

    return era

  def print_mc_debug_summary(self):
    print("\n" + "=" * 100)
    print("[MC PROCESS DEBUG SUMMARY]")
    print("=" * 100)

    print(f"[MC DEBUG] MC era directories found: {len(self.debug_mc_era_dirs_on_disk)}")
    if len(self.debug_mc_era_dirs_on_disk) > 0:
      for era in self.debug_mc_era_dirs_on_disk:
        print(f"  - {era}")

    print(f"\n[MC DEBUG] configured processes: {len(self.samples_list)}")
    for sample in self.samples_list:
      print(f"  - {sample}")

    n_records = len(self.debug_mc_process_records)
    n_present = sum(1 for r in self.debug_mc_process_records if r.get("HasY") and r.get("HasEvents"))
    n_loaded = sum(1 for r in self.debug_mc_process_records if r.get("Status") == "loaded")
    n_empty = sum(1 for r in self.debug_mc_process_records if r.get("Status") == "empty_after_selection")
    n_missing = sum(1 for r in self.debug_mc_process_records if r.get("Status") == "missing")
    n_failed = sum(1 for r in self.debug_mc_process_records if r.get("Status") == "failed")

    print("\n[MC DEBUG] process-era file status:")
    print(f"  total process-era checks: {n_records}")
    print(f"  files present:             {n_present}")
    print(f"  loaded with events:        {n_loaded}")
    print(f"  empty after selection:     {n_empty}")
    print(f"  missing y/events:          {n_missing}")
    print(f"  failed while reading:      {n_failed}")

    if len(self.debug_mc_files_missing) > 0:
      print("\n[MC DEBUG WARNING] Missing MC process-era inputs:")
      for item in self.debug_mc_files_missing:
        print(f"  - {item}")

    if len(self.debug_mc_files_failed) > 0:
      print("\n[MC DEBUG WARNING] MC files that failed while reading:")
      for item in self.debug_mc_files_failed:
        print(f"  - {item}")

    print("\n[MC DEBUG] Per-process loading summary:")
    header = (
      f"{'Process':<45} {'present':>7} {'loaded':>7} {'empty':>7} "
      f"{'missing':>8} {'failed':>7} {'N_before':>12} {'N_presel':>12} "
      f"{'N_prompt':>12} {'sumW_prompt':>14}"
    )
    print(header)
    print("-" * len(header))

    expected_eras = list(self.debug_mc_era_dirs_configured)

    for sample in self.samples_list:
      recs = [r for r in self.debug_mc_process_records if r.get("Process") == sample]
      present_eras = sorted([r["Era"] for r in recs if r.get("HasY") and r.get("HasEvents")])
      loaded_eras = sorted([r["Era"] for r in recs if r.get("Status") == "loaded"])
      empty_eras = sorted([r["Era"] for r in recs if r.get("Status") == "empty_after_selection"])
      missing_eras = sorted([r["Era"] for r in recs if r.get("Status") == "missing"])
      failed_eras = sorted([r["Era"] for r in recs if r.get("Status") == "failed"])

      n_before = sum(int(r.get("N_before", 0)) for r in recs)
      n_presel = sum(int(r.get("N_after_preselection", 0)) for r in recs)
      n_prompt = sum(int(r.get("N_after_prompt", 0)) for r in recs)
      sumw_prompt = sum(float(r.get("SumW_after_prompt", 0.0)) for r in recs)

      print(
        f"{sample:<45} {len(present_eras):>7} {len(loaded_eras):>7} {len(empty_eras):>7} "
        f"{len(missing_eras):>8} {len(failed_eras):>7} {n_before:>12} {n_presel:>12} "
        f"{n_prompt:>12} {sumw_prompt:>14.6f}"
      )

      if sample == "TTGG":
        print("\n[MC DEBUG TTGG DETAIL]")
        print(f"  expected eras: {expected_eras}")
        print(f"  present eras:  {present_eras}")
        print(f"  loaded eras:   {loaded_eras}")
        print(f"  empty eras:    {empty_eras}")
        print(f"  missing eras:  {missing_eras}")
        print(f"  failed eras:   {failed_eras}")
        if len(missing_eras) > 0 or len(failed_eras) > 0 or len(empty_eras) > 0:
          print("  [MC DEBUG WARNING] TTGG is not fully loaded in all detected eras.")
        else:
          print("  [MC DEBUG] TTGG appears loaded in all detected eras.")

    if self.samples_all is None:
      print("\n[MC DEBUG WARNING] self.samples_all is None. No MC arrays were concatenated.")
    else:
      print("\n[MC DEBUG] Final concatenated MC events by process:")
      unique_samples, counts = np.unique(self.samples_all, return_counts=True)
      for sample in self.samples_list:
        count = int(counts[unique_samples == sample][0]) if sample in unique_samples else 0
        sample_mask = self.samples_all == sample
        sumw = float(self.weights_all[sample_mask].sum()) if count > 0 else 0.0
        print(f"  - {sample}: events={count}, sumW={sumw:.6f}")

      missing_after_concat = [sample for sample in self.samples_list if sample not in set(unique_samples)]
      if len(missing_after_concat) > 0:
        print("\n[MC DEBUG WARNING] These configured processes have zero events after loading/selection:")
        for sample in missing_after_concat:
          print(f"  - {sample}")

    print("=" * 100 + "\n")

  def print_data_debug_summary(self):
    print("\n" + "=" * 100)
    print("[DATA DEBUG SUMMARY]")
    print("=" * 100)

    print(f"[DATA DEBUG] data files seen:   {len(self.debug_data_files_seen)}")
    print(f"[DATA DEBUG] data files loaded: {len(self.debug_data_files_loaded)}")
    print(f"[DATA DEBUG] missing scores:    {len(self.debug_data_files_missing_scores)}")
    print(f"[DATA DEBUG] mismatched scores: {len(self.debug_data_files_mismatched_scores)}")
    print(f"[DATA DEBUG] failed files:      {len(self.debug_data_files_failed)}")

    if len(self.debug_data_era_dirs_on_disk) > 0:
      print("\n[DATA DEBUG] Era directories found on disk:")
      for era in self.debug_data_era_dirs_on_disk:
        print(f"  - {era}")

    if len(self.debug_data_era_dirs_configured) > 0:
      configured = set(self.debug_data_era_dirs_configured)
      on_disk = set(self.debug_data_era_dirs_on_disk)

      not_configured = sorted(on_disk - configured)
      configured_but_missing = sorted(configured - on_disk)

      if len(not_configured) > 0:
        print("\n[DATA DEBUG WARNING] These era folders exist on disk but are not configured for loading:")
        for era in not_configured:
          print(f"  - {era}")

      if len(configured_but_missing) > 0:
        print("\n[DATA DEBUG INFO] These configured eras are not found on disk:")
        for era in configured_but_missing:
          print(f"  - {era}")

    if len(self.debug_data_files_missing_scores) > 0:
      print("\n[DATA DEBUG WARNING] Files with missing score files. Script 2 keeps them with dummy zero scores:")
      for path in self.debug_data_files_missing_scores:
        print(f"  - {path}")

    if len(self.debug_data_files_mismatched_scores) > 0:
      print("\n[DATA DEBUG WARNING] Files with score/event length mismatch. Script 2 keeps them with dummy zero scores:")
      for item in self.debug_data_files_mismatched_scores:
        print(f"  - {item}")

    if len(self.debug_data_files_failed) > 0:
      print("\n[DATA DEBUG WARNING] Files that failed while reading:")
      for item in self.debug_data_files_failed:
        print(f"  - {item}")

    if self.data_mass_all is None:
      print("\n[DATA DEBUG WARNING] self.data_mass_all is None. No actual data loaded.")
    else:
      print(f"\n[DATA DEBUG] total loaded data events after preselection: {len(self.data_mass_all)}")

      sb_mask = (
        ((self.data_mass_all >= self.mass_left_sb[0]) & (self.data_mass_all < self.mass_left_sb[1]))
        |
        ((self.data_mass_all >= self.mass_right_sb[0]) & (self.data_mass_all < self.mass_right_sb[1]))
      )

      print(f"[DATA DEBUG] total loaded data events in sidebands: {int(np.sum(sb_mask))}")
      print(f"[DATA DEBUG] left SB:  [{self.mass_left_sb[0]}, {self.mass_left_sb[1]})")
      print(f"[DATA DEBUG] right SB: [{self.mass_right_sb[0]}, {self.mass_right_sb[1]})")

    if self.data_scores_all is None:
      print("\n[DATA DEBUG WARNING] self.data_scores_all is None.")
      print("[DATA DEBUG WARNING] Data category masks will be disabled.")
      print("[DATA DEBUG WARNING] N_data_SB may fall back to the full sideband count for each category.")
    else:
      print(f"\n[DATA DEBUG] data_scores_all shape: {self.data_scores_all.shape}")

    if self.data_eras_all is not None and self.data_mass_all is not None:
      print("\n[DATA DEBUG] Loaded data events by normalized era:")
      unique_eras, counts = np.unique(self.data_eras_all, return_counts=True)
      for era, count in zip(unique_eras, counts):
        era_mask = self.data_eras_all == era
        era_sb = (
          ((self.data_mass_all >= self.mass_left_sb[0]) & (self.data_mass_all < self.mass_left_sb[1]))
          |
          ((self.data_mass_all >= self.mass_right_sb[0]) & (self.data_mass_all < self.mass_right_sb[1]))
        ) & era_mask
        print(f"  - {era}: total={int(count)}, sideband={int(np.sum(era_sb))}")

    print("=" * 100 + "\n")

  def load_data_samples(self):
    print("[load_data_samples] Start loading actual data samples...")

    data = {
      "score": [],
      "diphoton_mass": [],
      "dijet_mass": [],
      "weights": [],
      "sample": [],
      "era": [],
    }

    data_base = self._resolve_data_base_dir()

    if not os.path.exists(data_base):
      print(f"[load_data_samples] Data directory does not exist: {data_base}")
      return

    eras = self._discover_subdirs(data_base, "load_data_samples")
    self.debug_data_era_dirs_on_disk = list(eras)
    self.debug_data_era_dirs_configured = list(eras)

    if len(eras) == 0:
      print(f"[load_data_samples] No actual-data era directories found under: {data_base}")
      return

    has_all_scores = True

    for era in eras:
      era_dir = os.path.join(data_base, era)

      if not os.path.exists(era_dir):
        continue

      print(f"[load_data_samples] Era: {era}")
      parquet_files = self._find_data_parquet_files(era_dir)

      if len(parquet_files) == 0:
        print(f"  [load_data_samples] No parquet files found in {era_dir}")
        continue

      for evt_file in parquet_files:
        try:
          columns = [
            "mass",
            self.dijet_mass_key,
            "lead_mvaID",
            "sublead_mvaID",
          ]

          if self.apply_boosted_veto:
            columns.append("is_boosted")

          print(f"  [load_data_samples] Reading parquet: {evt_file}", flush=True)
          self.debug_data_files_seen.append(evt_file)
          events = ak.from_parquet(evt_file, columns=columns)
          n_before = len(events)

          y_file = self._find_matching_y_file_for_data(evt_file)

          if y_file is not None:
            y = np.load(y_file)
            if len(y) != len(events):
              print(
                f"  [load_data_samples] WARNING: score length mismatch for {evt_file}. "
                f"len(y)={len(y)}, len(events)={len(events)}. Data categories disabled."
              )
              self.debug_data_files_mismatched_scores.append(
                f"{evt_file} | len(y)={len(y)}, len(events)={len(events)}"
              )
              y = np.zeros((len(events), 1), dtype=float)
              has_all_scores = False
            else:
              print(f"  [load_data_samples] Found data scores: {y_file}")
          else:
            y = np.zeros((len(events), 1), dtype=float)
            has_all_scores = False
            print(
              f"  [load_data_samples] No y.npy/scores found for {evt_file}. "
              "Preselection data plots will still work, but data category plots/counts need scores."
            )
            self.debug_data_files_missing_scores.append(evt_file)

          if self.apply_preselection:
            events, y = self.preselection(events, y)

          n_after = len(events)

          print(
            f"  [load_data_samples] {os.path.relpath(evt_file, era_dir)}: "
            f"{n_before} -> {n_after} after preselection"
          )

        except Exception as exc:
          print(f"[load_data_samples] ERROR while reading {evt_file}: {exc}")
          self.debug_data_files_failed.append(f"{evt_file} | {exc}")
          continue

        if len(events) == 0:
          continue

        self.debug_data_files_loaded.append(evt_file)

        normalized_era = self._normalize_data_era_label(era)

        data["score"].append(y)
        data["diphoton_mass"].append(np.asarray(events["mass"]))
        data["dijet_mass"].append(np.asarray(events[self.dijet_mass_key]))
        data["weights"].append(np.ones(len(events), dtype=float))
        data["sample"].append(np.repeat("Data", len(events)))
        data["era"].append(np.repeat(normalized_era, len(events)))

    if not data["diphoton_mass"]:
      print("[load_data_samples] No actual data events found.")
      return

    for key in data:
      data[key] = np.concatenate(data[key], axis=0)

    self.data_mass_all = data["diphoton_mass"]
    self.data_dijet_mass_all = data["dijet_mass"]
    self.data_weights_all = data["weights"]
    self.data_samples_all = data["sample"]
    self.data_eras_all = data["era"]

    if has_all_scores:
      self.data_scores_all = data["score"]
      print("[load_data_samples] Data DNN scores loaded. Data category plots/counts are enabled.")
    else:
      self.data_scores_all = None
      print("[load_data_samples] Data DNN scores are incomplete/missing. Only preselection data plots are enabled.")

    print(f"[load_data_samples] Loaded {len(self.data_mass_all)} actual data events total")
    self.print_data_debug_summary()

  def _sr_from_sidebands_linear(self, mass, weights, left_sb, sr, right_sb):
    m = mass
    w = weights

    L = (m >= left_sb[0]) & (m < left_sb[1])
    R = (m >= right_sb[0]) & (m < right_sb[1])

    sumw_L = w[L].sum()
    sumw2_L = (w[L] ** 2).sum()
    sumw_R = w[R].sum()
    sumw2_R = (w[R] ** 2).sum()

    width_L = max(1e-9, left_sb[1] - left_sb[0])
    width_R = max(1e-9, right_sb[1] - right_sb[0])
    width_SR = max(1e-9, sr[1] - sr[0])

    dens_L = sumw_L / width_L if width_L > 0 else 0.0
    var_dens_L = sumw2_L / (width_L ** 2) if width_L > 0 else 0.0

    dens_R = sumw_R / width_R if width_R > 0 else 0.0
    var_dens_R = sumw2_R / (width_R ** 2) if width_R > 0 else 0.0

    have_L = sumw_L > 0.0
    have_R = sumw_R > 0.0

    if have_L and have_R:
      centroid_L = 0.5 * (left_sb[0] + left_sb[1])
      centroid_R = 0.5 * (right_sb[0] + right_sb[1])
      centroid_SR = 0.5 * (sr[0] + sr[1])
      span = max(1e-9, centroid_R - centroid_L)
      w_L = (centroid_R - centroid_SR) / span
      w_R = (centroid_SR - centroid_L) / span
      dens = w_L * dens_L + w_R * dens_R
      var_dens = w_L ** 2 * var_dens_L + w_R ** 2 * var_dens_R
    elif have_L:
      dens = dens_L
      var_dens = var_dens_L
    elif have_R:
      dens = dens_R
      var_dens = var_dens_R
    else:
      return 0.0, 0.0

    b_sr = dens * width_SR
    var_b_sr = var_dens * (width_SR ** 2)
    return float(b_sr), float(var_b_sr)

  def asymptotic_significance(self, s, b):
    if s <= 0.0 or b <= 0.0:
      return float("nan")
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))

  def parse_best_cut_params(self):
    print(f"[parse_best_cut_params] Loading threshold json from: {self.best_cut_json}")
    with open(self.best_cut_json, "r") as f:
      best_params = json.load(f)

    if not isinstance(best_params, list):
      raise ValueError(
        f"{self.best_cut_json} does not contain a list. "
        "Please pass best_cut_params.json, not best_cut_params.txt."
      )

    print(f"[parse_best_cut_params] Loaded {len(best_params)} ggHH categories")
    return best_params

  def parse_vbfhh_info(self):
    if not os.path.exists(self.vbfhh_info_json):
      print(f"[parse_vbfhh_info] No VBFHH info file found: {self.vbfhh_info_json}")
      return []

    print(f"[parse_vbfhh_info] Loading VBFHH info from: {self.vbfhh_info_json}")
    with open(self.vbfhh_info_json, "r") as f:
      vbfhh_info = json.load(f)

    if not isinstance(vbfhh_info, list):
      raise ValueError(f"{self.vbfhh_info_json} does not contain a list.")

    print(f"[parse_vbfhh_info] Loaded {len(vbfhh_info)} VBFHH categories")
    return vbfhh_info

  def make_category_mask_explicit_from_scores(self, scores, params, signal_class, bkg_classes):
    mask = np.ones(len(scores), dtype=bool)
    mask &= scores[:, signal_class] > params["th_signal"]
    for b in bkg_classes:
      mask &= scores[:, b] < params[f"th_bg_{b}"]
    return mask

  def make_vbfhh_mask_explicit_from_scores(self, scores, threshold, vbfhh_class):
    mask = np.ones(len(scores), dtype=bool)
    mask &= scores[:, vbfhh_class] > threshold
    return mask

  def make_category_mask_explicit(self, params, signal_class, bkg_classes):
    return self.make_category_mask_explicit_from_scores(
      self.scores_all,
      params,
      signal_class,
      bkg_classes,
    )

  def make_vbfhh_mask_explicit(self, threshold, vbfhh_class):
    return self.make_vbfhh_mask_explicit_from_scores(
      self.scores_all,
      threshold,
      vbfhh_class,
    )

  def build_all_category_masks(self, signal_class, bkg_classes, vbfhh_class=None):
    print("[build_all_category_masks] Building MC category masks...")
    ggHH_best_params_list = self.parse_best_cut_params()
    ggHH_best_params_list = ggHH_best_params_list[:3]

    vbfhh_info_list = self.parse_vbfhh_info() if vbfhh_class is not None else []

    masks = {}
    already_used = np.zeros(len(self.mass_all), dtype=bool)

    if vbfhh_class is not None and len(vbfhh_info_list) > 0:
      info = vbfhh_info_list[0]
      threshold = info["best_threshold"]
      base_mask = self.make_vbfhh_mask_explicit(threshold, vbfhh_class)
      cat_mask = base_mask & (~already_used)
      masks["vbfhh_cat1"] = cat_mask
      already_used |= cat_mask
      print(f"[build_all_category_masks] vbfhh_cat1: {int(cat_mask.sum())} selected MC events")

    for i_cat, params in enumerate(ggHH_best_params_list, start=1):
      base_mask = self.make_category_mask_explicit(params, signal_class, bkg_classes)
      cat_mask = base_mask & (~already_used)
      cat_name = f"cat{i_cat}"
      masks[cat_name] = cat_mask
      already_used |= cat_mask
      print(f"[build_all_category_masks] {cat_name}: {int(cat_mask.sum())} selected MC events")

    return masks

  def build_all_data_category_masks(self, signal_class, bkg_classes, vbfhh_class=None):
    if self.data_mass_all is None:
      return {}

    if self.data_scores_all is None:
      print("[build_all_data_category_masks] No data scores available. Category masks disabled for data.")
      return {}

    print("[build_all_data_category_masks] Building actual-data category masks...")

    needed_score_columns = [signal_class] + list(bkg_classes)
    if vbfhh_class is not None:
      needed_score_columns.append(vbfhh_class)

    max_needed_col = max(needed_score_columns) if len(needed_score_columns) > 0 else None

    print(f"[DATA DEBUG] data_scores_all shape: {self.data_scores_all.shape}")
    print(f"[DATA DEBUG] needed score columns: {needed_score_columns}")

    if max_needed_col is not None and self.data_scores_all.shape[1] <= max_needed_col:
      print(
        "[DATA DEBUG WARNING] data_scores_all does not have enough columns for the requested "
        f"signal/background/VBFHH class indices. shape={self.data_scores_all.shape}, "
        f"max_needed_col={max_needed_col}"
      )

    ggHH_best_params_list = self.parse_best_cut_params()
    ggHH_best_params_list = ggHH_best_params_list[:3]

    vbfhh_info_list = self.parse_vbfhh_info() if vbfhh_class is not None else []

    masks = {}
    already_used = np.zeros(len(self.data_mass_all), dtype=bool)

    if vbfhh_class is not None and len(vbfhh_info_list) > 0:
      info = vbfhh_info_list[0]
      threshold = info["best_threshold"]
      base_mask = self.make_vbfhh_mask_explicit_from_scores(
        self.data_scores_all,
        threshold,
        vbfhh_class,
      )
      cat_mask = base_mask & (~already_used)
      masks["vbfhh_cat1"] = cat_mask
      already_used |= cat_mask
      print(f"[build_all_data_category_masks] vbfhh_cat1: {int(cat_mask.sum())} selected data events")

    for i_cat, params in enumerate(ggHH_best_params_list, start=1):
      base_mask = self.make_category_mask_explicit_from_scores(
        self.data_scores_all,
        params,
        signal_class,
        bkg_classes,
      )
      cat_mask = base_mask & (~already_used)
      cat_name = f"cat{i_cat}"
      masks[cat_name] = cat_mask
      already_used |= cat_mask
      print(f"[build_all_data_category_masks] {cat_name}: {int(cat_mask.sum())} selected data events")

    return masks

  def _get_signal_samples_for_region(self, region_name):
    if region_name == "vbfhh_cat1":
      return self.vbfhh_samples
    return self.ggHH_signal_samples if len(self.ggHH_signal_samples) > 0 else self.signal_samples

  def _actual_data_sideband_count(self, data_cat_mask=None, era=None):
    if self.data_mass_all is None:
      return 0

    sb_mask = (
      ((self.data_mass_all >= self.mass_left_sb[0]) & (self.data_mass_all < self.mass_left_sb[1]))
      |
      ((self.data_mass_all >= self.mass_right_sb[0]) & (self.data_mass_all < self.mass_right_sb[1]))
    )

    if data_cat_mask is not None:
      sb_mask = sb_mask & data_cat_mask

    if era is not None:
      if self.data_eras_all is None:
        return 0
      sb_mask = sb_mask & (self.data_eras_all == era)

    return int(np.sum(sb_mask))

  def compute_yield_table(self, signal_class, bkg_classes, vbfhh_class=None):
    print("[compute_yield_table] Start computing yields...")
    masks = self.build_all_category_masks(signal_class, bkg_classes, vbfhh_class=vbfhh_class)
    data_masks = self.build_all_data_category_masks(signal_class, bkg_classes, vbfhh_class=vbfhh_class)

    if self.data_mass_all is not None and len(data_masks) == 0:
      print("\n[DATA DEBUG WARNING] data_masks is empty.")
      print("[DATA DEBUG WARNING] This means category-specific data masks are not available.")
      print("[DATA DEBUG WARNING] In Script 2's current logic, N_data_SB will be counted with data_cat_mask=None.")
      print("[DATA DEBUG WARNING] Therefore each category may receive the FULL data sideband count, not category-selected data.\n")

    rows = []
    sr_low, sr_high = self.mass_sr

    for cat_name, cat_mask in masks.items():
      print(f"[compute_yield_table] Processing {cat_name} ...")
      masses = self.mass_all[cat_mask]
      weights = self.weights_all[cat_mask]
      samples = self.samples_all[cat_mask]

      signal_samples_this_region = self._get_signal_samples_for_region(cat_name)

      sr_mask = (masses > sr_low) & (masses < sr_high)
      sb_mask = (
        ((masses >= self.mass_left_sb[0]) & (masses < self.mass_left_sb[1]))
        |
        ((masses >= self.mass_right_sb[0]) & (masses < self.mass_right_sb[1]))
      )

      signal_mask, background_mask = self._make_signal_and_background_masks(
        samples,
        signal_samples_this_region,
      )

      s_sr = float(weights[sr_mask & signal_mask].sum())

      b_interp_sr = 0.0
      interp_mask = background_mask & np.isin(samples, list(self.interp_samples))
      if np.any(interp_mask):
        b_interp_sr, _ = self._sr_from_sidebands_linear(
          masses[interp_mask],
          weights[interp_mask],
          left_sb=self.mass_left_sb,
          sr=self.mass_sr,
          right_sb=self.mass_right_sb,
        )

      b_other_sr = float(
        weights[
          sr_mask
          &
          background_mask
          &
          (~np.isin(samples, list(self.interp_samples)))
        ].sum()
      )

      b_total = float(b_interp_sr + b_other_sr)
      b_total_sb = float(weights[sb_mask & background_mask].sum())

      data_cat_mask = data_masks.get(cat_name, None)
      if data_cat_mask is None:
        print(
          f"[DATA DEBUG WARNING] {cat_name}: data_cat_mask is None. "
          "N_data_SB will be the full data sideband count, not category-specific."
        )
      else:
        cat_data_sb_debug = self._actual_data_sideband_count(data_cat_mask=data_cat_mask)
        full_data_sb_debug = self._actual_data_sideband_count(data_cat_mask=None)
        print(
          f"[DATA DEBUG] {cat_name}: category data sideband={cat_data_sb_debug}, "
          f"full data sideband={full_data_sb_debug}"
        )
      n_data_sb = self._actual_data_sideband_count(data_cat_mask=data_cat_mask)

      z_asimov = self.asymptotic_significance(s_sr, b_total)

      rows.append({
        "Category": cat_name,
        "S_SR": s_sr,
        "B_interp_SR": float(b_interp_sr),
        "B_other_SR": b_other_sr,
        "B_total": b_total,
        "B_total_SB": b_total_sb,
        "N_data_SB": n_data_sb,
        "Z_asimov": z_asimov,
      })

      print(
        f"[compute_yield_table] {cat_name}: "
        f"N_MC={int(cat_mask.sum())}, S_SR={s_sr:.3f}, "
        f"B_interp_SR={b_interp_sr:.3f}, B_other_SR={b_other_sr:.3f}, "
        f"B_total={b_total:.3f}, B_total_SB={b_total_sb:.3f}, "
        f"N_data_SB={n_data_sb}, "
        f"Z={z_asimov if pd.notnull(z_asimov) else float('nan')}"
      )

    return pd.DataFrame(rows)

  def compute_yield_table_per_era(self, signal_class, bkg_classes, vbfhh_class=None):
    print("[compute_yield_table_per_era] Start computing per-era yields...")

    masks = self.build_all_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )

    data_masks = self.build_all_data_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )

    if self.eras_all is None:
      raise RuntimeError("[compute_yield_table_per_era] self.eras_all is not available.")

    era_order = [
      "2016preVFP",
      "2016postVFP",
      "2017",
      "2018",
      "preEE",
      "postEE",
      "preBPix",
      "postBPix",
      "2024",
      "2025",
    ]

    loaded_eras = list(np.unique(self.eras_all))
    era_order = [era for era in era_order if era in loaded_eras] + [
      era for era in loaded_eras if era not in era_order
    ]

    rows = []
    sr_low, sr_high = self.mass_sr

    for era in era_order:
      era_mask_all = self.eras_all == era
      print(f"[compute_yield_table_per_era] Era: {era}")

      for cat_name, cat_mask in masks.items():
        print(f"[compute_yield_table_per_era] Processing {era} / {cat_name} ...")

        full_mask = cat_mask & era_mask_all

        masses = self.mass_all[full_mask]
        weights = self.weights_all[full_mask]
        samples = self.samples_all[full_mask]

        signal_samples_this_region = self._get_signal_samples_for_region(cat_name)

        if len(masses) == 0:
          data_cat_mask = data_masks.get(cat_name, None)
          n_data_sb = self._actual_data_sideband_count(
            data_cat_mask=data_cat_mask,
            era=era,
          )

          rows.append({
            "Era": era,
            "Category": cat_name,
            "S_SR": 0.0,
            "B_interp_SR": 0.0,
            "B_other_SR": 0.0,
            "B_total": 0.0,
            "B_total_SB": 0.0,
            "N_data_SB": n_data_sb,
            "Z_asimov": float("nan"),
          })

          continue

        sr_mask = (masses > sr_low) & (masses < sr_high)

        sb_mask = (
          ((masses >= self.mass_left_sb[0]) & (masses < self.mass_left_sb[1]))
          |
          ((masses >= self.mass_right_sb[0]) & (masses < self.mass_right_sb[1]))
        )

        signal_mask, background_mask = self._make_signal_and_background_masks(
          samples,
          signal_samples_this_region,
        )

        s_sr = float(weights[sr_mask & signal_mask].sum())

        b_interp_sr = 0.0
        interp_mask = background_mask & np.isin(samples, list(self.interp_samples))

        if np.any(interp_mask):
          b_interp_sr, _ = self._sr_from_sidebands_linear(
            masses[interp_mask],
            weights[interp_mask],
            left_sb=self.mass_left_sb,
            sr=self.mass_sr,
            right_sb=self.mass_right_sb,
          )

        b_other_sr = float(
          weights[
            sr_mask
            &
            background_mask
            &
            (~np.isin(samples, list(self.interp_samples)))
          ].sum()
        )

        b_total = float(b_interp_sr + b_other_sr)
        b_total_sb = float(weights[sb_mask & background_mask].sum())

        data_cat_mask = data_masks.get(cat_name, None)
        n_data_sb = self._actual_data_sideband_count(
          data_cat_mask=data_cat_mask,
          era=era,
        )

        z_asimov = self.asymptotic_significance(s_sr, b_total)

        rows.append({
          "Era": era,
          "Category": cat_name,
          "S_SR": s_sr,
          "B_interp_SR": float(b_interp_sr),
          "B_other_SR": b_other_sr,
          "B_total": b_total,
          "B_total_SB": b_total_sb,
          "N_data_SB": n_data_sb,
          "Z_asimov": z_asimov,
        })

        print(
          f"[compute_yield_table_per_era] {era} / {cat_name}: "
          f"N_MC={int(full_mask.sum())}, "
          f"S_SR={s_sr:.6f}, "
          f"B_interp_SR={b_interp_sr:.6f}, "
          f"B_other_SR={b_other_sr:.6f}, "
          f"B_total={b_total:.6f}, "
          f"B_total_SB={b_total_sb:.6f}, "
          f"N_data_SB={n_data_sb}, "
          f"Z={z_asimov if pd.notnull(z_asimov) else float('nan')}"
        )

    df_per_era = pd.DataFrame(rows)

    print("[compute_yield_table_per_era] Done.")
    return df_per_era

  def compute_sr_process_column_yield_table(
    self,
    signal_class,
    bkg_classes,
    vbfhh_class=None,
  ):
    masks = self.build_all_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )
    data_masks = self.build_all_data_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )

    category_names = list(masks.keys())
    sr_low, sr_high = self.mass_sr

    sample_yields = {
      sample: {cat_name: 0.0 for cat_name in category_names}
      for sample in self.samples_list
    }

    signal_total = {cat_name: 0.0 for cat_name in category_names}
    interp_bkg_total = {cat_name: 0.0 for cat_name in category_names}
    other_bkg_total = {cat_name: 0.0 for cat_name in category_names}
    bkg_total = {cat_name: 0.0 for cat_name in category_names}
    total_yield = {cat_name: 0.0 for cat_name in category_names}
    n_data_sb = {cat_name: 0 for cat_name in category_names}
    z_asimov = {cat_name: float("nan") for cat_name in category_names}

    for cat_name, cat_mask in masks.items():
      masses = self.mass_all[cat_mask]
      weights = self.weights_all[cat_mask]
      samples = self.samples_all[cat_mask]

      sr_mask = (masses > sr_low) & (masses < sr_high)
      signal_samples_this_region = self._get_signal_samples_for_region(cat_name)

      data_cat_mask = data_masks.get(cat_name, None)
      n_data_sb[cat_name] = self._actual_data_sideband_count(data_cat_mask=data_cat_mask)

      for sample in self.samples_list:
        sample_mask = samples == sample

        if not np.any(sample_mask):
          yld = 0.0
        elif sample in self.interp_samples:
          yld, _ = self._sr_from_sidebands_linear(
            masses[sample_mask],
            weights[sample_mask],
            left_sb=self.mass_left_sb,
            sr=self.mass_sr,
            right_sb=self.mass_right_sb,
          )
        else:
          yld = float(weights[sample_mask & sr_mask].sum())

        sample_yields[sample][cat_name] = float(yld)

      for sample in self.samples_list:
        yld = sample_yields[sample][cat_name]
        if sample in signal_samples_this_region:
          signal_total[cat_name] += yld

      _, background_mask = self._make_signal_and_background_masks(
        samples,
        signal_samples_this_region,
      )

      interp_mask = background_mask & np.isin(samples, list(self.interp_samples))
      if np.any(interp_mask):
        interp_bkg_total[cat_name], _ = self._sr_from_sidebands_linear(
          masses[interp_mask],
          weights[interp_mask],
          left_sb=self.mass_left_sb,
          sr=self.mass_sr,
          right_sb=self.mass_right_sb,
        )

      other_mask = (
        sr_mask
        &
        background_mask
        &
        (~np.isin(samples, list(self.interp_samples)))
      )
      other_bkg_total[cat_name] = float(weights[other_mask].sum())

      bkg_total[cat_name] = interp_bkg_total[cat_name] + other_bkg_total[cat_name]
      total_yield[cat_name] = signal_total[cat_name] + bkg_total[cat_name]
      z_asimov[cat_name] = self.asymptotic_significance(
        signal_total[cat_name],
        bkg_total[cat_name],
      )

    rows = []

    for sample in self.samples_list:
      row = {"Process": sample}
      for cat_name in category_names:
        row[cat_name] = sample_yields[sample][cat_name]
      rows.append(row)

    separator_row = {"Process": "---"}
    for cat_name in category_names:
      separator_row[cat_name] = np.nan
    rows.append(separator_row)

    grouped_rows = [
      ("Signal total", signal_total),
      ("Interpolated background total", interp_bkg_total),
      ("Other background total", other_bkg_total),
      ("Background total", bkg_total),
      ("Total yield", total_yield),
      ("N_data_SB", n_data_sb),
      ("Z_asimov", z_asimov),
    ]

    for row_name, values in grouped_rows:
      row = {"Process": row_name}
      for cat_name in category_names:
        row[cat_name] = values[cat_name]
      rows.append(row)

    return pd.DataFrame(rows)

  def compute_sr_process_column_yield_table_per_era(
    self,
    signal_class,
    bkg_classes,
    vbfhh_class=None,
  ):
    print("[compute_sr_process_column_yield_table_per_era] Start...")

    masks = self.build_all_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )

    data_masks = self.build_all_data_category_masks(
      signal_class,
      bkg_classes,
      vbfhh_class=vbfhh_class,
    )

    if self.eras_all is None:
      raise RuntimeError(
        "[compute_sr_process_column_yield_table_per_era] self.eras_all is not available."
      )

    category_names = list(masks.keys())
    sr_low, sr_high = self.mass_sr

    era_order = [
      "2016preVFP",
      "2016postVFP",
      "2017",
      "2018",
      "preEE",
      "postEE",
      "preBPix",
      "postBPix",
      "2024",
      "2025",
    ]

    loaded_eras = list(np.unique(self.eras_all))
    era_order = [era for era in era_order if era in loaded_eras] + [
      era for era in loaded_eras if era not in era_order
    ]

    rows = []

    for era in era_order:
      print(f"[compute_sr_process_column_yield_table_per_era] Era: {era}")

      era_mask_all = self.eras_all == era

      sample_yields = {
        sample: {cat_name: 0.0 for cat_name in category_names}
        for sample in self.samples_list
      }

      signal_total = {cat_name: 0.0 for cat_name in category_names}
      interp_bkg_total = {cat_name: 0.0 for cat_name in category_names}
      other_bkg_total = {cat_name: 0.0 for cat_name in category_names}
      bkg_total = {cat_name: 0.0 for cat_name in category_names}
      total_yield = {cat_name: 0.0 for cat_name in category_names}
      n_data_sb = {cat_name: 0 for cat_name in category_names}
      z_asimov = {cat_name: float("nan") for cat_name in category_names}

      for cat_name, cat_mask in masks.items():
        print(
          f"[compute_sr_process_column_yield_table_per_era] Processing {era} / {cat_name} ..."
        )

        full_mask = cat_mask & era_mask_all

        masses = self.mass_all[full_mask]
        weights = self.weights_all[full_mask]
        samples = self.samples_all[full_mask]

        signal_samples_this_region = self._get_signal_samples_for_region(cat_name)

        data_cat_mask = data_masks.get(cat_name, None)
        n_data_sb[cat_name] = self._actual_data_sideband_count(
          data_cat_mask=data_cat_mask,
          era=era,
        )

        if len(masses) == 0:
          continue

        sr_mask = (masses > sr_low) & (masses < sr_high)

        for sample in self.samples_list:
          sample_mask = samples == sample

          if not np.any(sample_mask):
            yld = 0.0
          elif sample in self.interp_samples:
            yld, _ = self._sr_from_sidebands_linear(
              masses[sample_mask],
              weights[sample_mask],
              left_sb=self.mass_left_sb,
              sr=self.mass_sr,
              right_sb=self.mass_right_sb,
            )
          else:
            yld = float(weights[sample_mask & sr_mask].sum())

          sample_yields[sample][cat_name] = float(yld)

        for sample in self.samples_list:
          yld = sample_yields[sample][cat_name]

          if sample in signal_samples_this_region:
            signal_total[cat_name] += yld

        _, background_mask = self._make_signal_and_background_masks(
          samples,
          signal_samples_this_region,
        )

        interp_mask = background_mask & np.isin(samples, list(self.interp_samples))

        if np.any(interp_mask):
          interp_bkg_total[cat_name], _ = self._sr_from_sidebands_linear(
            masses[interp_mask],
            weights[interp_mask],
            left_sb=self.mass_left_sb,
            sr=self.mass_sr,
            right_sb=self.mass_right_sb,
          )

        other_mask = (
          sr_mask
          &
          background_mask
          &
          (~np.isin(samples, list(self.interp_samples)))
        )

        other_bkg_total[cat_name] = float(weights[other_mask].sum())

        bkg_total[cat_name] = interp_bkg_total[cat_name] + other_bkg_total[cat_name]
        total_yield[cat_name] = signal_total[cat_name] + bkg_total[cat_name]
        z_asimov[cat_name] = self.asymptotic_significance(
          signal_total[cat_name],
          bkg_total[cat_name],
        )

        print(
          f"[compute_sr_process_column_yield_table_per_era] {era} / {cat_name}: "
          f"S={signal_total[cat_name]:.6f}, "
          f"B_interp={interp_bkg_total[cat_name]:.6f}, "
          f"B_other={other_bkg_total[cat_name]:.6f}, "
          f"B_total={bkg_total[cat_name]:.6f}, "
          f"N_data_SB={n_data_sb[cat_name]}, "
          f"Z={z_asimov[cat_name]:.6f}"
        )

      for sample in self.samples_list:
        row = {
          "Era": era,
          "Process": sample,
        }

        for cat_name in category_names:
          row[cat_name] = sample_yields[sample][cat_name]

        rows.append(row)

      separator_row = {
        "Era": era,
        "Process": "---",
      }

      for cat_name in category_names:
        separator_row[cat_name] = np.nan

      rows.append(separator_row)

      grouped_rows = [
        ("Signal total", signal_total),
        ("Interpolated background total", interp_bkg_total),
        ("Other background total", other_bkg_total),
        ("Background total", bkg_total),
        ("Total yield", total_yield),
        ("N_data_SB", n_data_sb),
        ("Z_asimov", z_asimov),
      ]

      for row_name, values in grouped_rows:
        row = {
          "Era": era,
          "Process": row_name,
        }

        for cat_name in category_names:
          row[cat_name] = values[cat_name]

        rows.append(row)

    df_process_sr_per_era = pd.DataFrame(rows)

    print("[compute_sr_process_column_yield_table_per_era] Done.")
    return df_process_sr_per_era

  def _hist_norm_with_err(self, values, weights, bins):
    y, edges = np.histogram(values, bins=bins, weights=weights)
    y2, _ = np.histogram(values, bins=bins, weights=weights * weights)
    err = np.sqrt(y2)

    tot = float(np.sum(y))
    if tot > 0.0:
      y = y / tot
      err = err / tot
    else:
      y = y.astype(float)
      err = err.astype(float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, y, err

  def _get_data_base_presel_mask(self):
    if self.data_mass_all is None:
      return None

    return (
      (self.data_mass_all >= 100.0) & (self.data_mass_all <= 180.0) &
      (self.data_dijet_mass_all >= 80.0) & (self.data_dijet_mass_all <= 190.0)
    )

  def plot_sr_vs_preselection(self, signal_class, bkg_classes, vbfhh_class=None, out_prefix="sr_vs_preselection"):
    print("[plot_sr_vs_preselection] Start...")
    masks = self.build_all_category_masks(signal_class, bkg_classes, vbfhh_class=vbfhh_class)

    presel_mask = (
      (self.mass_all >= 100.0) & (self.mass_all <= 180.0) &
      (self.dijet_mass_all >= 80.0) & (self.dijet_mass_all <= 190.0) &
      np.isin(self.samples_all, list(self.plot_samples))
    )

    cats_for_plot = []
    if "vbfhh_cat1" in masks:
      cats_for_plot.append("vbfhh_cat1")
    for c in ["cat1", "cat2", "cat3"]:
      if c in masks:
        cats_for_plot.append(c)

    if not cats_for_plot:
      print("[plot_sr_vs_preselection] No categories found. Skip.")
      return

    os.makedirs(self.output_dir, exist_ok=True)

    print("[plot_sr_vs_preselection] Making mgg plot...")
    bins_mass = np.linspace(100.0, 180.0, 41)
    fig_m, ax_m = plt.subplots(figsize=(8.0, 6.5))

    m_pre = self.mass_all[presel_mask]
    w_pre = self.weights_all[presel_mask]

    if len(m_pre) == 0:
      print("[plot_sr_vs_preselection] Preselection mask is empty. Skip.")
      plt.close(fig_m)
      return

    self._draw_norm_hist_with_err(
      ax=ax_m,
      values=m_pre,
      weights=w_pre,
      bins=bins_mass,
      label="MC Preselection: GGJets + TTGG",
      color=self.region_colors["Preselection"],
      linewidth=2.2,
    )
    for cat in cats_for_plot:
      mask_cat = masks[cat] & presel_mask
      m_cat = self.mass_all[mask_cat]
      w_cat = self.weights_all[mask_cat]
      print(f"[plot_sr_vs_preselection] {cat}: {len(m_cat)} MC events in preselection mask")

      if len(m_cat) == 0:
        continue

      self._draw_norm_hist_with_err(
        ax=ax_m,
        values=m_cat,
        weights=w_cat,
        bins=bins_mass,
        label=f"MC {cat}",
        color=self.region_colors.get(cat, None),
        linewidth=2.0,
      )

    self._apply_cms_axes(
      ax=ax_m,
      xlabel=r"$m_{\gamma\gamma}$ [GeV]",
      ylabel="Normalized Events",
      xlim=(100.0, 180.0),
      add_label=True,
    )
    self._add_panel_text(ax_m, "GGJets + TTGG")
    ax_m.legend(loc="upper right", ncol=1)
    fig_m.tight_layout()

    fig_m.savefig(os.path.join(self.output_dir, f"{out_prefix}_mass.png"), bbox_inches="tight")
    fig_m.savefig(os.path.join(self.output_dir, f"{out_prefix}_mass.pdf"), bbox_inches="tight")
    plt.close(fig_m)

    print("[plot_sr_vs_preselection] Making mjj plot...")
    bins_dijet = np.linspace(80.0, 190.0, 13)
    fig_j, ax_j = plt.subplots(figsize=(8.0, 6.5))

    j_pre = self.dijet_mass_all[presel_mask]
    self._draw_norm_hist_with_err(
      ax=ax_j,
      values=j_pre,
      weights=w_pre,
      bins=bins_dijet,
      label="MC Preselection: GGJets + TTGG",
      color=self.region_colors["Preselection"],
      linewidth=2.2,
    )
    for cat in cats_for_plot:
      mask_cat = masks[cat] & presel_mask
      j_cat = self.dijet_mass_all[mask_cat]
      w_cat = self.weights_all[mask_cat]

      if len(j_cat) == 0:
        continue

      self._draw_norm_hist_with_err(
        ax=ax_j,
        values=j_cat,
        weights=w_cat,
        bins=bins_dijet,
        label=f"MC {cat}",
        color=self.region_colors.get(cat, None),
        linewidth=2.0,
      )

    self._apply_cms_axes(
      ax=ax_j,
      xlabel=r"$m_{jj}$ [GeV]",
      ylabel="Normalized Events",
      xlim=(80.0, 190.0),
      add_label=True,
    )
    self._add_panel_text(ax_j, "GGJets + TTGG")
    ax_j.legend(loc="upper right", ncol=1)
    fig_j.tight_layout()

    fig_j.savefig(os.path.join(self.output_dir, f"{out_prefix}_dijet_mass.png"), bbox_inches="tight")
    fig_j.savefig(os.path.join(self.output_dir, f"{out_prefix}_dijet_mass.pdf"), bbox_inches="tight")
    plt.close(fig_j)

    print("[plot_sr_vs_preselection] Done.")

  def plot_presel_and_srs_separately(
    self,
    signal_class,
    bkg_classes,
    vbfhh_class=None,
    out_prefix="presel_and_srs",
  ):
    print("[plot_presel_and_srs_separately] Start...")
    masks = self.build_all_category_masks(signal_class, bkg_classes, vbfhh_class=vbfhh_class)
    region_names = ["Preselection"]
    if "vbfhh_cat1" in masks:
      region_names.append("vbfhh_cat1")
    for cat in ["cat1", "cat2", "cat3"]:
      if cat in masks:
        region_names.append(cat)

    if len(region_names) < 2:
      print("[plot_presel_and_srs_separately] No categories found. Skip.")
      return

    base_presel_mask = (
      (self.mass_all >= 100.0) & (self.mass_all <= 180.0) &
      (self.dijet_mass_all >= 80.0) & (self.dijet_mass_all <= 190.0) &
      np.isin(self.samples_all, list(self.plot_samples))
    )

    sample_rows = [
      ("All (GGJets + TTGG)", np.isin(self.samples_all, list(self.plot_samples))),
      ("GGJets", self.samples_all == "GGJets"),
      ("TTGG", self.samples_all == "TTGG"),
    ]

    os.makedirs(self.output_dir, exist_ok=True)

    n_rows = len(sample_rows)
    n_cols = len(region_names)
    bins_mgg = np.linspace(100.0, 180.0, 41)

    fig_mgg, axes_mgg = plt.subplots(
      n_rows,
      n_cols,
      figsize=(5.4 * n_cols, 4.1 * n_rows),
      sharey="row",
    )

    if n_cols == 1:
      axes_mgg = np.array(axes_mgg).reshape(n_rows, 1)

    for i_row, (row_name, row_sample_mask) in enumerate(sample_rows):
      for i_col, region_name in enumerate(region_names):
        ax = axes_mgg[i_row, i_col]

        if region_name == "Preselection":
          mask = base_presel_mask & row_sample_mask
        else:
          mask = base_presel_mask & row_sample_mask & masks[region_name]

        values = self.mass_all[mask]
        weights = self.weights_all[mask]
        n_evt = int(mask.sum())
        yield_sum = float(weights.sum())

        if len(values) > 0:
          self._draw_norm_hist_with_err(
            ax=ax,
            values=values,
            weights=weights,
            bins=bins_mgg,
            label=None,
            color=self.sample_row_colors.get(row_name, None),
            linewidth=2.0,
          )

        self._apply_cms_axes(
          ax=ax,
          xlabel=r"$m_{\gamma\gamma}$ [GeV]",
          ylabel=f"{row_name}\nNormalized Events" if i_col == 0 else None,
          xlim=(100.0, 180.0),
          add_label=(i_row == 0 and i_col == 0),
        )

        panel_text = f"{region_name}\nN_raw = {n_evt}\nYield = {yield_sum:.3f}"
        self._add_panel_text(ax, panel_text, fontsize=12)

    fig_mgg.tight_layout()
    fig_mgg.savefig(
      os.path.join(self.output_dir, f"{out_prefix}_mgg.png"),
      bbox_inches="tight"
    )
    fig_mgg.savefig(
      os.path.join(self.output_dir, f"{out_prefix}_mgg.pdf"),
      bbox_inches="tight"
    )
    plt.close(fig_mgg)

    bins_mjj = np.linspace(80.0, 190.0, 13)

    fig_mjj, axes_mjj = plt.subplots(
      n_rows,
      n_cols,
      figsize=(5.4 * n_cols, 4.1 * n_rows),
      sharey="row",
    )

    if n_cols == 1:
      axes_mjj = np.array(axes_mjj).reshape(n_rows, 1)

    for i_row, (row_name, row_sample_mask) in enumerate(sample_rows):
      for i_col, region_name in enumerate(region_names):
        ax = axes_mjj[i_row, i_col]

        if region_name == "Preselection":
          mask = base_presel_mask & row_sample_mask
        else:
          mask = base_presel_mask & row_sample_mask & masks[region_name]

        values = self.dijet_mass_all[mask]
        weights = self.weights_all[mask]
        n_evt = int(mask.sum())
        yield_sum = float(weights.sum())

        if len(values) > 0:
          self._draw_norm_hist_with_err(
            ax=ax,
            values=values,
            weights=weights,
            bins=bins_mjj,
            label=None,
            color=self.sample_row_colors.get(row_name, None),
            linewidth=2.0,
          )

        self._apply_cms_axes(
          ax=ax,
          xlabel=r"$m_{jj}$ [GeV]",
          ylabel=f"{row_name}\nNormalized Events" if i_col == 0 else None,
          xlim=(80.0, 190.0),
          add_label=(i_row == 0 and i_col == 0),
        )

        panel_text = f"{region_name}\nN_raw = {n_evt}\nYield = {yield_sum:.3f}"
        self._add_panel_text(ax, panel_text, fontsize=12)

    fig_mjj.tight_layout()
    fig_mjj.savefig(
      os.path.join(self.output_dir, f"{out_prefix}_mjj.pdf"),
      bbox_inches="tight"
    )
    fig_mjj.savefig(
      os.path.join(self.output_dir, f"{out_prefix}_mjj.png"),
      bbox_inches="tight"
    )
    plt.close(fig_mjj)

    print("[plot_presel_and_srs_separately] Done.")

  def weighted_pearson_corr(self, x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[m]
    y = y[m]
    w = w[m]
    if x.size < 2:
      return float("nan")

    wsum = np.sum(w)
    if not np.isfinite(wsum) or abs(wsum) < 1e-12:
      return float("nan")

    mx = np.sum(w * x) / wsum
    my = np.sum(w * y) / wsum

    cov = np.sum(w * (x - mx) * (y - my)) / wsum
    vx = np.sum(w * (x - mx) ** 2) / wsum
    vy = np.sum(w * (y - my) ** 2) / wsum

    if vx <= 0.0 or vy <= 0.0:
      return float("nan")

    return float(cov / np.sqrt(vx * vy))

  def plot_mgg_mjj_correlation(self, signal_class, bkg_classes, vbfhh_class=None, out_prefix="mgg_mjj_corr"):
    print("[plot_mgg_mjj_correlation] Start...")
    masks = self.build_all_category_masks(signal_class, bkg_classes, vbfhh_class=vbfhh_class)
    cats_for_plot = []
    if "vbfhh_cat1" in masks:
      cats_for_plot.append("vbfhh_cat1")
    for c in ["cat1", "cat2", "cat3"]:
      if c in masks:
        cats_for_plot.append(c)

    if not cats_for_plot:
      print("[plot_mgg_mjj_correlation] No categories found. Skip.")
      return

    presel_mask = (
      (self.mass_all >= 100.0) & (self.mass_all <= 180.0) &
      (self.dijet_mass_all >= 80.0) & (self.dijet_mass_all <= 190.0) &
      np.isin(self.samples_all, list(self.plot_samples))
    )

    groups = [
      ("nonResBkg", np.isin(self.samples_all, list(self.plot_samples))),
      ("Signal", np.isin(self.samples_all, self.signal_samples)),
    ]

    bins_mgg = np.linspace(100.0, 180.0, 41)
    bins_mjj = np.linspace(80.0, 190.0, 41)

    rows = []
    os.makedirs(self.output_dir, exist_ok=True)

    for cat in cats_for_plot:
      region_mask = masks[cat] & presel_mask
      print(f"[plot_mgg_mjj_correlation] {cat}: total {int(region_mask.sum())} MC events after presel")

      for gname, gsel in groups:
        mask = region_mask & gsel
        n_sel = int(mask.sum())
        print(f"  [plot_mgg_mjj_correlation] {cat} / {gname}: {n_sel} MC events")

        if n_sel == 0:
          continue

        x = self.mass_all[mask]
        y = self.dijet_mass_all[mask]
        w = self.weights_all[mask]

        corr = self.weighted_pearson_corr(x, y, w)
        rows.append({
          "Category": cat,
          "Group": gname,
          "N_events": n_sel,
          "Corr_weighted": corr,
        })

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        hist, xedges, yedges = np.histogram2d(x, y, bins=[bins_mgg, bins_mjj], weights=w)

        positive = hist[hist > 0]
        if positive.size > 0:
          mesh = ax.pcolormesh(
            xedges,
            yedges,
            hist.T,
            norm=LogNorm(vmin=max(np.min(positive), 1e-12), vmax=np.max(positive)),
            shading="auto",
          )
        else:
          mesh = ax.pcolormesh(
            xedges,
            yedges,
            hist.T,
            shading="auto",
          )

        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label("Weighted Events")

        self._apply_cms_axes(
          ax=ax,
          xlabel=r"$m_{\gamma\gamma}$ [GeV]",
          ylabel=r"$m_{jj}$ [GeV]",
          xlim=(100.0, 180.0),
          add_label=True,
        )

        title_corr = f"{corr:.4f}" if np.isfinite(corr) else "NaN"
        panel_text = f"{gname} | {cat}\nWeighted Pearson r = {title_corr}"
        self._add_panel_text(ax, panel_text, fontsize=13)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{out_prefix}_{gname}_{cat}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(self.output_dir, f"{out_prefix}_{gname}_{cat}.pdf"), bbox_inches="tight")
        plt.close(fig)
    if rows:
      corr_df = pd.DataFrame(rows).sort_values(["Category", "Group"]).reset_index(drop=True)
      out_csv = os.path.join(self.output_dir, f"{out_prefix}_coeffs.csv")
      corr_df.to_csv(out_csv, index=False)
      print(f"[plot_mgg_mjj_correlation] Saved correlation coefficients CSV: {out_csv}")
    else:
      print("[plot_mgg_mjj_correlation] No correlation rows produced.")


def save_table(df, csv_path, txt_path):
  df.to_csv(csv_path, index=False)

  with open(txt_path, "w") as f:
    f.write(
      df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        line_width=100000,
        float_format=lambda x: f"{x:.6f}" if pd.notnull(x) else "NaN",
      )
    )
    f.write("\n")



def run_quietly(func, *args, quiet=True, **kwargs):
  if not quiet:
    return func(*args, **kwargs)

  with contextlib.redirect_stdout(io.StringIO()):
    return func(*args, **kwargs)

def main():
  parser = argparse.ArgumentParser(
    description="Read best_cut_params.json / vbfhh_sr_info.json and produce category yield tables and MC-only plots."
  )
  parser.add_argument("--base_path", type=str, required=True, help="Base path to the input samples")
  parser.add_argument("--opt", type=str, required=True, help="Subdirectory under base_path containing best_cut_params.json")
  parser.add_argument(
    "--signal_samples",
    type=str,
    default="[GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00]",
    help="Bracket-enclosed comma-separated list of signal sample names"
  )
  parser.add_argument(
    "--vbfhh_samples",
    type=str,
    default=None,
    help="Bracket-enclosed comma-separated list of VBFHH signal sample names"
  )
  parser.add_argument("--signal_class", type=int, default=2, help="Index of ggHH signal class in the score array")
  parser.add_argument("--vbfhh_class", type=int, default=None, help="Index of VBFHH class in the score array")
  parser.add_argument("--bkg_classes", type=str, default="0,1", help="Comma-separated list of background class indices")
  parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override")
  parser.add_argument("--apply_boosted_veto", action="store_true", default=False)
  parser.add_argument(
    "--mgg_sr_low",
    type=float,
    default=120.0,
    help="Lower boundary of the mgg signal region. Default: 120.0",
  )
  parser.add_argument(
    "--mgg_sr_high",
    type=float,
    default=130.0,
    help="Upper boundary of the mgg signal region. Default: 130.0",
  )

  parser.add_argument("--cms_data", action="store_true", default=False, help="Use data-style CMS label")
  parser.add_argument("--cms_label", type=str, default="Private Work", help="Extra CMS label text")
  parser.add_argument("--lumi", type=float, default=419.5, help="Integrated luminosity in fb^-1")
  parser.add_argument("--com", type=str, default="13 / 13.6", help="Center-of-mass energy text")
  parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Print detailed progress messages. By default, only debug summaries and final output paths are printed.",
  )

  args = parser.parse_args()

  if args.bkg_classes.strip() == "":
    bkg_classes = []
  else:
    bkg_classes = [int(x.strip()) for x in args.bkg_classes.split(",")]

  quiet = not args.verbose

  maker = YieldTableMaker(
    base_path=args.base_path,
    opt=args.opt,
    output_dir=args.output_dir,
    signal_samples=args.signal_samples,
    vbfhh_samples=args.vbfhh_samples,
    apply_boosted_veto=args.apply_boosted_veto,
    mgg_sr_low=args.mgg_sr_low,
    mgg_sr_high=args.mgg_sr_high,
    cms_data=args.cms_data,
    cms_label=args.cms_label,
    cms_lumi=args.lumi,
    cms_com=args.com,
  )

  run_quietly(maker.load_samples, quiet=quiet)
  if quiet:
    maker.print_mc_debug_summary()

  run_quietly(maker.load_data_samples, quiet=quiet)
  if quiet:
    maker.print_data_debug_summary()

  df = run_quietly(
    maker.compute_yield_table,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  df_per_era = run_quietly(
    maker.compute_yield_table_per_era,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )
  df_sr_process_column = run_quietly(
    maker.compute_sr_process_column_yield_table,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  df_sr_process_column_per_era = run_quietly(
    maker.compute_sr_process_column_yield_table_per_era,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  out_dir = maker.output_dir
  os.makedirs(out_dir, exist_ok=True)

  save_table(
    df,
    os.path.join(out_dir, "yield_table.csv"),
    os.path.join(out_dir, "yield_table.txt"),
  )

  save_table(
    df_per_era,
    os.path.join(out_dir, "yield_table_per_era.csv"),
    os.path.join(out_dir, "yield_table_per_era.txt"),
  )

  save_table(
    df_sr_process_column,
    os.path.join(out_dir, "yield_table_process_by_sr.csv"),
    os.path.join(out_dir, "yield_table_process_by_sr.txt"),
  )

  save_table(
    df_sr_process_column_per_era,
    os.path.join(out_dir, "yield_table_process_by_sr_per_era.csv"),
    os.path.join(out_dir, "yield_table_process_by_sr_per_era.txt"),
  )

  print("\n[main] Saved output tables:")
  for filename in [
    "yield_table.csv",
    "yield_table.txt",
    "yield_table_per_era.csv",
    "yield_table_per_era.txt",
    "yield_table_process_by_sr.csv",
    "yield_table_process_by_sr.txt",
    "yield_table_process_by_sr_per_era.csv",
    "yield_table_process_by_sr_per_era.txt",
  ]:
    print(f"  - {os.path.join(out_dir, filename)}")

  run_quietly(
    maker.plot_sr_vs_preselection,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  run_quietly(
    maker.plot_presel_and_srs_separately,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  run_quietly(
    maker.plot_mgg_mjj_correlation,
    signal_class=args.signal_class,
    bkg_classes=bkg_classes,
    vbfhh_class=args.vbfhh_class,
    quiet=quiet,
  )

  print("[main] Plotting finished.")


if __name__ == "__main__":
  main()
