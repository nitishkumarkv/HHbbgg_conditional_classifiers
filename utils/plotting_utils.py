import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import yaml

hep.style.use("CMS")


def parse_best_cut_params(best_cut_json, max_gghh_cats=3):
  if best_cut_json is None or (not os.path.exists(best_cut_json)):
    print(f"[SR] best_cut_params.json not found: {best_cut_json}")
    return []

  with open(best_cut_json, "r") as f:
    best_params = json.load(f)

  if not isinstance(best_params, list):
    raise ValueError(
      f"{best_cut_json} does not contain a list. "
      "Please pass best_cut_params.json, not best_cut_params.txt."
    )

  return best_params[:max_gghh_cats]


def parse_vbfhh_info(vbfhh_info_json, max_vbfhh_cats=1):
  if vbfhh_info_json is None or (not os.path.exists(vbfhh_info_json)):
    return []

  with open(vbfhh_info_json, "r") as f:
    vbfhh_info = json.load(f)

  if not isinstance(vbfhh_info, list):
    raise ValueError(f"{vbfhh_info_json} does not contain a list.")

  return vbfhh_info[:max_vbfhh_cats]


def unique_keep_order(seq):
  out = []
  seen = set()

  for x in seq:
    if x not in seen:
      out.append(x)
      seen.add(x)

  return out


def find_scored_sample_dirs(base_dir):
  if not os.path.isdir(base_dir):
    return []

  sample_dirs = []

  for name in sorted(os.listdir(base_dir)):
    sample_dir = os.path.join(base_dir, name)

    if not os.path.isdir(sample_dir):
      continue

    events_path = os.path.join(sample_dir, "events.parquet")
    score_path = os.path.join(sample_dir, "y.npy")

    if os.path.exists(events_path) and os.path.exists(score_path):
      sample_dirs.append((name, events_path, score_path))

  return sample_dirs


def build_full_path(base_path, relative_path_or_list):
  if isinstance(relative_path_or_list, list):
    return [
      os.path.join(base_path, rel_path)
      for rel_path in relative_path_or_list
    ]

  return os.path.join(base_path, relative_path_or_list)


def load_parquet_from_path_or_list(path_or_list, columns=None):
  if isinstance(path_or_list, list):
    parts = []

    for path in path_or_list:
      if not os.path.exists(path):
        print(f"[load] Missing parquet path: {path}. Skip this part.")
        continue

      parts.append(ak.from_parquet(path, columns=columns))

    if len(parts) == 0:
      return None

    return ak.concatenate(parts, axis=0)

  if not os.path.exists(path_or_list):
    print(f"[load] Missing parquet path: {path_or_list}")
    return None

  return ak.from_parquet(path_or_list, columns=columns)


def data_sample_aliases(sample_name):
  alias_groups = [
    [
      "preEE",
      "2022preEE",
      "2022_preEE",
      "2022_EraC",
      "2022_EraD",
    ],
    [
      "postEE",
      "2022postEE",
      "2022_postEE",
      "2022_EraE",
      "2022_EraF",
      "2022_EraG",
    ],
    [
      "preBPix",
      "2023preBPix",
      "2023_preBPix",
      "2023_EraC",
    ],
    [
      "postBPix",
      "2023postBPix",
      "2023_postBPix",
      "2023_EraD",
    ],
  ]

  aliases = [sample_name]

  for group in alias_groups:
    if sample_name in group:
      aliases += group
      break

  return unique_keep_order(aliases)


def resolve_scored_data_dirs(data_folder, data_sample_names):
  scored_dirs = {
    name: (events_path, score_path)
    for name, events_path, score_path in find_scored_sample_dirs(data_folder)
  }

  if len(scored_dirs) == 0:
    return []

  resolved = []
  used_dirs = set()

  for data_sample in data_sample_names:
    if data_sample in scored_dirs:
      aliases_to_try = [data_sample]
    else:
      aliases_to_try = [
        alias
        for alias in data_sample_aliases(data_sample)
        if alias != data_sample
      ]

    for alias in aliases_to_try:
      if alias not in scored_dirs:
        continue

      if alias in used_dirs:
        continue

      events_path, score_path = scored_dirs[alias]
      resolved.append((alias, events_path, score_path, data_sample))
      used_dirs.add(alias)

  if len(resolved) == 0:
    print(
      f"[data] Found scored data directories in {data_folder}, "
      "but none matched samples_info.data keys or known era aliases."
    )

  return resolved


def is_vbfpair_vbf_variable(variable):
  return variable.startswith("nonResReg_vbfpair_VBF")


def build_sr_masks_for_events(
  events,
  class_names,
  signal_class_idx=None,
  bkg_class_indices=None,
  best_cut_params=None,
  vbfhh_class_idx=None,
  vbfhh_info_list=None,
  boosted_field="is_boosted",
  include_boosted_cat=True,
):
  """
  Build exclusive category masks for a single awkward array of events.

  Exclusivity order:
    1) boosted_cat
    2) vbfhh_cat*
    3) ggHH cat*
  """

  if events is None or len(events) == 0:
    return {}

  if best_cut_params is None:
    best_cut_params = []

  if vbfhh_info_list is None:
    vbfhh_info_list = []

  if bkg_class_indices is None:
    bkg_class_indices = []

  required_score_fields = []

  if len(best_cut_params) > 0 and signal_class_idx is not None:
    required_score_fields.append(class_names[signal_class_idx])
    required_score_fields += [class_names[i] for i in bkg_class_indices]

  if len(vbfhh_info_list) > 0 and vbfhh_class_idx is not None:
    required_score_fields.append(class_names[vbfhh_class_idx])

  required_score_fields = unique_keep_order(required_score_fields)

  for field in required_score_fields:
    if field not in events.fields:
      print(f"[SR] Missing score field '{field}' in events. Skip SR mask building.")
      return {}

  masks = {}
  already_used = np.zeros(len(events), dtype=bool)

  # -------------------------------------------------
  # 1) boosted category first
                                                
  # -------------------------------------------------
  if include_boosted_cat:
    if boosted_field in events.fields:
      boosted_mask = ak.to_numpy(
        ak.fill_none(events[boosted_field], False)
      ).astype(bool)

      boosted_mask = boosted_mask & (~already_used)
      masks["boosted_cat"] = boosted_mask
      already_used |= boosted_mask
    else:
      print(f"[SR] boosted field '{boosted_field}' not found. Skip boosted_cat.")

  # -------------------------------------------------
  # 2) VBF categories next
                                               
  # -------------------------------------------------
  if vbfhh_class_idx is not None and len(vbfhh_info_list) > 0:
    for i_cat, info in enumerate(vbfhh_info_list, start=1):
      threshold = info["best_threshold"]

      base_mask = ak.to_numpy(events[class_names[vbfhh_class_idx]]) > threshold
      cat_mask = base_mask & (~already_used)

      masks[f"vbfhh_cat{i_cat}"] = cat_mask
      already_used |= cat_mask

  # -------------------------------------------------
  # 3) ggHH categories last
  # -------------------------------------------------
  if signal_class_idx is not None and len(best_cut_params) > 0 and len(bkg_class_indices) > 0:
    for i_cat, params in enumerate(best_cut_params, start=1):
      base_mask = ak.to_numpy(events[class_names[signal_class_idx]]) > params["th_signal"]

      for b in bkg_class_indices:
        base_mask &= ak.to_numpy(events[class_names[b]]) < params[f"th_bg_{b}"]

      cat_mask = base_mask & (~already_used)

      masks[f"cat{i_cat}"] = cat_mask
      already_used |= cat_mask

  return masks


def plot_stacked_histogram(
  samples_info,
  sim_folder,
  data_folder,
  sim_samples,
  variables,
  out_path,
  bins=40,
  mass_window=(115, 135),
  mjj_mass_window=(110, 140),
  signal_scale=100,
  only_MC=False,
  var_prefix="nonResReg",
  best_cut_json=None,
  vbfhh_info_json=None,
  signal_class_idx=None,
  bkg_class_indices=None,
  vbfhh_class_idx=None,
  max_gghh_cats=3,
  max_vbfhh_cats=1,
  boosted_field="is_boosted",
  make_inclusive_plots=True,
  make_score_process_plots=True,
  make_mgg_sideband_preselection_plots=True,
  make_sr_plots=True,
  mc_percentage=100.0,
):
  """
  Load data first, then loop over variables to plot stacked histograms with MC and Data.

    boosted -> VBF -> ggHH
  """

  if not 0.0 < mc_percentage <= 100.0:
    raise ValueError("mc_percentage must be greater than 0 and at most 100")

  mc_weight_scale = 100.0 / mc_percentage
  print(
    f"[MC normalization] Using {mc_percentage:g}% of MC events; "
    f"scaling MC weights by {mc_weight_scale:g}."
  )

  if only_MC:
    out_path = os.path.join(out_path, "MC_Plots_preselection")
  else:
    out_path = os.path.join(out_path, "Data_MC_Plots_preselection")

  os.makedirs(out_path, exist_ok=True)

  mc_colors = [
    "#FF8A50",
    "#FFB300",
    "#66BB6A",
    "#42A5F5",
    "#AB47BC",
    "#C0CA33",
    "#26A69A",
    "blue",
    "red",
    "#795548",
    "#757575",
    "#66BB6A",
  ]

  label_dict = {
    "GGJets": "GGJets",
    "GJetPt20To40": "GJetPt20To40",
    "GJetPt40": "GJetPt40",
    "TTGG": "TTGG",
    "ttHtoGG_M_125": "ttH",
    "BBHto2G_M_125": "bbH",
    "GluGluHToGG_M_125": "ggH",
    "VBFHToGG_M_125": "VBFH",
    "VHtoGG_M_125": "VH",
    "WmHtoGG_M_125": "VH",
    "WpHtoGG_M_125": "VH",
    "ZHtoGG_M_125": "VH",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "ggHH SM",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "ggHH kl=5.00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "ggHH kl=0.00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "ggHH kl=2.45",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000": "VBFHH",
    "DDQCDGJET": "DDQCDGJets",
    "TTG_10_100": "TTG_10_100",
    "TTG_100_200": "TTG_100_200",
    "TTG_200": "TTG_200",
    "TT": "TT",
  }

  stack_mc_dict = {}
  signal_mc_dict = {}

  def deltaR(eta1, phi1, eta2, phi2, fill_none=True):
    valid = (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999)

    eta1 = ak.mask(eta1, valid)
    phi1 = ak.mask(phi1, valid)
    eta2 = ak.mask(eta2, valid)
    phi2 = ak.mask(phi2, valid)

    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    delta_r = np.sqrt(deta**2 + dphi**2)

    if fill_none:
      return ak.fill_none(delta_r, -999.0)

    return delta_r

  def add_var(events):
    hh_mass = events[f"{var_prefix}_HHbbggCandidate_mass"]
    dijet_mass_reg = events[f"{var_prefix}_dijet_mass_DNNreg"]

    valid_hh_mass = hh_mass > 0
    valid_dijet_mass = dijet_mass_reg > 0

    events["diphoton_PtOverM_ggjj"] = ak.where(
      valid_hh_mass,
      events.pt / hh_mass,
      -999.0,
    )

    events[f"{var_prefix}_dijet_PtOverM_ggjj"] = ak.where(
      valid_hh_mass,
      events[f"{var_prefix}_dijet_pt"] / hh_mass,
      -999.0,
    )

    events[f"{var_prefix}_lead_bjet_over_M_regressed"] = ak.where(
      valid_dijet_mass,
      events[f"{var_prefix}_lead_bjet_pt"] / dijet_mass_reg,
      -999.0,
    )

    events[f"{var_prefix}_sublead_bjet_over_M_regressed"] = ak.where(
      valid_dijet_mass,
      events[f"{var_prefix}_sublead_bjet_pt"] / dijet_mass_reg,
      -999.0,
    )

    events["deltaR_gg"] = deltaR(
      events.lead_eta,
      events.lead_phi,
      events.sublead_eta,
      events.sublead_phi,
    )

    return events

  def add_preselection(events):
    mass_bool = (events.mass > 100) & (events.mass < 180)

    dijet_mass_bool = (
      (events[f"{var_prefix}_dijet_mass_DNNreg"] > 80)
      & (events[f"{var_prefix}_dijet_mass_DNNreg"] < 190)
    )
                                                    
                                                          

    lead_mvaID_bool = events.lead_mvaID > -0.7
    sublead_mvaID_bool = events.sublead_mvaID > -0.7

    events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]
    events = add_var(events)

    return events

  def sanitize_filename(name):
    safe_name = str(name)

    for old, new in [
      (" ", "_"),
      ("/", "_"),
      ("\\", "_"),
      ("=", "eq"),
      (":", "_"),
      (";", "_"),
      (",", "_"),
      ("(", ""),
      (")", ""),
    ]:
      safe_name = safe_name.replace(old, new)

    return safe_name


  def plot_score_for_each_process(
    process_dict,
    score_names,
    output_dir,
    use_weights=True,
  ):
    """
    Plot every classifier score separately for each MC process.

    Output structure:
      output_dir/
        process_name/
          class_score_log.png
          class_score_linear.png
    """

    os.makedirs(output_dir, exist_ok=True)

    luminosities = {
      "2016preVFP": 19.5,
      "2016postVFP": 16.8,
      "2017": 42.07,
      "2018": 59.56,
      "preEE": 7.98,
      "postEE": 26.67,
      "preBPix": 17.79,
      "postBPix": 9.45,
      "2024": 108.95,
      "2025": 110.73  # Integrated luminosity in fb^-1
    }

    lumi = 0.0

    for era_ in eras:
      if era_ in luminosities:
        lumi += luminosities[era_]

    bin_edges = np.linspace(0.0, 1.0, 21)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for process_label, events in process_dict.items():
      if events is None or len(events) == 0:
        continue

      process_dir = os.path.join(output_dir, sanitize_filename(process_label))
      os.makedirs(process_dir, exist_ok=True)

      for score_name in score_names:
        if score_name not in events.fields:
          print(f"[score plot] {score_name} not found for {process_label}. Skip.")
          continue

        values = ak.to_numpy(events[score_name])
        valid = np.isfinite(values)

        if use_weights and "weight_tot" in events.fields:
          weights = ak.to_numpy(events["weight_tot"])
          valid = valid & np.isfinite(weights)
        else:
          weights = np.ones(len(values), dtype=float)

        values = values[valid]
        weights = weights[valid]

        score_valid = (values >= 0.0) & (values <= 1.0)
        values = values[score_valid]
        weights = weights[score_valid]

        if len(values) == 0:
          print(f"[score plot] No valid entries for {process_label}, {score_name}. Skip.")
          continue

        hist, _ = np.histogram(
          values,
          bins=bin_edges,
          weights=weights,
        )

        hist_err, _ = np.histogram(
          values,
          bins=bin_edges,
          weights=weights**2,
        )

        hist_err = np.sqrt(hist_err)
        max_ref = max(float(np.max(hist)), 1.0)

        for scale_name, use_log in [
          ("log", True),
          ("linear", False),
        ]:
          fig, ax = plt.subplots(figsize=(10, 8))

          hep.cms.label(
            data=False,
            lumi=lumi,
            ax=ax,
            loc=0,
            fontsize=16,
            label="Private Work",
            com="13 / 13.6",
          )

          ax.step(
            centers,
            hist,
            where="mid",
            linewidth=2.0,
            label=process_label,
          )

          ax.fill_between(
            centers,
            hist - hist_err,
            hist + hist_err,
            alpha=0.35,
            step="mid",
          )

          ax.set_xlabel(score_name)
          ax.set_ylabel("Weighted events" if use_weights else "Events")
          ax.set_xlim(0.0, 1.0)
          ax.set_title(f"{process_label}: {score_name}")
          ax.legend(fontsize=14)

          if use_log:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 100.0 * max_ref)
          else:
            ax.set_yscale("linear")
            ax.set_ylim(0.0, 1.4 * max_ref)

          output_file = os.path.join(
            process_dir,
            f"{sanitize_filename(score_name)}_{scale_name}.png",
          )

          plt.savefig(output_file, dpi=300, bbox_inches="tight")
          plt.close(fig)

    print("[score plot] Per-process score plots saved in", output_dir)


  def select_mgg_sideband_events(events, label="events"):
    """Keep only mgg sideband events outside the Higgs signal window."""
    if events is None:
      return None

    if "mass" not in events.fields:
      print(
        f"[mgg sideband] {label} does not contain 'mass'. "
        "Return an empty sample for the sideband plot."
      )
      return events[:0]

    sideband_low, sideband_high = mass_window
    mgg = ak.to_numpy(events["mass"])
    keep_sideband = np.isfinite(mgg) & (
      (mgg <= sideband_low) | (mgg >= sideband_high)
    )

    return events[keep_sideband]


  def blind_data_events_in_mgg_window(data_events):
    """Remove data events inside the mgg signal window before making any data histogram."""
    return select_mgg_sideband_events(data_events, label="Data")


  def make_single_plot(
    variable,
    data_events,
    stack_dict,
    signal_dict,
    output_file,
    title_extra=None,
    force_log=None,
    blind_data=False,
  ):
    shape_only_vbf = is_vbfpair_vbf_variable(variable)

    if variable not in var_config:
      min_ = 0.0
      max_ = 0.0

      for _, data_ in stack_dict.items():
        if len(data_) == 0 or variable not in data_.fields:
          continue
                                                
                                  
                                                                    
                                                                

        values = ak.to_numpy(data_[variable])
        values = values[np.isfinite(values)]
        values = values[values != -999]
                 
                                                                                                          

        if len(values) == 0:
          continue

        min_ = min(min_, float(np.min(values)))
        max_ = max(max_, float(np.max(values)))

      if min_ == max_:
        max_ = min_ + 1.0

      var_config[variable] = {
        "label": variable,
        "bins": 30,
        "range": (min_, max_),
        "log": True,
      }

    bin_edges = np.linspace(
      *var_config[variable]["range"],
      var_config[variable]["bins"] + 1,
    )

    mc_hist = []
    mc_err = np.zeros(len(bin_edges) - 1)
    mc_labels = []
    mc_colors_used = []

    for i, (sample, data_) in enumerate(stack_dict.items()):
      if len(data_) == 0 or variable not in data_.fields:
        continue

      values = ak.to_numpy(data_[variable])
      weights = ak.to_numpy(data_["weight_tot"])

      valid = np.isfinite(values) & np.isfinite(weights)
      values = values[valid]
      weights = weights[valid]

      if shape_only_vbf:
        valid_vbf = values != -999
        values = values[valid_vbf]
        weights = weights[valid_vbf]

      if len(values) == 0:
        continue

      hist, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=weights,
      )
                                
                                    
                                                                

      hist_err, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=weights**2,
      )

      mc_hist.append(hist)
      mc_err += hist_err
      mc_labels.append(sample)
      mc_colors_used.append(mc_colors[i % len(mc_colors)])

    if len(mc_hist) == 0:
      print(f"[plot] No MC entries for {variable} in {output_file}. Skip.")
      return

    mc_total = np.sum(mc_hist, axis=0)
    mc_err = np.sqrt(mc_err)

    if blind_data:
      data_events_for_plot = blind_data_events_in_mgg_window(data_events)
    else:
      data_events_for_plot = data_events

    if data_events_for_plot is not None and variable in data_events_for_plot.fields:
      data_values = ak.to_numpy(data_events_for_plot[variable])
      data_values = data_values[np.isfinite(data_values)]

      if shape_only_vbf:
        data_values = data_values[data_values != -999]

      data_hist, _ = np.histogram(
        data_values,
        bins=bin_edges,
      )

      data_err = np.sqrt(data_hist).astype(float)
      data_hist = data_hist.astype(float)

                                        
                            
                                        
                                                                                

                                      
                                     
    else:
      data_hist = None
      data_err = None

    signal_color_list = [
      "red",
      "green",
      "blue",
      "purple",
      "brown",
      "magenta",
    ]

    signal_histograms = {}

    for signal, data_ in signal_dict.items():
      if len(data_) == 0 or variable not in data_.fields:
        continue

      values = ak.to_numpy(data_[variable])
      weights = ak.to_numpy(data_["weight_tot"])

      valid = np.isfinite(values) & np.isfinite(weights)
      values = values[valid]
      weights = weights[valid]

      if shape_only_vbf:
        valid_vbf = values != -999
        values = values[valid_vbf]
        weights = weights[valid_vbf]

      if len(values) == 0:
        continue

      hist, _ = np.histogram(
        values,
        bins=bin_edges,
        weights=weights,
      )

      if "ggHH" in signal:
        signal_histograms[signal] = hist * signal_scale
      else:
        signal_histograms[signal] = hist * signal_scale * 10

    if shape_only_vbf:
      print(
        f"[VBF variable] {variable}: removed -999 from data and MC; "
        "kept the cross-section/luminosity MC normalization."
      )

    if only_MC:
      fig, ax = plt.subplots(figsize=(10, 10))
    else:
      fig, axs = plt.subplots(
        2,
        1,
        gridspec_kw={
          "height_ratios": [3, 1],
          "hspace": 0.05,
        },
        figsize=(10, 10),
        sharex=True,
      )

      ax, ax_ratio = axs

    luminosities = {
      "2016preVFP": 19.5,
      "2016postVFP": 16.8,
      "2017": 42.07,
      "2018": 59.56,
      "preEE": 7.98,
      "postEE": 26.67,
      "preBPix": 17.79,
      "postBPix": 9.45,
      "2024": 108.95,
    }

    lumi = 0.0

    for era_ in eras:
      if era_ in luminosities:
        lumi += luminosities[era_]

    hep.cms.label(
      data=True,
      lumi=lumi,
      ax=ax,
      loc=0,
      fontsize=16,
      label="Private Work",
      com="13 / 13.6",
    )

    hep.histplot(
      mc_hist,
      bin_edges,
      histtype="fill",
      stack=True,
      label=mc_labels,
      color=mc_colors_used,
      edgecolor="black",
      ax=ax,
    )

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.fill_between(
      centers,
      mc_total - mc_err,
      mc_total + mc_err,
      color="gray",
      alpha=0.5,
      step="mid",
    )

    if (not only_MC) and (data_hist is not None):
      ax.errorbar(
        centers,
        data_hist,
        yerr=data_err,
        fmt="o",
        color="black",
        label="Data",
        markersize=5,
      )

    color_idx = 0

    for signal, hist in signal_histograms.items():
      if "ggHH" in signal:
        ax.step(
          centers,
          hist,
          where="mid",
          linestyle="dashed",
          linewidth=2.0,
          color=signal_color_list[color_idx % len(signal_color_list)],
          label=f"{signal} x {signal_scale}",
        )

        color_idx += 1
      else:
        signal_scale_ = signal_scale * 10

        ax.step(
          centers,
          hist,
          where="mid",
          linestyle="dashed",
          linewidth=2.0,
          color="orange",
          label=f"{signal} x {signal_scale_}",
        )

    if variable == "mass":
      blind_low, blind_high = mass_window

      ax.axvspan(
        blind_low,
        blind_high,
        color="gray",
        alpha=0.15,
        hatch="//",
      )

      if not only_MC:
        ax_ratio.axvspan(
          blind_low,
          blind_high,
          color="gray",
          alpha=0.15,
          hatch="//",
        )

    if title_extra is not None:
      ax.set_title(title_extra)

    ax.legend(fontsize=20, ncol=2)
    ax.set_ylabel("Events")
    ax.set_xlim(var_config[variable]["range"])

    mc_max = np.max(mc_total) if len(mc_total) > 0 else 0.0

    if data_hist is not None and len(data_hist) > 0:
      finite_data = data_hist[np.isfinite(data_hist)]
      data_max = np.max(finite_data) if len(finite_data) > 0 else 0.0
    else:
      data_max = 0.0

    max_ref = max(mc_max, data_max, 1.0)

    use_log = var_config[variable]["log"] if force_log is None else force_log

    if use_log:
      ax.set_yscale("log")
      ax.set_ylim(0.1, 600 * max_ref)
    else:
      ax.set_yscale("linear")
      ax.set_ylim(0, 1.7 * max_ref)

    if not only_MC:
      safe_ratio = np.divide(
        data_hist,
        mc_total,
        out=np.full_like(mc_total, np.nan, dtype=float),
        where=(mc_total > 0),
      )

      data_ratio_err = np.divide(
        data_err,
        mc_total,
        out=np.full_like(mc_total, np.nan, dtype=float),
        where=(mc_total > 0),
      )

      mc_ratio_err = np.divide(
        mc_err,
        mc_total,
        out=np.full_like(mc_total, np.nan, dtype=float),
        where=(mc_total > 0),
      )

      ax_ratio.errorbar(
        centers,
        safe_ratio,
        yerr=data_ratio_err,
        fmt="o",
        color="black",
        markersize=5,
      )

      ax_ratio.fill_between(
        centers,
        1 - mc_ratio_err,
        1 + mc_ratio_err,
        color="gray",
        alpha=0.3,
        hatch="xx",
        edgecolor="black",
        linewidth=0.0,
        step="mid",
      )

      ax_ratio.axhline(1, linestyle="dashed", color="gray")
      ax_ratio.set_ylim(0.5, 1.5)
      ax_ratio.set_ylabel("Data / MC")
      ax_ratio.set_xlabel(var_config[variable]["label"])

      plt.savefig(output_file, dpi=300, bbox_inches="tight")
      plt.close(fig)
    else:
      ax.set_xlabel(var_config[variable]["label"])

      plt.savefig(output_file, dpi=300, bbox_inches="tight")
      plt.close(fig)

  class_names_raw = training_config["classes"]
  class_names = [f"{class_name}_score" for class_name in class_names_raw]

  print(f"Loaded class names from config: {class_names}")

  events_path = samples_info["samples_path"]
  data_combined = None
  data_samples = training_config["samples_info"]["data"]

  data_columns = unique_keep_order(
    variables
    + [
      "lead_isScEtaEB",
      "lead_isScEtaEE",
      "sublead_isScEtaEB",
      "sublead_isScEtaEE",
      boosted_field,
    ]
  )

  scored_data_dirs = resolve_scored_data_dirs(data_folder, data_samples.keys())

  if len(scored_data_dirs) > 0:
    print(f"[data] Loading scored data directories from {data_folder}")

    for data_sample, data_events_path, data_score_path, config_data_sample in scored_data_dirs:
      print(
        f"[data] Loading {data_sample} for config key {config_data_sample}: "
        f"{data_events_path}"
      )

      data_part = ak.from_parquet(
        data_events_path,
        columns=data_columns,
      )

      data_score = np.load(data_score_path)

      if len(data_part) != data_score.shape[0]:
        raise RuntimeError(
          f"Event / score length mismatch for data {data_sample}: "
          f"len(events) = {len(data_part)}, scores.shape[0] = {data_score.shape[0]}. "
          "Check whether events.parquet and y.npy come from the same production."
        )

      print(f"[data] {data_sample}: events={len(data_part)}, scores={data_score.shape}")

      num_classes = data_score.shape[1]

      for i, class_name in enumerate(class_names):
        if i < num_classes:
          data_part[class_name] = data_score[:, i]

      if data_combined is None:
        data_combined = data_part
      else:
        data_combined = ak.concatenate([data_combined, data_part], axis=0)
  else:
    print(
      f"[data] No scored data directories with events.parquet and y.npy found in {data_folder}. "
      "Fall back to samples_info.data raw paths."
    )

    for data_sample, path in data_samples.items():
      raw_data_path = build_full_path(events_path, path)
      data_part = load_parquet_from_path_or_list(raw_data_path, columns=data_columns)

      if data_part is None:
        print(f"[data] Could not load raw data for {data_sample}. Skip.")
        continue

      print(f"[data] Loaded raw data {data_sample}: {len(data_part)}")

      if data_combined is None:
        data_combined = data_part
      else:
        data_combined = ak.concatenate([data_combined, data_part], axis=0)

  if data_combined is None:
    raise RuntimeError(
      f"No data events were loaded. Check {data_folder} or samples_info.data in the config."
    )

  if "minMVAID" in variables:
    data_combined["minMVAID"] = np.min(
      [
        data_combined.lead_mvaID,
        data_combined.sublead_mvaID,
      ],
      axis=0,
    )

    data_combined["maxMVAID"] = np.max(
      [
        data_combined.lead_mvaID,
        data_combined.sublead_mvaID,
      ],
      axis=0,
    )

  data_combined = add_preselection(data_combined)

  print("before: ", len(data_combined))

  eras = samples_info["eras"]

  mc_columns = unique_keep_order(
    variables
    + [
      "lead_isScEtaEB",
      "lead_isScEtaEE",
      "sublead_isScEtaEB",
      "sublead_isScEtaEE",
      "lead_genPartFlav",
      "sublead_genPartFlav",
      "weight_tot",
      boosted_field,
    ]
  )

  for sample in sim_samples:
    sample_combined = []

    for era in eras:
      boosted_mc_path = f"{sim_folder}/{era}/{sample}/events.parquet"
      score_path = f"{sim_folder}/{era}/{sample}/y.npy"

      if not os.path.exists(boosted_mc_path):
        print(f"[mc] Missing scored parquet: {boosted_mc_path}. Skip {era}/{sample}.")
        continue

      if not os.path.exists(score_path):
        print(f"[mc] Missing score file: {score_path}. Skip {era}/{sample}.")
        continue

      events_ = ak.from_parquet(
        boosted_mc_path,
        columns=mc_columns,
      )

      scores_ = np.load(score_path)

      if len(events_) != scores_.shape[0]:
        raise RuntimeError(
          f"Event / score length mismatch for MC {era}/{sample}: "
          f"len(events) = {len(events_)}, scores.shape[0] = {scores_.shape[0]}. "
          "Check whether events.parquet and y.npy come from the same production."
        )

      for i, class_name in enumerate(class_names):
        if i < scores_.shape[1]:
          events_[class_name] = scores_[:, i]

      sample_combined.append(events_)

    if len(sample_combined) == 0:
      print(f"[mc] No scored events loaded for {sample}. Skip sample.")
      continue

    sample_combined = ak.concatenate(sample_combined, axis=0)

    if "minMVAID" in variables:
      sample_combined["minMVAID"] = np.min(
        [
          sample_combined.lead_mvaID,
          sample_combined.sublead_mvaID,
        ],
        axis=0,
      )

      sample_combined["maxMVAID"] = np.max(
        [
          sample_combined.lead_mvaID,
          sample_combined.sublead_mvaID,
        ],
        axis=0,
      )

    sample_combined = add_preselection(sample_combined)
    sample_combined["weight_tot"] = (
      sample_combined["weight_tot"] * mc_weight_scale
    )

    if sample == "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
      print(
        "number of events in GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        sum(sample_combined["weight_tot"]),
      )

    if sample in [
      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
      "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
      "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
      "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
      "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
    ]:
      signal_mc_dict[label_dict[sample]] = sample_combined
    else:
      stack_mc_dict[label_dict[sample]] = sample_combined

  variables = class_names + variables
  variables = unique_keep_order(variables)

  var_config = {
    "mass": {
      "label": r"$m_{\gamma\gamma}$ [GeV]",
      "bins": 40,
      "range": (100, 180),
      "log": True,
    },
    "nonRes_dijet_mass": {
      "label": r"$m_{jj}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    "dijet_mass": {
      "label": r"$m_{jj}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    "nonRes_mjj_regressed": {
      "label": r"$m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    f"{var_prefix}_dijet_mass": {
      "label": r"$m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    f"{var_prefix}_dijet_mass_DNNreg": {
      "label": r"$m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    f"{var_prefix}_DNNpair_dijet_mass": {
      "label": r"$m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    f"{var_prefix}_DNNpair_dijet_mass_DNNreg": {
      "label": r"$m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    "Res_mjj_regressed": {
      "label": r"Resonant $m_{jj}^{reg}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    "Res_dijet_mass": {
      "label": r"Resonant $m_{jj}$ [GeV]",
      "bins": 30,
      "range": (80, 180),
      "log": True,
    },
    "minMVAID": {
      "label": "minMVAID",
      "bins": 30,
      "range": (-0.7, 1),
      "log": True,
    },
    "maxMVAID": {
      "label": "maxMVAID",
      "bins": 30,
      "range": (-0.7, 1),
      "log": True,
    },
    "n_jets": {
      "label": "n_jets",
      "bins": 10,
      "range": (0, 10),
      "log": False,
    },
    "sublead_eta": {
      "label": "sublead_eta",
      "bins": 30,
      "range": (-3.2, 3.2),
      "log": False,
    },
    "lead_eta": {
      "label": "lead_eta",
      "bins": 30,
      "range": (-3.2, 3.2),
      "log": True,
    },
    "sublead_pt": {
      "label": "sublead_pt [GeV]",
      "bins": 30,
      "range": (0, 200),
      "log": True,
    },
    "lead_pt": {
      "label": "lead_pt [GeV]",
      "bins": 30,
      "range": (0, 200),
      "log": True,
    },
    "pt": {
      "label": "Diphoton pt [GeV]",
      "bins": 30,
      "range": (0, 400),
      "log": True,
    },
    "eta": {
      "label": "Diphoton eta",
      "bins": 30,
      "range": (-3.2, 3.2),
      "log": False,
    },
    "lead_mvaID": {
      "label": "lead_mvaID",
      "bins": 30,
      "range": (-0.7, 1),
      "log": True,
    },
    "sublead_mvaID": {
      "label": "sublead_mvaID",
      "bins": 30,
      "range": (-0.7, 1),
      "log": True,
    },
    "diphoton_PtOverM_ggjj": {
      "label": "diphoton_PtOverM_ggjj",
      "bins": 30,
      "range": (0, 2.1),
      "log": True,
    },
    f"{var_prefix}_dijet_PtOverM_ggjj": {
      "label": f"{var_prefix}_dijet_PtOverM_ggjj",
      "bins": 30,
      "range": (0, 2.1),
      "log": True,
    },
    f"{var_prefix}_lead_bjet_over_M_regressed": {
      "label": f"{var_prefix}_lead_bjet_over_M_regressed",
      "bins": 30,
      "range": (0, 2.1),
      "log": True,
    },
    f"{var_prefix}_sublead_bjet_over_M_regressed": {
      "label": f"{var_prefix}_sublead_bjet_over_M_regressed",
      "bins": 30,
      "range": (0, 2.1),
      "log": True,
    },
    "deltaR_gg": {
      "label": r"$\Delta R_{\gamma\gamma}$",
      "bins": 30,
      "range": (0, 5),
      "log": True,
    },
  }

  for class_name in class_names:
    var_config[class_name] = {
      "label": class_name,
      "bins": 30,
      "range": (0, 1),
      "log": True,
    }

  if make_inclusive_plots:
    # =========================
    # Per-process score plotting
    # =========================
    if make_score_process_plots:
      all_mc_process_dict = {}
      all_mc_process_dict.update(stack_mc_dict)
      all_mc_process_dict.update(signal_mc_dict)

      score_process_out_dir = os.path.join(out_path, "Score_Plots_By_Process")

      plot_score_for_each_process(
        process_dict=all_mc_process_dict,
        score_names=class_names,
        output_dir=score_process_out_dir,
        use_weights=True,
      )
    else:
      print("[score plot] Score_Plots_By_Process disabled by switch.")

    # =========================
    # Inclusive plotting
    # =========================
    for variable in variables:
      if variable not in data_combined.fields:
        print(f"Variable {variable} not found in data fields. Skipping...")
        continue

      output_file_log = f"{out_path}/{variable}_log.png"

      make_single_plot(
        variable=variable,
        data_events=None if only_MC else data_combined,
        stack_dict=stack_mc_dict,
        signal_dict=signal_mc_dict,
        output_file=output_file_log,
        force_log=True,
        blind_data=False,
      )

      output_file_linear = f"{out_path}/{variable}_linear.png"

      make_single_plot(
        variable=variable,
        data_events=None if only_MC else data_combined,
        stack_dict=stack_mc_dict,
        signal_dict=signal_mc_dict,
        output_file=output_file_linear,
        force_log=False,
        blind_data=False,
      )

    if make_mgg_sideband_preselection_plots and (not only_MC):
      # =========================
      # mgg sideband Data/MC preselection plotting
      # =========================
      mgg_sideband_out_path = f"{out_path}_mggSideband"
      os.makedirs(mgg_sideband_out_path, exist_ok=True)

      data_sideband = select_mgg_sideband_events(data_combined, label="Data")

      stack_sideband_dict = {
        sample_label: select_mgg_sideband_events(events, label=sample_label)
        for sample_label, events in stack_mc_dict.items()
      }

      signal_sideband_dict = {
        sample_label: select_mgg_sideband_events(events, label=sample_label)
        for sample_label, events in signal_mc_dict.items()
      }

      sideband_low, sideband_high = mass_window
      sideband_title = (
        f"mgg sideband preselection: "
        f"mgg <= {sideband_low} or mgg >= {sideband_high} GeV"
      )

      print(
        f"[mgg sideband] Data events after sideband selection: "
        f"{0 if data_sideband is None else len(data_sideband)}"
      )

      for variable in variables:
        has_variable = False

        if data_sideband is not None and variable in data_sideband.fields:
          has_variable = True

        for _, evs in stack_sideband_dict.items():
          if variable in evs.fields:
            has_variable = True
            break

        if not has_variable:
          for _, evs in signal_sideband_dict.items():
            if variable in evs.fields:
              has_variable = True
              break

        if not has_variable:
          print(f"[mgg sideband] Variable {variable} not found. Skip.")
          continue

        output_file_log = os.path.join(
          mgg_sideband_out_path,
          f"{variable}_log.png",
        )

        make_single_plot(
          variable=variable,
          data_events=data_sideband,
          stack_dict=stack_sideband_dict,
          signal_dict=signal_sideband_dict,
          output_file=output_file_log,
          title_extra=sideband_title,
          force_log=True,
          blind_data=False,
        )

        output_file_linear = os.path.join(
          mgg_sideband_out_path,
          f"{variable}_linear.png",
        )

        make_single_plot(
          variable=variable,
          data_events=data_sideband,
          stack_dict=stack_sideband_dict,
          signal_dict=signal_sideband_dict,
          output_file=output_file_linear,
          title_extra=sideband_title,
          force_log=False,
          blind_data=False,
        )

      print("mgg sideband preselection output saved in ", mgg_sideband_out_path)
    elif make_mgg_sideband_preselection_plots and only_MC:
      print("[mgg sideband] Data/MC sideband plots are skipped for only_MC=True.")
    else:
      print("[mgg sideband] Preselection sideband plotting disabled by switch.")

    print("inclusive output saved in ", out_path)
  else:
    print("[plot] Inclusive *_Plots_70_190 plotting disabled by switch.")

  if not make_sr_plots:
    print("[SR] *_SR_Plots plotting disabled by switch.")
    return

  # =========================
  # Exclusive category plotting
  # boosted -> VBF -> ggHH
  # =========================
  best_cut_params = []
  vbfhh_info_list = []

  if best_cut_json is not None:
    best_cut_params = parse_best_cut_params(
      best_cut_json,
      max_gghh_cats=max_gghh_cats,
    )

  if vbfhh_info_json is not None:
    vbfhh_info_list = parse_vbfhh_info(
      vbfhh_info_json,
      max_vbfhh_cats=max_vbfhh_cats,
    )

  has_boosted_field = boosted_field in data_combined.fields

  if not has_boosted_field:
    for _, evs in stack_mc_dict.items():
      if boosted_field in evs.fields:
        has_boosted_field = True
        break

    if not has_boosted_field:
      for _, evs in signal_mc_dict.items():
        if boosted_field in evs.fields:
          has_boosted_field = True
          break

  has_any_region_definition = (
    has_boosted_field
    or (len(vbfhh_info_list) > 0)
    or (len(best_cut_params) > 0)
  )

  if not has_any_region_definition:
    print("[SR] No boosted/VBF/ggHH region definition found. Skip category plotting.")
    return

  print("[SR] Building exclusive category definitions (boosted -> VBF -> ggHH)...")

  data_sr_masks = None

  if not only_MC:
    data_sr_masks = build_sr_masks_for_events(
      events=data_combined,
      class_names=class_names,
      signal_class_idx=signal_class_idx,
      bkg_class_indices=bkg_class_indices,
      best_cut_params=best_cut_params,
      vbfhh_class_idx=vbfhh_class_idx,
      vbfhh_info_list=vbfhh_info_list,
      boosted_field=boosted_field,
      include_boosted_cat=True,
    )

  stack_sr_masks = {}

  for sample_label, events in stack_mc_dict.items():
    stack_sr_masks[sample_label] = build_sr_masks_for_events(
      events=events,
      class_names=class_names,
      signal_class_idx=signal_class_idx,
      bkg_class_indices=bkg_class_indices,
      best_cut_params=best_cut_params,
      vbfhh_class_idx=vbfhh_class_idx,
      vbfhh_info_list=vbfhh_info_list,
      boosted_field=boosted_field,
      include_boosted_cat=True,
    )

  signal_sr_masks = {}

  for sample_label, events in signal_mc_dict.items():
    signal_sr_masks[sample_label] = build_sr_masks_for_events(
      events=events,
      class_names=class_names,
      signal_class_idx=signal_class_idx,
      bkg_class_indices=bkg_class_indices,
      best_cut_params=best_cut_params,
      vbfhh_class_idx=vbfhh_class_idx,
      vbfhh_info_list=vbfhh_info_list,
      boosted_field=boosted_field,
      include_boosted_cat=True,
    )

  region_names = []

  if data_sr_masks is not None and len(data_sr_masks) > 0:
    region_names = list(data_sr_masks.keys())
  else:
    for _, mask_dict in stack_sr_masks.items():
      if len(mask_dict) > 0:
        region_names = list(mask_dict.keys())
        break

  if len(region_names) == 0:
    print("[SR] No category masks built. Skip category plotting.")
    return

  if only_MC:
    sr_out_path = os.path.join(os.path.dirname(out_path), "MC_SR_Plots")
  else:
    sr_out_path = os.path.join(os.path.dirname(out_path), "Data_MC_SR_Plots")

  os.makedirs(sr_out_path, exist_ok=True)

  # Plot all variables in each SR/category.
                                                                                              
  sr_variables = variables

  for region_name in region_names:
    region_dir = os.path.join(sr_out_path, region_name)
    os.makedirs(region_dir, exist_ok=True)

    if (not only_MC) and (data_sr_masks is not None) and (region_name in data_sr_masks):
      data_region = data_combined[data_sr_masks[region_name]]
    else:
      data_region = None

    stack_region_dict = {}

    for sample_label, events in stack_mc_dict.items():
      if region_name in stack_sr_masks[sample_label]:
        stack_region_dict[sample_label] = events[stack_sr_masks[sample_label][region_name]]
      else:
        stack_region_dict[sample_label] = events[:0]

    signal_region_dict = {}

    for sample_label, events in signal_mc_dict.items():
      if region_name in signal_sr_masks[sample_label]:
        signal_region_dict[sample_label] = events[signal_sr_masks[sample_label][region_name]]
      else:
        signal_region_dict[sample_label] = events[:0]

    for variable in sr_variables:
      has_variable = False

      if data_region is not None and variable in data_region.fields:
        has_variable = True

      for _, evs in stack_region_dict.items():
        if variable in evs.fields:
          has_variable = True
          break

      if not has_variable:
        for _, evs in signal_region_dict.items():
          if variable in evs.fields:
            has_variable = True
            break

      if not has_variable:
        print(f"[SR] Variable {variable} not found in {region_name}. Skip.")
        continue

      output_file_log = os.path.join(region_dir, f"{variable}_log.png")

      make_single_plot(
        variable=variable,
        data_events=None if only_MC else data_region,
        stack_dict=stack_region_dict,
        signal_dict=signal_region_dict,
        output_file=output_file_log,
        title_extra=f"{region_name}",
        force_log=True,
        blind_data=True,
      )

      output_file_linear = os.path.join(region_dir, f"{variable}_linear.png")

      make_single_plot(
        variable=variable,
        data_events=None if only_MC else data_region,
        stack_dict=stack_region_dict,
        signal_dict=signal_region_dict,
        output_file=output_file_linear,
        title_extra=f"{region_name}",
        force_log=False,
        blind_data=True,
      )

  print("[SR] Exclusive category output saved in", sr_out_path)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Plot stacked histograms for MC and Data."
  )

  parser.add_argument(
    "--base-path",
    type=str,
    required=True,
    help="Path to the base directory containing MC and Data folders.",
  )

  parser.add_argument(
    "--training_config_path",
    type=str,
    required=True,
    help="Path to the training config file.",
  )

  parser.add_argument(
    "--sr-opt",
    type=str,
    default=None,
    help="Subdirectory under base_path containing best_cut_params.json and optionally vbfhh_sr_info.json.",
  )

  parser.add_argument(
    "--signal-class",
    type=int,
    default=None,
    help="Index of ggHH signal class in the score array for SR definition.",
  )

  parser.add_argument(
    "--bkg-classes",
    type=str,
    default=None,
    help="Comma-separated list of background class indices for SR definition, e.g. '0,1'.",
  )

  parser.add_argument(
    "--vbfhh-class",
    type=int,
    default=None,
    help="Optional index of VBFHH class in the score array.",
  )

  parser.add_argument(
    "--max-gghh-cats",
    type=int,
    default=3,
    help="Maximum number of ggHH SR categories to use from best_cut_params.json.",
  )

  parser.add_argument(
    "--max-vbfhh-cats",
    type=int,
    default=1,
    help="Maximum number of VBFHH SR categories to use from vbfhh_sr_info.json.",
  )

  parser.add_argument(
    "--mc-percentage",
    type=float,
    default=100.0,
    help=(
      "Percentage of the full MC sample present in base-path (greater than 0 and at most 100). "
      "MC weights are scaled by 100/percentage; data is unchanged."
    ),
  )

  parser.add_argument(
    "--boosted-field",
    type=str,
    default="is_boosted",
    help="Field name used to identify boosted events. Default: is_boosted",
  )

  parser.add_argument(
    "--plot-inclusive",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Control whether to make inclusive *_Plots_preselection plots. "
      "Use --no-plot-inclusive to disable."
    ),
  )

  parser.add_argument(
    "--plot-score-process",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Control whether to make Score_Plots_By_Process under the inclusive plot directory. "
      "Use --no-plot-score-process to disable."
    ),
  )

  parser.add_argument(
    "--plot-mgg-sideband-preselection",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Control whether to make Data/MC preselection plots in the mgg sideband, "
      "using mass outside the mass_window. Use --no-plot-mgg-sideband-preselection to disable."
    ),
  )

  parser.add_argument(
    "--plot-sr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Control whether to make exclusive *_SR_Plots. "
      "Use --no-plot-sr to disable."
    ),
  )

  parser.add_argument(
    "--plot-mc-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "Also produce the separate MC-only plot suite. Disabled by default because "
      "it requires loading all MC inputs a second time."
    ),
  )

  args = parser.parse_args()

  with open(args.training_config_path, "r") as f:
    training_config = yaml.safe_load(f)

  samples_info = training_config["samples_info"]

  base_path = args.base_path
  sim_folder = f"{base_path}/individual_samples"
  data_folder = f"{base_path}/individual_samples_data"
  var_prefix = training_config.get("var_prefix", "nonResReg")

  sim_samples = [
    "VBFHToGG_M_125",
    "VHtoGG_M_125",
    "ttHtoGG_M_125",
    "GluGluHToGG_M_125",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
    "TTGG",
    "GGJets",
    "DDQCDGJET",
  ]

  variables_ = [
    "mass",
    f"{var_prefix}_dijet_mass_DNNreg",
    "pt",
    f"{var_prefix}_HHbbggCandidate_mass",
    f"{var_prefix}_dijet_pt",
    f"{var_prefix}_lead_bjet_pt",
    f"{var_prefix}_sublead_bjet_pt",
  ]

  extra_vars = []

  for BSM_sample in [
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
  ]:
    if BSM_sample in training_config["sample_to_class"].keys():
      sim_samples.append(BSM_sample)

  input_vars_path = args.training_config_path.replace(
    "training_config.yaml",
    "input_variables.yaml",
  )

  with open(input_vars_path, "r") as f:
    input_vars = yaml.safe_load(f)

  input_vars = [
    var.replace("regcol_", f"{var_prefix}_") if "regcol_" in var else var
    for var in input_vars["mlp"]["vars"]
  ]

  variables = variables_ + extra_vars + input_vars
  variables = unique_keep_order(variables)

  best_cut_json = None
  vbfhh_info_json = None

  if args.sr_opt is not None:
    best_cut_json = os.path.join(
      base_path,
      args.sr_opt,
      "best_cut_params.json",
    )

    vbfhh_info_json = os.path.join(
      base_path,
      args.sr_opt,
      "vbfhh_sr_info.json",
    )

  if args.bkg_classes is None or args.bkg_classes.strip() == "":
    bkg_class_indices = None
  else:
    bkg_class_indices = [
      int(x.strip())
      for x in args.bkg_classes.split(",")
    ]

  out_path = f"{base_path}/"

  plot_stacked_histogram(
    samples_info,
    sim_folder,
    data_folder,
    sim_samples,
    variables,
    out_path,
    signal_scale=1000,
    only_MC=False,
    var_prefix=var_prefix,
    best_cut_json=best_cut_json,
    vbfhh_info_json=vbfhh_info_json,
    signal_class_idx=args.signal_class,
    bkg_class_indices=bkg_class_indices,
    vbfhh_class_idx=args.vbfhh_class,
    max_gghh_cats=args.max_gghh_cats,
    max_vbfhh_cats=args.max_vbfhh_cats,
    boosted_field=args.boosted_field,
    make_inclusive_plots=args.plot_inclusive,
    make_score_process_plots=args.plot_score_process,
    make_mgg_sideband_preselection_plots=args.plot_mgg_sideband_preselection,
    make_sr_plots=args.plot_sr,
    mc_percentage=args.mc_percentage,
  )

  if args.plot_mc_only:
    print("[plot] Producing optional MC-only plots; MC inputs will be loaded again.")
    plot_stacked_histogram(
      samples_info,
      sim_folder,
      data_folder,
      sim_samples,
      variables,
      out_path,
      signal_scale=1000,
      only_MC=True,
      var_prefix=var_prefix,
      best_cut_json=best_cut_json,
      vbfhh_info_json=vbfhh_info_json,
      signal_class_idx=args.signal_class,
      bkg_class_indices=bkg_class_indices,
      vbfhh_class_idx=args.vbfhh_class,
      max_gghh_cats=args.max_gghh_cats,
      max_vbfhh_cats=args.max_vbfhh_cats,
      boosted_field=args.boosted_field,
      make_inclusive_plots=args.plot_inclusive,
      make_score_process_plots=args.plot_score_process,
      make_mgg_sideband_preselection_plots=args.plot_mgg_sideband_preselection,
      make_sr_plots=args.plot_sr,
      mc_percentage=args.mc_percentage,
    )
  else:
    print("[plot] Separate MC-only plots disabled; input files were loaded once.")
