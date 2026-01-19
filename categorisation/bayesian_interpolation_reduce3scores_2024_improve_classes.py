import argparse
import json
import math
import os

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import optuna
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class OptunaCategorizer:
  def __init__(
    self,
    base_path,
    cat_folder=None,
    signal_class=3,  # ggHH index
    signal_class_name="ggHH",  # Name for output cut strings
    signal_samples=None,
    samples_list=None,
    bkg_samples=None,
    bkg_classes=None,  # List of background class indices for score cuts
    bkg_class_names=None,  # Names for output cut strings
    n_categories=5,
    n_trials_optuna=300,
    n_runs=30,
    side_band_threshold_low=10,
    side_band_threshold_high=15,
    beta=0.1,
    gamma_strategy="linear",
    SR_strategy="sequential",
    sig_type=None,
  ):
    self.base_path = base_path
    self.cat_folder = cat_folder
    self.signal_class = signal_class
    self.signal_class_name = signal_class_name
    self.signal_samples = signal_samples
    self.bkg_samples = bkg_samples
    self.best_cut_params = []
    self.best_cut_params_cr = []
    self.samples_list = samples_list
    self.n_categories = n_categories
    self.n_trials_optuna = n_trials_optuna
    print(side_band_threshold_low, side_band_threshold_high)
    self.side_band_threshold_low = side_band_threshold_low
    self.side_band_threshold_high = side_band_threshold_high
    self.n_runs = n_runs
    self.beta = beta
    self.gamma_strategy = gamma_strategy
    self.SR_strategy = SR_strategy
    self.sig_type = sig_type

    # Background classes for score-based cuts (default: ttH=1, singleH=2)
    self.bkg_classes = bkg_classes if bkg_classes is not None else [1, 2]
    if bkg_class_names is not None:
      self.bkg_class_names = bkg_class_names
    elif self.bkg_classes:
      # Default names based on indices
      self.bkg_class_names = [f"bkg_{i}" for i in self.bkg_classes]
    else:
      self.bkg_class_names = []

    # Samples for which SR yield is estimated from SB by linear interpolation
    self.interp_samples = {"TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"}
    # Sideband and SR windows (GeV)
    self.mass_left_sb = (100.0, 120.0)
    self.mass_sr = (120.0, 130.0)
    self.mass_right_sb = (130.0, 180.0)

    if self.cat_folder is None:
      print(
        "INFO: No output directory specified, using default: optuna_categorization"
      )
      print(
        "INFO: If there is a previous run with the same output directory, it will be overwritten."
      )
      self.cat_folder = "optuna_categorization"
    if self.samples_list is None:
      self.samples_list = [
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
        "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
        "TTGG",
        "GGJets",
        "DDQCDGJET",
        "TTG_100_200",
        "TTG_200",
      ]
    if self.bkg_samples is None:
      self.bkg_samples = [
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "TTGG",
        "GGJets",
        "DDQCDGJET",
        "TTG_100_200",
        "TTG_200",
      ]
    if self.signal_samples is None:
      raise ValueError("Missing signal sample definition")

    # will be filled in load_samples()
    self.scores_all = None
    self.mass_all = None
    self.dijet_mass_all = None
    self.weights_all = None
    self.labels_all = None
    self.samples_all = None

    self.apply_preselection = True

  def gamma_fn(self):
    def gamma_linear(n):
      return min(int(np.ceil(self.beta * n)), 25)

    def gamma_sqrt(n):
      return min(int(np.ceil(self.beta * np.sqrt(n))), 25)

    if self.gamma_strategy == "linear":
      return gamma_linear
    elif self.gamma_strategy == "sqrt":
      return gamma_sqrt
    else:
      raise ValueError("Unsupported gamma_strategy. Use 'linear' or 'sqrt'.")

  def plot_optuna_history(self, study, out_dir, category):
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig = ax.get_figure()
    fig.suptitle(
      f"Optuna Optimization History for Category {category}", fontsize=14
    )
    out_dir = os.path.join(out_dir, "optuna_history_plots")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"optuna_history_cat_{category}.png"))
    plt.clf()

  def plot_parallel_coordinates(self, study, out_dir, category):
    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig = ax.get_figure()
    fig.suptitle(f"Parallel Coordinates for Category {category}", fontsize=14)
    out_dir = os.path.join(out_dir, "optuna_history_plots")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"parallel_coordinates_cat_{category}.png"))
    plt.clf()

  def preselection(self, events, scores):
    mass_bool = (events.mass > 100) & (events.mass < 180)
    dijet_mass_bool = (events.nonResReg_dijet_mass_DNNreg > 70) & (
      events.nonResReg_dijet_mass_DNNreg < 190
    )
    lead_mvaID_bool = events.lead_mvaID > -0.7
    sublead_mvaID_bool = events.sublead_mvaID > -0.7

    mask = mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool
    events = events[mask]
    scores = scores[mask]
    return events, scores

  def load_samples(self):

    data = {
      k: []
      for k in (
        "score",
        "diphoton_mass",
        "dijet_mass",
        "weights",
        "labels",
        "sample",
      )
    }
    eras = ["2024"]
    dijet_mass_key = "nonResReg_dijet_mass_DNNreg"

    for era in eras:
      for sample in self.samples_list:
        samp_dir = os.path.join(self.base_path, "individual_samples", era, sample)
        y_file = os.path.join(samp_dir, "y.npy")
        evt_file = os.path.join(samp_dir, "events.parquet")

        if not (os.path.exists(y_file) and os.path.exists(evt_file)):
          print(
            f"[load_samples] WARNING: missing files for {samp_dir}, skipping."
          )
          continue

        try:
          y = np.load(y_file)
          events = ak.from_parquet(
            evt_file,
            columns=[
              "mass",
              dijet_mass_key,
              "lead_genPartFlav",
              "sublead_genPartFlav",
              "weight_tot",
              "lead_mvaID",
              "sublead_mvaID",
            ],
          )
          if self.apply_preselection:
            events, y = self.preselection(events, y)
        except Exception as exc:
          print(f"[load_samples] ERROR while reading {samp_dir}: {exc}")
          continue

        # Prompt-photon requirement for tt̄γ-like samples
        if sample.startswith("TTG_") or sample in {"TT", "TTGG"}:
          sel = (events["lead_genPartFlav"] == 1) & (
            events["sublead_genPartFlav"] == 1
          )
          events = events[sel]
          y = y[sel]

        if len(y) == 0:
          continue

        data["score"].append(y)
        data["diphoton_mass"].append(np.asarray(events["mass"]))
        data["dijet_mass"].append(np.asarray(events[dijet_mass_key]))
        data["weights"].append(np.asarray(events["weight_tot"]))

        if sample in self.signal_samples:
          print("!!!!!!=================using as signal: ", sample)
          label_val = 1
        elif "GluGlutoHHto2B2G" in sample:
          # exclude EFT HH variations
          label_val = -1
          print("!!!!!!=================excluding as signal: ", sample)
        else:
          label_val = 0

        data["labels"].append(
          np.full(len(y), label_val, dtype=int)
        )
        data["sample"].append(np.repeat(sample, len(y)))

    if not data["score"]:
      raise RuntimeError("[load_samples] No events found in any input sample.")

    # Concatenate everything
    for key in data:
      data[key] = np.concatenate(data[key], axis=0)

    scores = data.pop("score")  # shape (N, n_classes)
    df = pd.DataFrame(
      {
        "diphoton_mass": data["diphoton_mass"],
        "dijet_mass": data["dijet_mass"],
        "weights": data["weights"],
        "labels": data["labels"],
        "sample": data["sample"],
      }
    )

    # store arrays for fast access in optimization
    self.scores_all = scores
    self.mass_all = df["diphoton_mass"].to_numpy()
    self.dijet_mass_all = df["dijet_mass"].to_numpy()
    self.weights_all = df["weights"].to_numpy()
    self.labels_all = df["labels"].to_numpy()
    self.samples_all = df["sample"].to_numpy()

    in_peak = (df["diphoton_mass"] > 120) & (df["diphoton_mass"] < 130)
    print(
      f"Background weight: {df.loc[df['labels'] == 0, 'weights'].sum():.3g}"
    )
    print(f"Signal weight:     {df.loc[df['labels'] == 1, 'weights'].sum():.3g}")
    print(
      f"Bkg weight 120-130 GeV: {df.loc[(df['labels'] == 0) & in_peak, 'weights'].sum():.3g}"
    )
    print(
      f"Sig weight 120-130 GeV: {df.loc[(df['labels'] == 1) & in_peak, 'weights'].sum():.3g}"
    )

    return df

  def _sr_from_sidebands_linear(self, mass, weights, left_sb, sr, right_sb):
    """Return (b_sr, var_b_sr) using a linear (average) interpolation from SB densities."""
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
      dens = 0.5 * (dens_L + dens_R)
      var_dens = 0.25 * (var_dens_L + var_dens_R)  # assume independence
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

  #############################################
  # Sequential categorization using Optuna
  #############################################

  def optmize_SR_sequential(self, samples_input):

    if self.scores_all is None:
      raise RuntimeError(
        "scores_all is not set. Make sure to call load_samples() first."
      )

    cat_path = os.path.join(self.base_path, f"{self.cat_folder}")
    print(f"Creating output directory: {cat_path}")
    os.makedirs(cat_path, exist_ok=True)

    first_scores = self.scores_all

    # Handle 1D scores from binary classifiers
    if first_scores.ndim == 1:
      self.scores_all = self.scores_all.reshape(-1, 1)
      first_scores = self.scores_all

    # Determine minimum number of score components needed
    if self.bkg_classes:
      min_classes_needed = max(self.signal_class, max(self.bkg_classes)) + 1
    else:
      min_classes_needed = self.signal_class + 1

    if first_scores.shape[1] < min_classes_needed:
      raise RuntimeError(
        f"Expected score vectors with at least {min_classes_needed} components "
        f"(signal_class={self.signal_class}, bkg_classes={self.bkg_classes})."
      )

    # Use configured background classes for score-based cuts
    bg_classes = self.bkg_classes

    run_significance_list = []
    run_best_params_list = []
    run_sig_peak_list = []
    run_bkg_side_list = []

    total_signal_weight = self.weights_all[self.labels_all == 1].sum()
    print(f"Total signal weight (all events): {total_signal_weight}")

    for run in range(self.n_runs):
      print(f"--- Run {run} ---")

      # mask tracking which global events are still available
      remaining_mask = np.ones(self.scores_all.shape[0], dtype=bool)

      best_cut_params_list = []
      best_sig_values = []
      sig_peak_list = []
      bkg_side_list = []

      # optional: placeholders if you later want dynamic search ranges
      prev_signal_cut = 1.0
      prev_bg_cut = {b: 0.0 for b in bg_classes}

      for cat in range(1, self.n_categories + 1):
        print(f"\n--- Optimizing Category {cat} of {self.n_categories} ---")

        # slice arrays for currently remaining events
        scores = self.scores_all[remaining_mask]
        dipho_mass = self.mass_all[remaining_mask]
        labels = self.labels_all[remaining_mask]
        weights = self.weights_all[remaining_mask]
        samples = self.samples_all[remaining_mask]

        if scores.shape[0] == 0:
          print("No events remaining for further categorization.")
          break

        def objective(trial):
          # sample thresholds (kept 0-1 to preserve original behaviour;
          # prev_*_cut reserved if you later want to restrict ranges)
          th_signal = trial.suggest_float("th_signal", 0.0, 1.0)
          mask = scores[:, self.signal_class] > th_signal

          for b in bg_classes:
            th_bg = trial.suggest_float(f"th_bg_{b}", 0.0, 1.0)
            mask = mask & (scores[:, b] < th_bg)

          if not np.any(mask):
            return -1.0

          # Sideband requirement on selected background events
          side_mask = (
            ((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130))
            & (labels[mask] == 0)
          )
          bkg_side_val = weights[mask][side_mask].sum()
          if cat == 1:
            if (
              bkg_side_val < self.side_band_threshold_low
              or bkg_side_val > self.side_band_threshold_high
            ):
              return -1.0
          else:
            if bkg_side_val < self.side_band_threshold_low:
              return -1.0

          # SR window
          mass_mask = (dipho_mass[mask] > 120) & (
            dipho_mass[mask] < 130
          )
          if not np.any(mass_mask):
            return -1.0

          # compute s and b for Z (with SB interpolation for chosen samples)
          sel_labels_full = labels[mask]
          sel_masses_full = dipho_mass[mask]
          sel_weights_full = weights[mask]
          sel_samples_full = samples[mask]

          s = sel_weights_full[mass_mask & (sel_labels_full == 1)].sum()

          b_total = 0.0
          for sname in self.bkg_samples:
            smask = (sel_labels_full == 0) & (sel_samples_full == sname)
            if not np.any(smask):
              continue

            if sname in self.interp_samples:
              b_i, _ = self._sr_from_sidebands_linear(
                sel_masses_full[smask],
                sel_weights_full[smask],
                left_sb=self.mass_left_sb,
                sr=self.mass_sr,
                right_sb=self.mass_right_sb,
              )
              b_total += b_i
            else:
              sr_mask_i = smask & (
                (sel_masses_full > 120) & (sel_masses_full < 130)
              )
              b_total += sel_weights_full[sr_mask_i].sum()

          if s <= 0.0:
            return -1.0
          if b_total <= 0.0:
            b_total = 1e-9

          z = np.sqrt(
            2.0 * ((s + b_total) * np.log(1.0 + s / b_total) - s)
          )
          return float(z)

        sampler = optuna.samplers.TPESampler(gamma=self.gamma_fn())
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
          objective, n_trials=self.n_trials_optuna, show_progress_bar=False
        )

        best_params = study.best_params
        best_target = study.best_value

        if best_target <= 0.0:
          print(
            f"Category {cat}: no positive significance found, stopping categorization for this run."
          )
          break

        best_cut_params_list.append(best_params)
        best_sig_values.append(best_target)
        print(
          f"Category {cat}: Best parameters: {best_params} with significance {best_target:.4f}"
        )

        self.plot_optuna_history(
          study, cat_path, category=f"run_{run}_cat_{cat}"
        )
        self.plot_parallel_coordinates(
          study, cat_path, category=f"run_{run}_cat_{cat}"
        )

        # Apply best mask with the optimal thresholds
        mask_local = scores[:, self.signal_class] > best_params["th_signal"]
        for b in bg_classes:
          mask_local &= scores[:, b] < best_params[f"th_bg_{b}"]

        if not np.any(mask_local):
          print(
            f"Category {cat}: best parameters select no events, stopping categorization for this run."
          )
          break

        sel_labels = labels[mask_local]
        sel_masses = dipho_mass[mask_local]
        sel_weights = weights[mask_local]

        # bookkeeping (signal in peak, background in sidebands)
        in_peak = (sel_masses > 120) & (sel_masses < 130)
        signal_in_peak = sel_weights[in_peak & (sel_labels == 1)].sum()
        bkg_in_side = sel_weights[
          ((sel_masses < 120) | (sel_masses > 130)) & (sel_labels == 0)
        ].sum()

        sig_peak_list.append(signal_in_peak)
        bkg_side_list.append(bkg_in_side)

        # Update dynamic search ranges (placeholders if you want to change suggest ranges)
        prev_signal_cut = best_params["th_signal"]
        for b in bg_classes:
          prev_bg_cut[b] = best_params[f"th_bg_{b}"]

        # Remove selected events globally for next category
        global_selected_mask = np.zeros_like(remaining_mask)
        global_selected_mask[remaining_mask] = mask_local
        remaining_mask &= ~global_selected_mask

        if not np.any(remaining_mask):
          print("No events remaining after this category.")
          break

      run_significance_list.append(best_sig_values)
      run_best_params_list.append(best_cut_params_list)
      run_sig_peak_list.append(sig_peak_list)
      run_bkg_side_list.append(bkg_side_list)

    if not run_significance_list:
      raise RuntimeError("No successful categorization runs completed.")

    # pick best run based on quadrature sum of per-category significances
    run_sum_Z_quad = [
      np.sqrt(np.sum(np.array(sig) ** 2)) for sig in run_significance_list
    ]
    max_index = int(np.argmax(run_sum_Z_quad))

    best_sig_values = run_significance_list[max_index]
    best_cut_params_list = run_best_params_list[max_index]
    sig_peak_list = run_sig_peak_list[max_index]
    bkg_side_list = run_bkg_side_list[max_index]

    print("Best run index:", max_index)
    print("Per-category significances:", best_sig_values)
    print("Signal in SR (per category):", sig_peak_list)
    print("Bkg in SB (per category):", bkg_side_list)

    # Build per-category base cuts (from your best_cut_params_list)
    base_cuts = []
    for p in best_cut_params_list:
      parts = [f"{self.signal_class_name}_score > {p['th_signal']}"]
      for idx, bg_idx in enumerate(self.bkg_classes):
        name = self.bkg_class_names[idx]
        parts.append(f"{name}_score < {p[f'th_bg_{bg_idx}']}")
      parts.append("is_boosted == 0")
      base_cuts.append("(" + " & ".join(parts) + ")")

    # Add NOT-previous-cats and dijet mass window to form final category strings
    cat_strings = {}
    for i, base in enumerate(base_cuts, start=1):
      parts = [base]
      if i >= 2:
        for j in range(i - 1):
          parts.append(f"not({base_cuts[j]})")
      parts.append("dijet_mass > 80")
      parts.append("dijet_mass < 190")
      cat_strings[f"cat{i}"] = " & ".join(parts)

    # Save a single txt file with the requested JSON-style mapping
    best_params_path = os.path.join(cat_path, "best_cut_params.txt")
    with open(best_params_path, "w") as f:
      f.write(json.dumps(cat_strings, indent=2))

    # also save detailed best parameters as a JSON file
    best_params_json_path = os.path.join(cat_path, "best_cut_params.json")
    with open(best_params_json_path, "w") as f:
      json.dump(best_cut_params_list, f, indent=4)

    return best_cut_params_list, best_sig_values

  def run_categorisation(self):
    _ = self.load_samples()

    if self.SR_strategy == "sequential":
      best_params, best_sig_values = self.optmize_SR_sequential(None)
    elif self.SR_strategy == "simultaneous":
      raise NotImplementedError("Simultaneous SR strategy is not implemented yet.")


#############################################
# Main execution
#############################################

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description=(
      "Categorize multiclass scores using Optuna with dynamic search "
      "ranges, sideband requirements, and summary plots."
    )
  )
  parser.add_argument(
    "--n_categories",
    type=int,
    default=5,
    help="Number of categories to optimize",
  )
  parser.add_argument(
    "--base_path", type=str, required=True, help="Base path to the input samples"
  )
  parser.add_argument(
    "--optuna_folder",
    type=str,
    default="optuna_categorization",
    help="Folder name for Optuna results",
  )
  parser.add_argument(
    "--n_trials",
    type=int,
    default=200,
    help="Number of trials for Optuna optimization",
  )
  parser.add_argument(
    "--SR_strategy",
    type=str,
    choices=["sequential", "simultaneous"],
    default="sequential",
    help="Strategy for SR categorization",
  )
  parser.add_argument(
    "--n_runs",
    type=int,
    default=50,
    help="Number of complete runs for the categorization",
  )
  parser.add_argument(
    "--gamma_strategy",
    type=str,
    choices=["sqrt", "linear"],
    default="linear",
    help="Gamma strategy for TPE sampler",
  )
  parser.add_argument(
    "--side_band_threshold_low",
    type=int,
    default=10,
    help="Threshold for sideband requirements",
  )
  parser.add_argument(
    "--side_band_threshold_high",
    type=int,
    default=50,
    help="Threshold for sideband requirements",
  )
  parser.add_argument(
    "--signal_samples",
    type=str,
    required=True,
    help="Comma-separated list of signal sample names",
  )
  parser.add_argument(
    "--signal_class",
    type=int,
    default=3,
    help="Index of the signal class in the score array (default: 3 for ggHH)",
  )
  parser.add_argument(
    "--signal_class_name",
    type=str,
    default="ggHH",
    help="Name of the signal class for output cut strings (default: ggHH)",
  )
  parser.add_argument(
    "--bkg_classes",
    type=str,
    default="1,2",
    help="Comma-separated list of background class indices for score cuts (e.g., '1,2' for ttH and singleH). Use empty string for binary classifier.",
  )
  parser.add_argument(
    "--bkg_class_names",
    type=str,
    default=None,
    help="Comma-separated list of background class names for output (e.g., 'ttH,singleH'). If not provided, uses 'bkg_<index>' format.",
  )

  args = parser.parse_args()

  # Parse bkg_classes from comma-separated string to list of ints
  # Empty string means no background classes (binary classifier)
  if args.bkg_classes.strip() == "":
    bkg_classes = []
  else:
    bkg_classes = [int(x.strip()) for x in args.bkg_classes.split(",")]

  # Parse bkg_class_names if provided
  bkg_class_names = None
  if args.bkg_class_names is not None:
    bkg_class_names = [x.strip() for x in args.bkg_class_names.split(",")]
    if len(bkg_class_names) != len(bkg_classes):
      raise ValueError(
        f"Number of bkg_class_names ({len(bkg_class_names)}) must match "
        f"number of bkg_classes ({len(bkg_classes)})"
      )

  categoriser = OptunaCategorizer(
    base_path=args.base_path,
    cat_folder=args.optuna_folder,
    n_categories=args.n_categories,
    n_trials_optuna=args.n_trials,
    n_runs=args.n_runs,
    side_band_threshold_low=args.side_band_threshold_low,
    side_band_threshold_high=args.side_band_threshold_high,
    SR_strategy=args.SR_strategy,
    signal_samples=args.signal_samples,
    signal_class=args.signal_class,
    signal_class_name=args.signal_class_name,
    bkg_classes=bkg_classes,
    bkg_class_names=bkg_class_names,
    gamma_strategy=args.gamma_strategy,
  )
  categoriser.run_categorisation()
