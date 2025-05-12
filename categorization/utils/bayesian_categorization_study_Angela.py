
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import optuna
import math

from tools import load_samples, asymptotic_significance, approx_significance
import pandas as pd
import json
import mplhep
from ploting import plot_stacked_histogram

def plot_optuna_history(study, out_dir, category):
    """
    Plot the optimization history using Optuna's matplotlib interface.
    """
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig = ax.get_figure()  # Retrieve the parent Figure.
    fig.suptitle(f"Optuna Optimization History for Category {category}", fontsize=14)
    fig.savefig(os.path.join(out_dir, f"optuna_history_cat_{category}.png"))
    plt.clf()

def plot_parallel_coordinates(study, out_dir, category):
    """
    Plot parallel coordinates using Optuna's matplotlib interface.
    """
    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig = ax.get_figure()  # Retrieve the parent Figure.
    fig.suptitle(f"Parallel Coordinates for Category {category}", fontsize=14)
    fig.savefig(os.path.join(out_dir, f"parallel_coordinates_cat_{category}.png"))
    plt.clf()

def plot_mass_category(selected_events, out_dir, category):
    """
    Plot diphoton and dijet mass distributions for the given category.
    """
    import mplhep
    dipho_mass = selected_events["diphoton_mass"].values
    dijet_mass = selected_events["dijet_mass"].values
    labels = selected_events["labels"].values
    weights = selected_events["weights"].values
    sample = selected_events["sample"].values

    plt.style.use(mplhep.style.CMS)

    # Diphoton mass distribution.
    non_res_bkg_mask = ((sample == "GGJets") | (sample == "TTGG") | (sample == "DDQCDGJET"))
    plt.hist(dipho_mass[non_res_bkg_mask], bins=20, range=(100, 180),
             weights=weights[non_res_bkg_mask], histtype='step', label="Background")
    plt.hist(dipho_mass[labels == 1], bins=20, range=(100, 180),
             weights=weights[labels == 1], histtype='step', label="Signal")
    plt.xlabel("$m_{\\gamma\\gamma}$ [GeV]", fontsize=12)
    plt.ylabel("Number of events", fontsize=12)
    plt.legend()
    plt.title(f"Diphoton Mass (Category {category})", fontsize=14)
    plt.savefig(os.path.join(out_dir, f"diphoton_mass_cat_{category}.png"))
    plt.clf()

    # Dijet mass distribution.
    plt.hist(dijet_mass[non_res_bkg_mask], bins=20, range=(100, 180),
             weights=weights[non_res_bkg_mask], histtype='step', label="Background")
    plt.hist(dijet_mass[labels == 1], bins=20, range=(100, 180),
             weights=weights[labels == 1], histtype='step', label="Signal")
    plt.xlabel("$m_{jj}$ [GeV]", fontsize=12)
    plt.ylabel("Number of events", fontsize=12)
    plt.legend()
    plt.title(f"Dijet Mass (Category {category})", fontsize=14)
    plt.savefig(os.path.join(out_dir, f"dijet_mass_cat_{category}.png"))
    plt.clf()

def plot_mass_sideband_data(best_cut_values, base_path, plot_dir):
    # plot the diphoton mass and dijet mass for each category for data

    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    events = []
    scores = []
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        events.append(ak.from_parquet(inputs_path))
        scores.append(np.load(f"{inputs_path}/y.npy"))

    # concatenate the data samples
    events = ak.concatenate(events, axis=0)
    scores = np.concatenate(scores, axis=0)
    side_band_mask = ((events["mass"] < 120) | (events["mass"] > 130))
    events = events[side_band_mask]
    scores = scores[side_band_mask]
    # only consider sidebands data

    selected_events = events
    selected_scores = scores

    for i in range(len(best_cut_values)):
        # get diphoton and dijet mass for each category
        score_cuts = best_cut_values[i]
        mask = selected_scores[:, 3] > score_cuts["th_signal"]

        for b in [0, 1, 2]:
            mask = mask & (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

        data_diphoton_mass = selected_events[mask]["mass"]
        data_dijet_mass = selected_events[mask]["nonRes_mjj_regressed"]

        selected_events = selected_events[~mask]
        selected_scores = selected_scores[~mask]

        # plot the diphoton mass in range 100-180 from the sample weights
        plt.hist(data_diphoton_mass, bins=20, range=(100, 180), histtype='bar', label="Data 2022")
        plt.xlabel("$m_{\gamma\gamma}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/diphoton_mass_data_cat_{i+1}.png")
        plt.clf()

        # plot the dijet mass in range 100-180 from the sample weights
        plt.hist(data_dijet_mass, bins=20, range=(100, 180), histtype='bar', label="Data 2022")
        plt.xlabel("$m_{jj}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/dijet_mass_data_cat_{i+1}.png")
        plt.clf()

def plot_category_summary(best_sig_values, sig_peak_list, bkg_side_list, out_dir):
    """
    Plot the cumulative quadrature sum of significance (Z_sum_quad), individual category Z,
    signal in the peak, and background in the sideband versus category number.
    """
    plt.style.use(mplhep.style.CMS)
    categories = np.arange(1, len(best_sig_values) + 1)
    z_sum_quad = [np.sqrt(np.sum(np.array(best_sig_values[:i])**2)) for i in range(1, len(best_sig_values)+1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(categories, z_sum_quad, "bo-", label="Cumulative Z (quad sum)")
    plt.plot(categories, best_sig_values, "ro-", label="Category Z")
    plt.plot(categories, sig_peak_list, "go-", label="Signal in Peak")
    plt.plot(categories, bkg_side_list, "ko-", label="Bkg in Sideband")
    plt.xlabel("Category Number", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Category Summary", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(out_dir, "category_summary.png"))
    plt.clf()

def plot_category_summary_with_thresholds(
    best_sig_values,
    sig_peak_list,
    bkg_side_list,
    best_cut_params_list,
    save_path
):
    """
    Make a 4-subplot figure showing:
      1) Asymptotic significance (Z)
      2) Z sum in quadrature
      3) Signal under the peak
      4) Background in sidebands (log scale)
    Annotate each category's thresholds on the significance subplot.
    """
    plt.style.use(mplhep.style.CMS)
    n_cats = len(best_sig_values)
    cat_indices = np.arange(n_cats)

    # Compute the Z sum in quadrature cumulatively.
    z_sum_quad = []
    for i in range(1, n_cats + 1):
        z_sum_quad.append(np.sqrt(np.sum(np.array(best_sig_values[:i])**2)))

    # Create the figure and subplots with a shared x-axis
    fig, axs = plt.subplots(
        4, 1, 
        figsize=(10, 12), 
        gridspec_kw={'height_ratios': [1, 1, 1, 1], 'hspace': 0}
    )

    # -- 1) Plot Asymptotic Significance
    axs[0].plot(cat_indices, best_sig_values, ".b", markersize=8)
    axs[0].set_ylabel("Asymptotic Significance (Z)", fontsize=10)
    axs[0].grid(True)
    # Remove x-tick labels for this subplot (to avoid clutter)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    th_name_to_calss = {
        "th_signal": "th_ggFHH",
        "th_bg_0": "th_nonRes",
        "th_bg_1": "th_ttH",
        "th_bg_2": "th_singleH",
        "th_bg_4": "th_VBFHH",
    }

    # Annotate threshold values on the significance plot
    for i in range(n_cats):
        thresholds_dict = best_cut_params_list[i]
        # Construct a small text with the thresholds
        # Example: "th_signal=0.80\nth_bg_0=0.30\nth_bg_1=0.40"
        threshold_text = "\n".join(
            f"{th_name_to_calss[k]}={v:.2f}" for k, v in thresholds_dict.items()
        )
        # Place annotation above each point
        axs[0].annotate(
            threshold_text,
            xy=(i, best_sig_values[i]),
            xytext=(i, best_sig_values[i] * 1.10),  # adjust vertical offset
            ha="center",
            arrowprops=dict(color="black", arrowstyle="->", lw=1),
            fontsize=10
        )

    # -- 2) Plot Z sum in quadrature
    axs[1].plot(cat_indices, z_sum_quad, ".b", markersize=8)
    axs[1].set_ylabel("Z sum in quadrature", fontsize=10)
    axs[1].grid(True)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # -- 3) Plot Signal under the peak
    axs[2].plot(cat_indices, sig_peak_list, ".b", markersize=8)
    axs[2].set_ylabel("Signal under peak", fontsize=10)
    axs[2].grid(True)
    axs[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # -- 4) Plot Background in sidebands (log scale)
    axs[3].plot(cat_indices, bkg_side_list, ".b", markersize=8)
    axs[3].set_yscale("log")
    axs[3].set_ylabel("Bkg in sidebands", fontsize=10)
    axs[3].set_xlabel("Category Index", fontsize=10)
    axs[3].grid(True)
    # annot the background values
    for i in range(n_cats):
        axs[3].annotate(
            f"{bkg_side_list[i]:.5f}",
            xy=(i, bkg_side_list[i]),
            xytext=(i, bkg_side_list[i] + (bkg_side_list[i] * 0.5)),
            fontsize=8,
            arrowprops=dict(color="black", arrowstyle="->"),
            ha="center"
        )

    # Adjust spacing
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(save_path, "category_summary_new.png"))
    plt.close(fig)

def asimov_significance(s, b, eps=1e-9):
    """
    Approximate Asimov significance:
      Z ~ sqrt(2 * [ (s+b)*ln(1 + s/b) - s ])
    """
    # Guard against non-positive arguments
    if b <= 0 or s <= 0:
        return 0.0
    return np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s))

def optimize_sr_cr_cuts(
    samples_input,
    n_classes,
    signal_class,
    control_classes,
    class_names,
    n_srs=2,
    n_crs=2,
    n_trials=100,
    out_dir="optuna_output",
    min_bkg_sideband=10.0
):
    os.makedirs(out_dir, exist_ok=True)

    scores = np.stack(samples_input["score"].values)
    weights = samples_input["weights"].values
    labels = samples_input["labels"].values
    masses = samples_input["diphoton_mass"].values

    bkg_classes = [i for i in range(n_classes) if i != signal_class]

    best_params_list = []
    
    def objective(trial):
        nonlocal best_params_list

        # === 1) Build cut thresholds for SRs (n_srs) and CRs (n_crs) ===
        sr_thresholds = []
        for i in range(n_srs):
            sr_cut = {
                f"th_{class_names[signal_class]}": trial.suggest_float(f"sr_{i}_th_{class_names[signal_class]}", 0.0, 1.0),
            }
            for bg in bkg_classes:
                sr_cut[f"th_{class_names[bg]}_max"] = trial.suggest_float(
                    f"sr_{i}_th_{class_names[bg]}_max", 0.0, 1.0
                )
            sr_thresholds.append(sr_cut)

        cr_thresholds = []
        for i, ctrl_cls in enumerate(control_classes):
            cr_cut = {
                "target_class": ctrl_cls,
                f"th_{class_names[signal_class]}_max": trial.suggest_float(f"cr_{i}_th_{class_names[signal_class]}_max", 0.0, 1.0),
                f"th_{class_names[ctrl_cls]}_min": trial.suggest_float(f"cr_{i}_th_{class_names[ctrl_cls]}_min", 0.0, 1.0)
            }
            for bg in bkg_classes:
                if bg != ctrl_cls:
                    cr_cut[f"th_{class_names[bg]}_max"] = trial.suggest_float(
                        f"cr_{i}_th_{class_names[bg]}_max", 0.0, 1.0
                    )
            cr_thresholds.append(cr_cut)

        # === 2) Assign each event to an SR, CR, or "unused" region (region_ids=-1) ===
        region_ids = -1 * np.ones(len(scores), dtype=int)
        region_index = 0

        # -- Mark SR bins
        sr_masks = []
        for i, cut in enumerate(sr_thresholds):
            mask = scores[:, signal_class] > cut[f"th_{class_names[signal_class]}"]
            for bg in bkg_classes:
                mask &= scores[:, bg] < cut[f"th_{class_names[bg]}_max"]
            mask &= (region_ids == -1)  # ensure no double assignment
            region_ids[mask] = region_index
            sr_masks.append(mask)
            region_index += 1

        # -- Mark CR bins
        cr_masks = []
        if n_crs > 0:
            for i, cut in enumerate(cr_thresholds):
                target_cls = cut["target_class"]
                mask = (
                    (scores[:, signal_class] < cut[f"th_{class_names[signal_class]}_max"]) &
                    (scores[:, target_cls] > cut[f"th_{class_names[target_cls]}_min"])
                )
                for bg in bkg_classes:
                    if bg != target_cls:
                        mask &= scores[:, bg] < cut[f"th_{class_names[bg]}_max"]
                mask &= (region_ids == -1)
                region_ids[mask] = region_index
                cr_masks.append(mask)
                region_index += 1

        # === 4) Compute significance for SRs ===
        sr_significances = []
        for i in range(n_srs):
            # region i corresponds to the i-th SR
            sr_mask = (region_ids == i)
            # Consider mass window: 120 < m < 130
            mass_mask = (masses > 120.0) & (masses < 130.0)

            # Compute total signal in [120,130]
            s = np.sum(weights[sr_mask & mass_mask & (labels == signal_class)])
            # Compute total background in [120,130]
            b = np.sum(weights[sr_mask & mass_mask & (labels != signal_class)])
            # Asimov significance for this SR
            sr_significances.append(asimov_significance(s, b))

        # Combine SRs by summing the significances in quadrature
        total_sr_significance = np.sqrt(np.sum(np.square(sr_significances)))

        # === 5) Compute significance for CRs (if you actually want to) ===
        cr_significances = []
        for i in range(n_srs, n_srs + n_crs):
            cr_mask = (region_ids == i)
            # The same mass window (if relevant), or maybe you don't use a mass cut in CR:
            mass_mask = (masses > 120.0) & (masses < 130.0)
            s = np.sum(weights[cr_mask & mass_mask & (labels == signal_class)])
            b = np.sum(weights[cr_mask & mass_mask & (labels != signal_class)])
            cr_significances.append(asimov_significance(s, b))

        total_cr_significance = np.sqrt(np.sum(np.square(cr_significances)))

        # === 6) Check sideband counts for a penalty ===
        # Example sideband: [105,120) U (130,160]
        sideband_mask = ((masses > 105.0) & (masses < 120.0)) | \
                        ((masses > 130.0) & (masses < 160.0))

        penalty = 0.0
        # For each region (SR or CR), check background in the sideband
        for i in range(n_srs + n_crs):
            region_mask = (region_ids == i)
            # Weighted background in sideband
            bkg_sideband = np.sum(weights[region_mask & sideband_mask & (labels != signal_class)])
            if bkg_sideband < min_bkg_sideband:
                # Apply a penalty for low sideband stats
                # Adjust magnitude to taste
                penalty -= 1000.0

        # === 7) Final objective: SR + CR significance - penalty ===
        Z = total_sr_significance + total_cr_significance + penalty

        return Z

        # Store the best parameters for this trial        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Save optimization history and parallel coordinates plots.
    plot_optuna_history(study, out_dir, category=1)
    plot_parallel_coordinates(study, out_dir, category=1)

    print("Best significance:", study.best_value)
    print("Best parameters:", study.best_params)

    

    return study

#############################################
# Sequential categorization using Optuna
#############################################

def get_best_cut_params_using_optuna(n_categories, samples_input, out_dir, signal_class=3, n_trials=150, side_band_threshold=10):
    """
    For each category, use Optuna to find the best multidimensional cuts:
      - The signal score must be above a threshold.
      - Each background score must be below its respective threshold.
    In subsequent categories, the optimization search ranges are restricted:
      - Signal: (0, previous_best_signal_threshold)
      - Background (for each class): (previous_best_bg_threshold, 1)
    Additionally, the background in the sidebands (mass <120 or >130) must sum to at least 10.
    Selected events (based on a significance metric in the diphoton mass window 120–130 GeV)
    are removed from the DataFrame before optimizing the next category.
    For each category, the code stores:
      - The significance (Z)
      - The weighted signal in the peak (120-130 GeV)
      - The weighted background in the sidebands
    """
    cat_path = os.path.join(out_dir, "optuna_categorization")
    print(f"Creating output directory: {cat_path}")
    os.makedirs(cat_path, exist_ok=True)
    
    best_cut_params_list = []  # Best parameters per category.
    best_sig_values = []       # Best significance values per category.
    sig_peak_list = []         # Weighted signal in peak region for each category.
    bkg_side_list = []         # Weighted background in sideband for each category.
    mask_list = []           # Mask for selected events in each category.
    
    # Begin with all events.
    samples_remaining = samples_input.copy()

    # Determine background classes using the first event.
    first_scores = np.stack(samples_input["score"].values)
    n_classes = first_scores.shape[1]
    bg_classes = [j for j in range(n_classes) if j != signal_class]
    
    # Initialize dynamic search ranges.
    prev_signal_cut = 1.0  # For signal score: initial range is (0, 1).
    # For each background class, since we want to cut "less than" a threshold, we update the lower bound.
    prev_bg_cut = {b: 0.0 for b in bg_classes}

    for cat in range(1, n_categories + 1):
        print(f"\n--- Optimizing Category {cat} ---")
        scores = np.stack(samples_remaining["score"].values)  # Shape: (n_events, n_classes)
        dipho_mass = samples_remaining["diphoton_mass"].values
        labels = samples_remaining["labels"].values
        weights = samples_remaining["weights"].values
        samples = samples_remaining["sample"].values

        print(f"Signal threshold search range: (0, {prev_signal_cut})")
        for b in bg_classes:
            print(f"Background class {b} threshold search range: ({prev_bg_cut[b]}, 1)")

        def objective(trial):
            # Signal threshold search restricted to (0, prev_signal_cut)
            th_signal = round(trial.suggest_float("th_signal", 0, prev_signal_cut), 5)
            mask = scores[:, signal_class] > th_signal
            # Background thresholds for each background class, search in (prev_bg_cut[b], 1)
            for b in bg_classes:
                th_bg = round(trial.suggest_float(f"th_bg_{b}", prev_bg_cut[b], 1), 5)
                mask = mask & (scores[:, b] < th_bg)
            
            if np.sum(mask) == 0:
                return -1.0
            
            # Enforce sideband requirement: background outside 120-130 GeV must have at least 10 (weighted).
            side_mask = (((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130)) & (labels[mask] == 0))
            bkg_side_val = weights[mask][side_mask].sum()
            if bkg_side_val < side_band_threshold:
                return -1.0
            
            # Select events in the diphoton mass window (120 < m_γγ < 130).
            mass_mask = (dipho_mass[mask] > 120) & (dipho_mass[mask] < 130)
            if np.sum(mass_mask) == 0:
                return -1.0

            selected_labels = labels[mask][mass_mask]
            selected_weights = weights[mask][mass_mask]
            df_temp = pd.DataFrame({"labels": selected_labels, "weights": selected_weights})
            sig_val = asymptotic_significance(df_temp)
            return sig_val

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_target = study.best_value
        best_cut_params_list.append(best_params)
        best_sig_values.append(best_target)
        print(f"Category {cat}: Best parameters: {best_params} with significance {best_target:.4f}")

        # Save optimization history and parallel coordinates plots.
        plot_optuna_history(study, cat_path, category=cat)
        plot_parallel_coordinates(study, cat_path, category=cat)

        # Use best parameters to select events for this category.
        mask = scores[:, signal_class] > best_params["th_signal"]
        for b in bg_classes:
            mask = mask & (scores[:, b] < best_params[f"th_bg_{b}"])
        
        selected_mask = mask
        selected_events = samples_remaining[selected_mask]
        #plot_mass_category(selected_events, cat_path, category=cat)

        # Compute the weighted signal in the peak region (120-130) and background in the sidebands.
        sel_df = selected_events
        signal_in_peak = sel_df[(sel_df["labels"] == 1) & 
                                (sel_df["diphoton_mass"] > 120) & 
                                (sel_df["diphoton_mass"] < 130)]["weights"].sum()
        bkg_in_side = sel_df[(sel_df["labels"] == 0) & 
                             (((sel_df["diphoton_mass"] < 120) | (sel_df["diphoton_mass"] > 130)))]["weights"].sum()
        sig_peak_list.append(signal_in_peak)
        bkg_side_list.append(bkg_in_side)

        # Update dynamic search ranges.
        prev_signal_cut = best_params["th_signal"]
        for b in bg_classes:
            prev_bg_cut[b] = best_params[f"th_bg_{b}"]

        # Remove selected events from the remaining DataFrame.
        samples_remaining = samples_remaining[~selected_mask]
        if len(samples_remaining) == 0:
            print("No events remaining for further categorization.")
            break

    # Plot summary: cumulative Z, per-category Z, signal in peak, and bkg in sideband vs category.
    #plot_category_summary(best_sig_values, sig_peak_list, bkg_side_list, cat_path)
    # Plot summary with thresholds annotated on the significance plot.
    plot_category_summary_with_thresholds(best_sig_values,
    sig_peak_list,
    bkg_side_list,
    best_cut_params_list,
    cat_path)

    # save the best cut parameters
    best_params_path = os.path.join(cat_path, "best_cut_params.txt")
    with open(best_params_path, "w") as f:
        for i, params in enumerate(best_cut_params_list, start=1):
            f.write(f"Category {i}:\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    # also save it as a JSON file
    best_params_json_path = os.path.join(cat_path, "best_cut_params.json")
    with open(best_params_json_path, "w") as f:
        json.dump(best_cut_params_list, f, indent=4)


    # plot the diphoton and dijet mass for each category for data
    #plot_mass_sideband_data(best_cut_params_list, out_dir, cat_path)
    

    return best_cut_params_list, best_sig_values

def remove_selected_events(samples_df, best_cut_params_list, signal_class=3):
    """
    Given a DataFrame and a subset of SR category best-cut-parameters,
    remove all events that pass *any* of those category cuts.
    Returns a tuple of (filtered_df, mask_of_removed_events).
    """
    # Start with no event removed
    remove_mask = np.zeros(len(samples_df), dtype=bool)
    
    scores_all = np.stack(samples_df["score"].values)
    n_classes = scores_all.shape[1]
    bg_classes = [c for c in range(n_classes) if c != signal_class]

    current_df = samples_df.copy()

    for params in best_cut_params_list:
        # Build a mask *relative* to the full dataset
        th_signal = params["th_signal"]
        tmp_mask = scores_all[:, signal_class] > th_signal
        for b in bg_classes:
            th_bg = params[f"th_bg_{b}"]
            tmp_mask &= (scores_all[:, b] < th_bg)

        # Mark these events as removed
        remove_mask |= tmp_mask

        # Optionally, if you want the EXACT same sequential approach that the
        # original function used, you’d have to do it in a truly sequential manner:
        # e.g. recalculate the new 'scores_all' for the leftover events. But if
        # the categories are strictly orthogonal, you might be okay just combining them.
        #
        # For perfect replication of your get_best_cut_params_using_optuna approach,
        # you would do the selection *in sequence*, but that’s more code.
        #
        # This simpler approach lumps them all together, which is typically fine
        # if the categories are orthogonal cuts.

    filtered_df = samples_df[~remove_mask].copy()
    return filtered_df, remove_mask

def get_best_cut_params_for_cr(
    samples_input,
    out_dir,
    cr_classes=[1, 2],
    cr_class_names=[["ttHtoGG_M_125"], ["BBHto2G_M_125", "GluGluHToGG_M_125"]],
    cr_name = ['ttH', "ggH_bbH"],
    n_trials=150,
    sideband_threshold=10.0,
):
    """
    Optimize one category for each CR class in cr_classes, treating each CR class
    as a 'signal' for a significance-based cut. All other classes become 'background'.

    :param samples_input: DataFrame of *remaining* events after SR selection is removed.
                         Columns assumed:
                           - "score": np.array of shape (n_events, n_classes)
                           - "diphoton_mass": floats
                           - "labels": integer labels [0..n_classes-1]
                           - "weights": event weights
                           - "sample": sample names (e.g. ["ttHtoGG_M_125", ...])
    :param out_dir: Output directory for logs/plots.
    :param cr_classes: The list of class indices that we want to treat as “signal”
                       for each CR category (e.g., [1, 2]).
    :param cr_class_names: For human-readable reference or if you do special logic
                           based on sample names (e.g. [[“ttHtoGG_M_125”], [“BBHto2G_M_125”, ...]]).
    :param n_trials: Number of Optuna trials for each CR category.
    :param sideband_threshold: Weighted number of background events required in sidebands.
    :param significance_func: A function for calculating significance. If None,
                              you can reuse the asymptotic_significance from SR code,
                              but re-labeled for CR if needed.

    :return: (cr_best_params_list, cr_best_significance_list)
             - cr_best_params_list: list of best parameters (dict) for each CR category
             - cr_best_significance_list: list of best significance values
    """
    print(samples_input.columns)

    def asymptotic_significance_cr(df, sig_sample_list):
        # get signal by sum of weights for the signal sample
        s = df[df["sample"].isin(sig_sample_list)]["weights"].sum()
        # get background by sum of weights for the background sample
        b = df[~df["sample"].isin(sig_sample_list)]["weights"].sum()
        print(f"Signal: {s}, Background: {b}")
        z = np.sqrt(2 * ((s + b + 1e-10) * np.log(1 + (s / (b + 1e-10))) - s))
        #z = np.sqrt(2 * (((s + b + 1e-10) * np.log(1 + s / (b + 1e-10))) - s))
        print(f"Signal: {s}, Background: {b}, Significance: {z}")
        return z

    cr_path = os.path.join(out_dir, "optuna_categorization")
    os.makedirs(cr_path, exist_ok=True)

    cr_best_params_list = []
    cr_info = {
        "cr_names": [],
        "cr_significance": [],
        "cr_bkg_sideband": [],
    }

    # Copy the events that remain after SR is removed
    samples_remaining = samples_input.copy()

    # Number of total classes
    first_scores = np.stack(samples_remaining["score"].values)
    n_classes = first_scores.shape[1]

    # Loop over each CR class you want to treat as “signal”
    for i_cr, cr_class_idx in enumerate(cr_classes):
        print(f"\n=== Optimizing CR for class index {cr_class_idx} ===")
        print(f"   CR samples: {cr_class_names[i_cr] if i_cr < len(cr_class_names) else 'Unknown'}")

        scores = np.stack(samples_remaining["score"].values)
        dipho_mass = samples_remaining["diphoton_mass"].values
        labels = samples_remaining["labels"].values
        weights = samples_remaining["weights"].values

        # Everything except cr_class_idx is "background" for this CR
        bg_classes = [c for c in range(n_classes) if c != cr_class_idx]

        def objective(trial):
            # threshold for CR "signal" class
            th_cr = trial.suggest_float(f"th_cr_{cr_class_idx}", 0.0, 1.0)
            mask = scores[:, cr_class_idx] > th_cr

            #th_cr_up = trial.suggest_float(f"th_cr_{cr_class_idx}_up", 0.0, 1.0)
            #mask &= scores[:, cr_class_idx] < th_cr_up


            # For each background class, define a threshold so their score is below it
            for b in bg_classes:
                if b == 5:
                    th_bg = trial.suggest_float(f"th_bg_{b}", 0.0, 0.03)
                else:
                    th_bg = trial.suggest_float(f"th_bg_{b}", 0.0, 1.0)
                mask &= (scores[:, b] < th_bg)

            # If nothing passes the cut, significance is trivial
            if mask.sum() == 0:
                return -1.0

            # Sideband requirement: the weighted number of bkg events (label != cr_class_idx)
            # outside the 120-130 GeV window must be >= sideband_threshold
            side_mask = ((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130)) & (labels[mask] != cr_class_idx)
            bkg_side_val = weights[mask][side_mask].sum()
            if bkg_side_val < sideband_threshold:
                return -1.0

            # Now compute significance inside the 120-130 window
            mass_mask = (dipho_mass[mask] > 120) & (dipho_mass[mask] < 130)
            if np.sum(mass_mask) == 0:
                return -1.0

            selected_labels = labels[mask][mass_mask]
            selected_weights = weights[mask][mass_mask]
            selected_samples = samples_remaining["sample"].values[mask][mass_mask]

            df_temp = pd.DataFrame({"labels": selected_labels, "weights": selected_weights, "sample": selected_samples})
            sig_val = asymptotic_significance_cr(df_temp, cr_class_names[i_cr])
            return sig_val

        # Run Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_sig = study.best_value

        mask = scores[:, cr_class_idx] > best_params[f"th_cr_{cr_class_idx}"]
        for b in bg_classes:
            mask &= (scores[:, b] < best_params[f"th_bg_{b}"])
        selected_mask = mask

        cr_best_params_list.append(best_params)
        cr_info["cr_names"].append(cr_class_names[i_cr])
        cr_info["cr_significance"].append(best_sig)

        # Save optimization history and parallel coordinates plots.
        plot_optuna_history(study, cr_path, category=f"{cr_name[i_cr]}")
        plot_parallel_coordinates(study, cr_path, category=f"{cr_name[i_cr]}")


        print(f"CR for class {cr_class_idx} best params = {best_params},  significance = {best_sig:.4f}")

        # Optionally save study plots, e.g.:
        # plot_optuna_history(study, cr_path, category=f"CR{cr_class_idx}")
        # plot_parallel_coordinates(study, cr_path, category=f"CR{cr_class_idx}")

        # Use best params to select events for this CR
        final_mask = np.ones(len(samples_remaining), dtype=bool)
        final_mask &= (scores[:, cr_class_idx] > best_params[f"th_cr_{cr_class_idx}"])
        for b in bg_classes:
            final_mask &= (scores[:, b] < best_params[f"th_bg_{b}"])

        selected_events_cr = samples_remaining[final_mask]
        #bkg_side_val = selected_events_cr[final_mask][
        #    ((samples_remaining["diphoton_mass"] < 120) | (samples_remaining["diphoton_mass"] > 130))
        #]["weights"].sum()
        #cr_info["cr_bkg_sideband"].append(bkg_side_val)
        # Optionally, you can create a plot of the mass distribution for these CR events
        # plot_mass_category(selected_events_cr, cr_path, category=f"CR_{cr_class_idx}")

        # Remove these CR events to keep subsequent CR definitions orthogonal
        samples_remaining = samples_remaining[~final_mask]
        if len(samples_remaining) == 0:
            print("No events left for further CR categories.")
            break

    # Save CR best parameters
    cr_params_json_path = os.path.join(cr_path, "best_cr_cut_params.json")
    with open(cr_params_json_path, "w") as f:
        json.dump(cr_best_params_list, f, indent=4)
    print(cr_params_json_path)
    # Save CR best significance
    cr_sig_json_path = os.path.join(cr_path, "cr_info.json")
    with open(cr_sig_json_path, "w") as f:
        json.dump(cr_info, f, indent=4)

    return cr_best_params_list, cr_info

"""def store_categorization_events_with_score(base_path, best_cut_values):

    # create the output directory
    out_dir = f"{base_path}/optuna_categorization/"
    os.makedirs(out_dir, exist_ok=True)

    # category sim
    samples = [
                "GGJets",
                "DDQCDGJET",
                "TTGG",
                "ttHtoGG_M_125",
                "BBHto2G_M_125",
                "GluGluHToGG_M_125",
                "VBFHToGG_M_125",
                "VHtoGG_M_125",
                "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
                #"VBFHHto2B2G_CV_1_C2V_1_C3_1"
                ]
    for era in ["preEE", "postEE"]:
        for sample in samples:
            inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
            print(f"Processing {inputs_path}")
            # events parquet
            events = ak.from_parquet(inputs_path)
            

            # load score and get max score
            scores = np.load(f"{inputs_path}/y.npy")

            # load rel_w
            rel_w = np.load(f"{inputs_path}/rel_w.npy")
            #events["rel_w"] = rel_w
            events["weight_tot"] = rel_w
            #events["dijet_mass"] = events["dijet_mass"]

            selected_events = events
            selected_scores = scores

            for i in range(2):

                # get mask for category
                score_cuts = best_cut_values[i]
                mask = selected_scores[:, 3] > score_cuts["th_signal"]

                for b in [0, 1, 2]:
                    mask = mask & (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

                # store events, y, rel_w
                os.makedirs(f"{out_dir}/cat{i+1}/{era}/{sample}", exist_ok=True)
                ak.to_parquet(selected_events[mask], f"{out_dir}/cat{i+1}/{era}/{sample}/events.parquet")
                np.save(f"{out_dir}/cat{i+1}/{era}/{sample}/y.npy", selected_scores[mask])

                selected_events = selected_events[~mask]
                selected_scores = selected_scores[~mask]

            # also store singleH region
            singleH_cut = best_cut_values[1]
            mask = scores[:, 3] < singleH_cut["th_signal"]
            mask = mask & (scores[:, 2] > singleH_cut["th_bg_2"])
            mask = mask & (scores[:, 2] > 0.2)
            for b in [0, 1]:
                mask = mask & (scores[:, b] < 0.2)
            os.makedirs(f"{out_dir}/singleH_enriched/{era}/{sample}", exist_ok=True)
            ak.to_parquet(events[mask], f"{out_dir}/singleH_enriched/{era}/{sample}/events.parquet")
            np.save(f"{out_dir}/singleH_enriched/{era}/{sample}/y.npy", scores[mask])
            mask = mask & (scores[:, 2] > 0.85)
            os.makedirs(f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{era}/{sample}", exist_ok=True)
            ak.to_parquet(events[mask], f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{era}/{sample}/events.parquet")
            np.save(f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{era}/{sample}/y.npy", scores[mask])

            # for ttH enriched region
            ttH_cut = best_cut_values[1]
            mask = scores[:, 3] < ttH_cut["th_signal"]
            mask = mask & (scores[:, 1] > ttH_cut["th_bg_1"])
            mask = mask & (scores[:, 1] > 0.2)
            for b in [0, 2]:
                mask = mask & (scores[:, b] < 0.2)
            os.makedirs(f"{out_dir}/ttH_enriched/{era}/{sample}", exist_ok=True)
            ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched/{era}/{sample}/events.parquet")
            np.save(f"{out_dir}/ttH_enriched/{era}/{sample}/y.npy", scores[mask])
            mask = mask & (scores[:, 1] > 0.85)
            os.makedirs(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{era}/{sample}", exist_ok=True)
            ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{era}/{sample}/events.parquet")
            np.save(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{era}/{sample}/y.npy", scores[mask])

    # category data
    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        print(f"Processing {inputs_path}")
        # events parquet
        events = ak.from_parquet(inputs_path)

        # load score and get max score
        scores = np.load(f"{inputs_path}/y.npy")

        selected_events = events
        selected_scores = scores

        for i in range(2):

            # get mask for category
            score_cuts = best_cut_values[i]
            mask = selected_scores[:, 3] > score_cuts["th_signal"]

            for b in [0, 1, 2]:
                mask = mask & (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

            # remove mass in 120-130
            #mask = mask & ((selected_events["mass"] < 120) | (selected_events["mass"] > 130))

            selected_events["dijet_mass"] = selected_events["nonRes_mjj_regressed"]
            selected_events["weight_tot"] = ak.ones_like(selected_events["dijet_mass"])

            # store events, y, rel_w
            os.makedirs(f"{out_dir}/cat{i+1}/{data_sample}", exist_ok=True)
            ak.to_parquet(selected_events[mask], f"{out_dir}/cat{i+1}/{data_sample}/events.parquet")
            np.save(f"{out_dir}/cat{i+1}/{data_sample}/y.npy", selected_scores[mask])

            selected_events = selected_events[~mask]
            selected_scores = selected_scores[~mask]

        # also store singleH region
        singleH_cut = best_cut_values[1]
        mask = scores[:, 3] < singleH_cut["th_signal"]
        mask = mask & (scores[:, 2] > singleH_cut["th_bg_2"])
        mask = mask & (scores[:, 2] > 0.2)
        for b in [0, 1]:
            mask = mask & (scores[:, b] < 0.2)
        os.makedirs(f"{out_dir}/singleH_enriched/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/singleH_enriched/{data_sample}/events.parquet")
        np.save(f"{out_dir}/singleH_enriched/{data_sample}/y.npy", scores[mask])
        mask = mask & (scores[:, 2] > 0.85)
        os.makedirs(f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{data_sample}/events.parquet")
        np.save(f"{out_dir}/singleH_enriched_singleH_score_gt_0p85/{data_sample}/y.npy", scores[mask])

        # for ttH enriched region
        ttH_cut = best_cut_values[1]
        mask = scores[:, 3] < ttH_cut["th_signal"]
        mask = mask & (scores[:, 1] > ttH_cut["th_bg_1"])
        mask = mask & (scores[:, 1] > 0.2)
        for b in [0, 2]:
            mask = mask & (scores[:, b] < 0.2)
        os.makedirs(f"{out_dir}/ttH_enriched/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched/{data_sample}/events.parquet")
        np.save(f"{out_dir}/ttH_enriched/{data_sample}/y.npy", scores[mask])
        mask = mask & (scores[:, 1] > 0.85)
        os.makedirs(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}/events.parquet")
        np.save(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}/y.npy", scores[mask])

    return 0"""

"""def store_categorization_events_with_score(
    base_path,
    best_cut_values,
    best_cut_params_cr=None,
    cr_classes=None,
    cr_class_names=None,
    cr_name=None
):


    out_dir = f"{base_path}/optuna_categorization/"
    os.makedirs(out_dir, exist_ok=True)

    # Dictionary to store yields: {(era, sample, category_name): (sum_w, sum_w_sq)}
    yields_info = {}

    def update_yields_info(sample, category_name, events):
        Accumulate sum of weights and sum of weights^2 for the given events.
        sum_w = float(ak.sum(events["weight_tot"]))
        sum_w_sq = float(ak.sum(events["weight_tot"]**2))
        old_sum_w, old_sum_w_sq = yields_info.get((sample, category_name), (0.0, 0.0))
        yields_info[(sample, category_name)] = (old_sum_w + sum_w, old_sum_w_sq + sum_w_sq)

    # ---------------------
    # Process MC samples
    # ---------------------
    samples = [
        "GGJets",
        "DDQCDGJET",
        "TTGG",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        #"VBFHHto2B2G_CV_1_C2V_1_C3_1"
        "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
        "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00"
    ]
    #dijet_mass_key = "nonRes_mjj_regressed"
    dijet_mass_key = "nonRes_dijet_mass"
    for era in ["preEE", "postEE"]:
        for sample in samples:
            inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
            print(f"Processing {inputs_path}")

            # Load events
            events = ak.from_parquet(inputs_path)
            #btag_mask = events[events["nBLoose"]>=1]
            

            scores = np.load(f"{inputs_path}/y.npy")
            

            # Load weights
            rel_w = np.load(f"{inputs_path}/rel_w.npy")
            events["weight_tot"] = rel_w
            events["dijet_mass"] = events[dijet_mass_key]

            #diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
            #dijet_mass_cut = ((events.nonRes_mjj_regressed > 70) & (events.nonRes_mjj_regressed < 190))
            #nonRes = (events.nonRes_has_two_btagged_jets == True)
            #events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #events = events[btag_mask]

            #scores = scores[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #scores = scores[btag_mask]

            # Keep track of leftover (start with everything)
            selected_events = events
            selected_scores = scores

            # ============= SR Categories (cat1, cat2) =============
            for i in range(2):
                score_cuts = best_cut_values[i]

                mask = (selected_scores[:, 3] > score_cuts["th_signal"])
                for b in [0, 1, 2]:
                    mask &= (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

                # Store to cat{i+1}
                cat_outdir = f"{out_dir}/cat{i+1}/{era}/{sample}"
                os.makedirs(cat_outdir, exist_ok=True)

                cat_events = selected_events[mask]
                ak.to_parquet(cat_events, f"{cat_outdir}/events.parquet")
                np.save(f"{cat_outdir}/y.npy", selected_scores[mask])

                # Update yields (combine era)
                cat_name = f"cat{i+1}"
                update_yields_info(sample, cat_name, cat_events)

                # Remove from leftover
                selected_events = selected_events[~mask]
                selected_scores = selected_scores[~mask]

            # ============= CR Categories (if provided) =============
            # We only do this step if best_cut_params_cr is not None
            if best_cut_params_cr is not None and cr_classes is not None:
                # `best_cut_params_cr` is typically a list of dicts, one per CR category
                # each dict might look like: {"th_cr_1":..., "th_bg_0":..., "th_bg_2":..., ...}

                leftover_events = selected_events
                leftover_scores = selected_scores

                # Apply btag mask
                #btag_mask = (leftover_events["nBLoose"] >= 2)
                #leftover_events = leftover_events[btag_mask]
                #leftover_scores = leftover_scores[btag_mask]

                # Loop over each CR category
                for i_cr, cr_params in enumerate(best_cut_params_cr):
                    # The class we consider "signal" for this CR
                    cr_class_idx = cr_classes[i_cr]  
                    # e.g. cr_class_idx = 1 or 2, etc.

                    # Build the mask from the CR parameters
                    # For example, your get_best_cut_params_for_cr might produce keys:
                    #   "th_cr_{cr_class_idx}" = threshold for the "signal" score
                    #   "th_bg_x" = thresholds for other classes 
                    # We'll just replicate the same logic you used in your CR objective

                    mask_cr = np.ones(len(leftover_events), dtype=bool)
                    # threshold for "signal" class
                    thr_signal_key = f"th_cr_{cr_class_idx}"
                    mask_cr &= (leftover_scores[:, cr_class_idx] > cr_params[thr_signal_key])

                    #mask_cr &= (leftover_scores[:, cr_class_idx] < cr_params[thr_signal_key+"_up"])

                    # Then for each background class, we do "scores[:, b] < cr_params[f'th_bg_{b}']"
                    n_classes = leftover_scores.shape[1]
                    bg_list = [c for c in range(n_classes) if c != cr_class_idx]
                    for b_idx in bg_list:
                        thr_bg_key = f"th_bg_{b_idx}"
                        if thr_bg_key in cr_params:
                            mask_cr &= (leftover_scores[:, b_idx] < cr_params[thr_bg_key])

                    # Make an output dir for this CR category name
                    # e.g. cr_name[i_cr] might be "CR_ttH" or "CR_singleH"
                    cr_out_name = cr_name[i_cr] if cr_name else f"CR_{cr_class_idx}"
                    cr_outdir = f"{out_dir}/{cr_out_name}/{era}/{sample}"
                    os.makedirs(cr_outdir, exist_ok=True)

                    cr_events = leftover_events[mask_cr]
                    ak.to_parquet(cr_events, f"{cr_outdir}/events.parquet")
                    np.save(f"{cr_outdir}/y.npy", leftover_scores[mask_cr])

                    # Update yields (combine era)
                    update_yields_info(sample, cr_out_name, cr_events)

                    # Remove from leftover if you want them orthogonal
                    leftover_events = leftover_events[~mask_cr]
                    leftover_scores = leftover_scores[~mask_cr]

            # Done with this sample

    # ----------------------
    # Process Data samples
    # ----------------------
    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        print(f"Processing {inputs_path}")

        events = ak.from_parquet(inputs_path)

        #btag_mask = events[events["nBLoose"]>=1]
        

        scores = np.load(f"{inputs_path}/y.npy")
        

        # For data, define "weight_tot" = 1
        events["weight_tot"] = ak.ones_like(events[dijet_mass_key])
        events["dijet_mass"] = events[dijet_mass_key]

        #diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
        #dijet_mass_cut = ((events.nonRes_mjj_regressed > 70) & (events.nonRes_mjj_regressed < 190))
        #nonRes = (events.nonRes_has_two_btagged_jets == True)
        #events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]
        #events = events[btag_mask]

        #scores = scores[diphoton_mass_cut & dijet_mass_cut & nonRes]
        #scores = scores[btag_mask]

        selected_events = events
        selected_scores = scores

        # ============= SR (cat1, cat2) =============
        for i in range(2):
            score_cuts = best_cut_values[i]
            mask = (selected_scores[:, 3] > score_cuts["th_signal"])
            for b in [0, 1, 2]:
                mask &= (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

            cat_outdir = f"{out_dir}/cat{i+1}/{data_sample}"
            os.makedirs(cat_outdir, exist_ok=True)

            cat_events = selected_events[mask]
            ak.to_parquet(cat_events, f"{cat_outdir}/events.parquet")
            np.save(f"{cat_outdir}/y.npy", selected_scores[mask])

            # Update yields (since data_sample is effectively "data era X", 
            # we can just keep that as the "sample" key or unify them as "Data")
            # For simplicity, let's keep data_sample as is:
            cat_name = f"cat{i+1}"
            update_yields_info(data_sample, cat_name, cat_events)

            selected_events = selected_events[~mask]
            selected_scores = selected_scores[~mask]

        # ============= CR Categories (if provided) =============
        if best_cut_params_cr is not None and cr_classes is not None:
            leftover_events = selected_events
            leftover_scores = selected_scores

            # Apply btag mask
            #btag_mask = (leftover_events["nBLoose"] >= 2)
            #leftover_events = leftover_events[btag_mask]
            #leftover_scores = leftover_scores[btag_mask]


            for i_cr, cr_params in enumerate(best_cut_params_cr):
                cr_class_idx = cr_classes[i_cr]

                mask_cr = np.ones(len(leftover_events), dtype=bool)
                thr_signal_key = f"th_cr_{cr_class_idx}"
                mask_cr &= (leftover_scores[:, cr_class_idx] > cr_params[thr_signal_key])
                #mask_cr &= (leftover_scores[:, cr_class_idx] < cr_params[thr_signal_key+"_up"])

                n_classes = leftover_scores.shape[1]
                bg_list = [c for c in range(n_classes) if c != cr_class_idx]
                for b_idx in bg_list:
                    thr_bg_key = f"th_bg_{b_idx}"
                    if thr_bg_key in cr_params:
                        mask_cr &= (leftover_scores[:, b_idx] < cr_params[thr_bg_key])

                cr_out_name = cr_name[i_cr] if cr_name else f"CR_{cr_class_idx}"
                cr_outdir = f"{out_dir}/{cr_out_name}/{data_sample}"
                os.makedirs(cr_outdir, exist_ok=True)

                cr_events = leftover_events[mask_cr]
                ak.to_parquet(cr_events, f"{cr_outdir}/events.parquet")
                np.save(f"{cr_outdir}/y.npy", leftover_scores[mask_cr])

                # Update yields
                update_yields_info(data_sample, cr_out_name, cr_events)


                leftover_events = leftover_events[~mask_cr]
                leftover_scores = leftover_scores[~mask_cr]

    # ----------------------
    # Write combined yields to a text file
    # ----------------------
    out_txt = os.path.join(out_dir, "categorization_yields.txt")
    with open(out_txt, "w") as f:
        f.write("sample, category, sum_of_weights, uncertainty\n")
        for (sample, category), (sum_w, sum_w_sq) in yields_info.items():
            unc = math.sqrt(sum_w_sq)
            f.write(f"{sample}, {category}, {sum_w}, {unc}\n")

    return 0"""


def store_categorization_events_with_score(
    base_path,
    best_cut_values,
    best_cut_params_cr=None,
    cr_classes=None,
    cr_class_names=None,
    cr_name=None
):
    """
    Categorize events into:
      - 2 SR categories (cat1 and cat2) from 'best_cut_values'
      - CR categories from 'best_cut_params_cr' (if provided),
        using leftover events after SR.
        
    Additionally, save the yields (sum of weights) and uncertainties
    (sqrt of sum of weights^2) per combined era (preEE+postEE) in a 
    structured format in 'categorization_yields.txt'.

    Args:
      base_path (str): Base directory for reading/writing.
      best_cut_values (list[dict]): 
          2 dictionaries for SR categories: cat1, cat2
          e.g. best_cut_values[0] => cat1 thresholds, best_cut_values[1] => cat2
      best_cut_params_cr (list[dict] or None): 
          If not None, these are the CR thresholds from get_best_cut_params_for_cr.
          Each element in this list corresponds to a CR category 
          (e.g. for each class in cr_classes).
      cr_classes (list[int] or None):
          The CR class indices (e.g. [1,2]) that you used in get_best_cut_params_for_cr.
      cr_class_names (list[list[str]] or None):
          For reference, e.g. [["ttHtoGG_M_125"], ["BBHto2G_M_125","GluGluHToGG_M_125"]].
      cr_name (list[str] or None):
          The name you want to give each CR category (e.g. ["CR_ttH", "CR_singleH"]).
    """

    out_dir = f"{base_path}/optuna_categorization/"
    os.makedirs(out_dir, exist_ok=True)

    # Dictionary to store yields combined across preEE and postEE.
    # Key = (sample, category), Value = (sum_w, sum_w_sq)
    yields_info = {}

    def update_yields_info(sample, category_name, events):
        """Accumulate sum of weights and sum of weights^2 for the given events."""
        sum_w = float(ak.sum(events["weight_tot"]))
        sum_w_sq = float(ak.sum(events["weight_tot"]**2))

        old_sum_w, old_sum_w_sq = yields_info.get((sample, category_name), (0.0, 0.0))
        yields_info[(sample, category_name)] = (old_sum_w + sum_w, old_sum_w_sq + sum_w_sq)

    # ---------------------
    # Process MC samples
    # ---------------------
    samples = [
        "GGJets",
        "DDQCDGJET",
        "TTGG",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        # "VBFHHto2B2G_CV_1_C2V_1_C3_1"
        "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
        "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
        "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00"
    ]
    dijet_mass_key = "nonRes_dijet_mass"
    apply_selection = False
    def selection(events, scores, apply_selection):
        # Apply selection criteria here
        # For example, you can filter events based on certain conditions
        # For now, let's just return all events
        if not apply_selection:
            return events, scores
        else:
            print("Applying selection criteria...")
            #nBLoose_cut = (events["nBLoose"] >= 2)
            #events = events[nBLoose_cut]
            #scores = scores[nBLoose_cut]
            # nBMedium cut
            nBMedium_cut = (events["nBMedium"] >= 1)
            events = events[nBMedium_cut]
            scores = scores[nBMedium_cut]

            return events, scores

    for era in ["preEE", "postEE", "preBPix", "postBPix"]:
        for sample in samples:
            inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
            print(f"Processing {inputs_path}")

            # Load events
            events = ak.from_parquet(inputs_path)
            scores = np.load(f"{inputs_path}/y.npy")

            # Load weights
            rel_w = np.load(f"{inputs_path}/rel_w.npy")
            events["weight_tot"] = rel_w
            events["dijet_mass"] = events[dijet_mass_key]

            # Apply selection if needed
            events, scores = selection(events, scores, apply_selection)

            # Keep track of leftover
            selected_events = events
            selected_scores = scores

            # ============= SR Categories (cat1, cat2) =============
            for i in range(2):
                score_cuts = best_cut_values[i]

                mask = (selected_scores[:, 3] > score_cuts["th_signal"])
                for b in [0, 1, 2]:
                    mask &= (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

                cat_outdir = f"{out_dir}/cat{i+1}/{era}/{sample}"
                os.makedirs(cat_outdir, exist_ok=True)

                cat_events = selected_events[mask]
                ak.to_parquet(cat_events, f"{cat_outdir}/events.parquet")
                np.save(f"{cat_outdir}/y.npy", selected_scores[mask])

                # Update yields (combine era)
                cat_name = f"cat{i+1}"
                update_yields_info(sample, cat_name, cat_events)

                # Remove from leftover
                selected_events = selected_events[~mask]
                selected_scores = selected_scores[~mask]

            # ============= CR Categories (if provided) =============
            if best_cut_params_cr is not None and cr_classes is not None:
                leftover_events = selected_events
                leftover_scores = selected_scores

                # Apply btag mask
                #btag_mask = (leftover_events["nBLoose"] >= 1)
                #leftover_events = leftover_events[btag_mask]
                #leftover_scores = leftover_scores[btag_mask]

                for i_cr, cr_params in enumerate(best_cut_params_cr):
                    cr_class_idx = cr_classes[i_cr]

                    mask_cr = np.ones(len(leftover_events), dtype=bool)
                    thr_signal_key = f"th_cr_{cr_class_idx}"
                    mask_cr &= (leftover_scores[:, cr_class_idx] > cr_params[thr_signal_key])

                    n_classes = leftover_scores.shape[1]
                    bg_list = [c for c in range(n_classes) if c != cr_class_idx]
                    for b_idx in bg_list:
                        thr_bg_key = f"th_bg_{b_idx}"
                        if thr_bg_key in cr_params:
                            mask_cr &= (leftover_scores[:, b_idx] < cr_params[thr_bg_key])

                    cr_out_name = cr_name[i_cr] if cr_name else f"CR_{cr_class_idx}"
                    cr_outdir = f"{out_dir}/{cr_out_name}/{era}/{sample}"
                    os.makedirs(cr_outdir, exist_ok=True)

                    cr_events = leftover_events[mask_cr]
                    ak.to_parquet(cr_events, f"{cr_outdir}/events.parquet")
                    np.save(f"{cr_outdir}/y.npy", leftover_scores[mask_cr])

                    # Update yields (combine era)
                    update_yields_info(sample, cr_out_name, cr_events)

                    # Remove from leftover
                    leftover_events = leftover_events[~mask_cr]
                    leftover_scores = leftover_scores[~mask_cr]

    # ----------------------
    # Process Data samples
    # ----------------------
    data_samples = [
        "2022_EraE",
        "2022_EraF",
        "2022_EraG",
        "2022_EraC",
        "2022_EraD",
        "2023_EraCv1to3",
        "2023_EraCv4",
        "2023_EraD"
    ]
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        print(f"Processing {inputs_path}")

        events = ak.from_parquet(inputs_path)
        scores = np.load(f"{inputs_path}/y.npy")

        # For data, weight_tot = 1
        events["weight_tot"] = ak.ones_like(events[dijet_mass_key])
        events["dijet_mass"] = events[dijet_mass_key]

        # Apply selection if needed
        events, scores = selection(events, scores, apply_selection)

        selected_events = events
        selected_scores = scores

        # ============= SR (cat1, cat2) =============
        for i in range(2):
            score_cuts = best_cut_values[i]
            mask = (selected_scores[:, 3] > score_cuts["th_signal"])
            for b in [0, 1, 2]:
                mask &= (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

            cat_outdir = f"{out_dir}/cat{i+1}/{data_sample}"
            os.makedirs(cat_outdir, exist_ok=True)

            cat_events = selected_events[mask]
            ak.to_parquet(cat_events, f"{cat_outdir}/events.parquet")
            np.save(f"{cat_outdir}/y.npy", selected_scores[mask])

            # Update yields 
            cat_name = f"cat{i+1}"
            update_yields_info(data_sample, cat_name, cat_events)

            selected_events = selected_events[~mask]
            selected_scores = selected_scores[~mask]

        # ============= CR Categories (if provided) =============
        if best_cut_params_cr is not None and cr_classes is not None:
            leftover_events = selected_events
            leftover_scores = selected_scores

            # Apply btag mask
            #btag_mask = (leftover_events["nBLoose"] >= 1)
            #leftover_events = leftover_events[btag_mask]
            #leftover_scores = leftover_scores[btag_mask]

            for i_cr, cr_params in enumerate(best_cut_params_cr):
                cr_class_idx = cr_classes[i_cr]

                mask_cr = np.ones(len(leftover_events), dtype=bool)
                thr_signal_key = f"th_cr_{cr_class_idx}"
                mask_cr &= (leftover_scores[:, cr_class_idx] > cr_params[thr_signal_key])

                n_classes = leftover_scores.shape[1]
                bg_list = [c for c in range(n_classes) if c != cr_class_idx]
                for b_idx in bg_list:
                    thr_bg_key = f"th_bg_{b_idx}"
                    if thr_bg_key in cr_params:
                        mask_cr &= (leftover_scores[:, b_idx] < cr_params[thr_bg_key])

                cr_out_name = cr_name[i_cr] if cr_name else f"CR_{cr_class_idx}"
                cr_outdir = f"{out_dir}/{cr_out_name}/{data_sample}"
                os.makedirs(cr_outdir, exist_ok=True)

                cr_events = leftover_events[mask_cr]
                ak.to_parquet(cr_events, f"{cr_outdir}/events.parquet")
                np.save(f"{cr_outdir}/y.npy", leftover_scores[mask_cr])

                # Update yields
                update_yields_info(data_sample, cr_out_name, cr_events)

                leftover_events = leftover_events[~mask_cr]
                leftover_scores = leftover_scores[~mask_cr]

    # ----------------------
    # Write combined yields to a text file, grouped by category
    # ----------------------
    out_txt = os.path.join(out_dir, "categorization_yields.txt")

    # Get a list of all categories
    all_categories = set(cat for (_, cat) in yields_info.keys())
    # Sort them (e.g. cat1, cat2, CR_...) if you prefer alphabetical order
    all_categories = sorted(list(all_categories))

    # Group by category
    with open(out_txt, "w") as f:
        for cat in all_categories:
            f.write(f"{cat}:\n")
            # Loop over all (sample, category) pairs
            for (sample, this_cat), (sum_w, sum_w_sq) in yields_info.items():
                if this_cat == cat:
                    unc = math.sqrt(sum_w_sq)
                    f.write(f"{sample}, {sum_w}, {unc}\n")
            f.write("\n")  # Blank line after each category

    return 0


def convert_to_root(base_path):
    import uproot
    import awkward as ak
    import numpy as np
    import os

    # category sim
    samples = [
                "GGJets",
                "DDQCDGJET",
                "TTGG",
                "ttHtoGG_M_125",
                "BBHto2G_M_125",
                "GluGluHToGG_M_125",
                "VBFHToGG_M_125",
                "VHtoGG_M_125",
                "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
                #"VBFHHto2B2G_CV_1_C2V_1_C3_1"
                "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
                "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
                "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00"
                ]
    # combine preEE and postEE and convert to root
    for sample in samples:
        preEE_events = ak.from_parquet(f"{base_path}/preEE/{sample}/events.parquet")
        postEE_events = ak.from_parquet(f"{base_path}/postEE/{sample}/events.parquet")
        events = ak.concatenate([preEE_events, postEE_events], axis=0)
        os.makedirs(f"{base_path}/root_files/", exist_ok=True)
        
        # create dict
        tree_dict = {field: ak.to_numpy(events[field]) for field in events.fields}
        root_outfile = f"{base_path}/root_files/{sample}.root"
        tree_name = "Events"
        with uproot.recreate(root_outfile) as f:
            f[tree_name] = tree_dict
        print(f"Wrote {root_outfile}")

    # for data
    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    # combine all the data samples and convert to root
    data_combined = None

    for data_sample in data_samples:
        data_part = ak.from_parquet(f"{data_folder}/{data_sample}/events.parquet")
        if data_combined is None:
            data_combined = data_part
        else:
            data_combined = ak.concatenate([data_combined, data_part], axis=0)

    os.makedirs(f"{base_path}/root_files/", exist_ok=True)
    # create dict
    tree_dict = {field: ak.to_numpy(data_combined[field]) for field in data_combined.fields}
    root_outfile = f"{base_path}/root_files/Data.root"
    tree_name = "Events"
    with uproot.recreate(root_outfile) as f:
        f[tree_name] = tree_dict
    print(f"Wrote {root_outfile}")


#############################################
# Main execution
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize multiclass scores using Optuna with dynamic search ranges, sideband requirements, and summary plots.")
    parser.add_argument("--n_categories", type=int, default=6, help="Number of categories to optimize")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the input samples")
    args = parser.parse_args()

    samples_list = [
        "GGJets",
        "DDQCDGJET",
        "TTGG",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        #"VBFHHto2B2G_CV_1_C2V_1_C3_1"
    ]
    
    samples_input = load_samples(args.base_path, samples_list)

    best_params, best_sig_values = get_best_cut_params_using_optuna(
        n_categories=args.n_categories,
        samples_input=samples_input,
        out_dir=args.base_path,
        signal_class=3,  # adjust if necessary
        n_trials=150,
        side_band_threshold=10.0,
    )

    # load the best cut values
    with open(f"{args.base_path}/optuna_categorization/best_cut_params.json", "r") as f:
        best_cut_values = json.load(f)


    chosen_k = 2  # e.g. you discovered that 2 SR categories saturates your significance
    best_cats_for_sr = best_cut_values[:chosen_k]

    samples_for_cr, sr_mask = remove_selected_events(
    samples_df=samples_input,
    best_cut_params_list=best_cats_for_sr,
    signal_class=3,
    )

    #samples_for_cr = samples_for_cr[samples_for_cr["nBLoose"]>=1]

    cr_classes = [2, 1]  # or whichever you desire
    cr_class_names = [
        #["BBHto2G_M_125", "GluGluHToGG_M_125"]
        ["BBHto2G_M_125"],
        ["ttHtoGG_M_125"]
    ]
    cr_name = ["bbH", 'ttH']

    best_cut_params_cr, best_sig_cr = get_best_cut_params_for_cr(
        samples_input=samples_for_cr,
        out_dir=args.base_path,
        cr_classes=cr_classes,
        cr_class_names=cr_class_names,
        cr_name=cr_name,
        n_trials=150,
        sideband_threshold=5.0,
    )

    # load the best cut values for CR
    with open(f"{args.base_path}/optuna_categorization/best_cr_cut_params.json", "r") as f:
        best_cr_cut_values = json.load(f)

    store_categorization_events_with_score(
    args.base_path,
    best_cut_values,
    best_cut_params_cr=best_cr_cut_values,
    cr_classes=cr_classes,
    cr_class_names=cr_class_names,
    cr_name=cr_name
    )


    folder_list = ["cat1", "cat2", "ttH"]#, "ggH", "bbH"]
    #folder_list = ["ttH", "ggH_bbH"]

    for folder in folder_list:
        opt_cat_folder_name = "optuna_categorization"
        sim_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        data_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        out_path = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        variables = ["mass", "dijet_mass", "nonRes_dijet_mass", "nonRes_mjj_regressed", "Res_mjj_regressed", "Res_dijet_mass"]
        #variables = ["mass", "dijet_mass", "Res_dijet_mass"]
        #variables = ["mass", "dijet_mass"]AC.4
        #if "kk" in folder:
        #    plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100, mask=False)
        #else:
        #    plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100, mask=True)
        plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100)

        #convert_to_root(out_path)