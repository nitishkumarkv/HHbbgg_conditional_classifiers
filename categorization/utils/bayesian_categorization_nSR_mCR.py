
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import optuna

from tools import load_samples, asymptotic_significance, approx_significance
import pandas as pd
import json
import mplhep
from ploting import plot_stacked_histogram


import optuna
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import pi

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

def generalized_asimov_significance(
    s_array,         # shape (N,) — signal yields in SRs
    b_array,         # shape (N,) — total background in SRs (all backgrounds)
    b_components,    # shape (K, N) — per-process background contributions in SRs
    b_cr_array,      # shape (M,) — total background in CRs
    tau_matrix       # shape (M, K) — tau[j][k] = transfer factor: CR j gets bkg comp k
):
    """
    Implements Eq. (69) from arXiv:2102.04275.
    
    Arguments:
    ----------
    s_array : array-like, shape (N,)
        Expected signal yield in each SR.
    b_array : array-like, shape (N,)
        Total background yield in each SR (including constrained + unconstrained).
    b_components : array-like, shape (K, N)
        Background breakdown per process across SRs.
    b_cr_array : array-like, shape (M,)
        Total background in each CR.
    tau_matrix : array-like, shape (M, K)
        Transfer matrix: tau[j][k] = how much bkg comp k contributes to CR j.
    
    Returns:
    --------
    Z : float
        Asymptotic discovery significance.
    """
    s_array = np.array(s_array)
    b_array = np.array(b_array)
    b_cr_array = np.array(b_cr_array)
    b_components = np.array(b_components)
    tau_matrix = np.array(tau_matrix)

    N = len(s_array)
    K, N_b = b_components.shape
    M = len(b_cr_array)
    
    assert N == N_b, "Mismatch between SR count and b_component shape"
    assert tau_matrix.shape == (M, K), "Tau matrix should be (M, K)"

    n = np.sum(s_array + b_array)       # Total events in SRs
    m = b_cr_array                      # Total background in each CR
    b_hat_cr = m                       # Asimov: profiled background = observed

    # Estimate b_hat per background component as in Asimov case (just b_components sum over SR)
    b_hat_components = np.sum(b_components, axis=1)  # shape (K,)

    # Denominator of CR prediction: tau @ b_hat
    m_hat_cr = tau_matrix @ b_hat_components  # shape (M,)

    # term1: SR log term
    b_hat_total = np.sum(b_hat_cr)
    term1 = n * np.log(b_hat_total / n)

    # term2: sum over CRs
    term2 = 0.0
    for j in range(M):
        m_j = m[j]
        b_hat_j = b_hat_cr[j]
        num = np.sum(tau_matrix[j] * np.sum(b_components, axis=1))      
        denom = np.sum(tau_matrix[j] * b_hat_components)                

        log_term = np.log(num / denom) if denom > 0 and num > 0 else 0.0
        diff_term = np.sum(tau_matrix[j] * (np.sum(b_components, axis=1) - b_hat_components))

        term2 += (-b_hat_j + m_j) * log_term + diff_term

    Z2 = -2 * (term1 + n + term2)
    Z = np.sqrt(Z2) if Z2 > 0 else 0.0
    return Z

# Optuna-based optimization of N SRs and M CRs using multiclass classifier scores



def simple_asimov_significance(s, b):
    """
    Computes the simple Asimov significance:
    Z = sqrt(2 * ((s + b) * ln(1 + s / b) - s))
    """
    if b <= 0 or s <= 0:
        return 0.0
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))

def plot_radar_regions(best_params_list, class_names, n_srs, n_crs, out_path):
    os.makedirs(out_path, exist_ok=True)
    labels = class_names
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    cmap = plt.get_cmap("tab10")
    legend_elements = []

    for i, params in enumerate(best_params_list):
        values = []
        region_type = "SR" if i < n_srs else "CR"
        region_label = f"{region_type}{i if region_type == 'SR' else i - n_srs}"

        for cls in class_names:
            if f"th_{cls}" in params:
                values.append(params[f"th_{cls}"])  # lower cut
            elif f"th_{cls}_max" in params:
                values.append(params[f"th_{cls}_max"])  # upper cut
            elif f"th_{cls}_min" in params:
                values.append(params[f"th_{cls}_min"])  # lower cut
            else:
                values.append(0.0)  # default if not present

        values += values[:1]  # repeat first to close the loop
        color = cmap(i % 10)
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=region_label, color=color)
        ax.fill(angles, values, alpha=0.2, color=color)
        legend_elements.append(Patch(facecolor=color, label=region_label))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=10)
    ax.set_title("Cut thresholds per Region (Radar Plot)", fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "cut_radar_plot.png"))
    plt.close()


# Optuna-based optimization of N SRs and M CRs

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
    masses = samples_input["diphoton_mass"].values if "diphoton_mass" in samples_input.fields else samples_input["mass"].values

    bkg_classes = [i for i in range(n_classes) if i != signal_class]

    best_params_list = []

    def objective(trial):
        nonlocal best_params_list

        sr_thresholds = []
        for i in range(n_srs):
            sr_cut = {
                f"th_{class_names[signal_class]}": trial.suggest_float(f"sr_{i}_th_{class_names[signal_class]}", 0.0, 1.0),
            }
            for bg in bkg_classes:
                sr_cut[f"th_{class_names[bg]}_max"] = trial.suggest_float(f"sr_{i}_th_{class_names[bg]}_max", 0.0, 1.0)
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
                    cr_cut[f"th_{class_names[bg]}_max"] = trial.suggest_float(f"cr_{i}_th_{class_names[bg]}_max", 0.0, 1.0)
            cr_thresholds.append(cr_cut)

        region_ids = -1 * np.ones(len(scores), dtype=int)

        region_index = 0
        sr_masks = []
        for i, cut in enumerate(sr_thresholds):
            mask = scores[:, signal_class] > cut[f"th_{class_names[signal_class]}"]
            for bg in bkg_classes:
                mask &= scores[:, bg] < cut[f"th_{class_names[bg]}_max"]
            mask &= region_ids == -1
            region_ids[mask] = region_index
            sr_masks.append(mask)
            region_index += 1

        cr_masks = []
        if n_crs > 0:
            for i, cut in enumerate(cr_thresholds):
                target_cls = cut["target_class"]
                mask = (scores[:, signal_class] < cut[f"th_{class_names[signal_class]}_max"]) & \
                       (scores[:, target_cls] > cut[f"th_{class_names[target_cls]}_min"])
                for bg in bkg_classes:
                    if bg != target_cls:
                        mask &= scores[:, bg] < cut[f"th_{class_names[bg]}_max"]
                mask &= region_ids == -1
                region_ids[mask] = region_index
                cr_masks.append(mask)
                region_index += 1

        s_array, b_array, b_components = [], [], []
        total_bkg_sideband = 0.0

        for mask in sr_masks:
            s_array.append(weights[mask & (labels == 1)].sum())
            b_array.append(weights[mask & (labels == 0)].sum())
            comps = [weights[mask & (labels == 0) & (scores[:, c] > 0.3)].sum() for c in control_classes]
            b_components.append(comps)

            sideband_mask = (labels == 0) & mask & ((masses < 120) | (masses > 130))
            total_bkg_sideband += weights[sideband_mask].sum()

        b_components = np.array(b_components).T if b_components else np.zeros((0, n_srs))

        if total_bkg_sideband < min_bkg_sideband:
            return -1e6

        if n_crs == 0:
            Z_values = [simple_asimov_significance(s, b) for s, b in zip(s_array, b_array)]
            Z = np.sqrt(np.sum(np.array(Z_values) ** 2))
        else:
            b_cr_array = [weights[mask & (labels == 0)].sum() for mask in cr_masks]

            tau_matrix = np.zeros((n_crs, len(control_classes)))
            for j, mask in enumerate(cr_masks):
                for k, ctrl_cls in enumerate(control_classes):
                    b_in_cr = weights[mask & (labels == 0) & (scores[:, ctrl_cls] > 0.3)].sum()
                    b_in_sr = b_components[k].sum()
                    tau_matrix[j, k] = b_in_cr / b_in_sr if b_in_sr > 0 else 0.0

            Z = generalized_asimov_significance(
                s_array=np.array(s_array),
                b_array=np.array(b_array),
                b_components=b_components,
                b_cr_array=np.array(b_cr_array),
                tau_matrix=tau_matrix
            )

        best_params_list = sr_thresholds + cr_thresholds
        return Z

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best significance:", study.best_value)
    print("Best parameters:", study.best_params)

    plot_radar_regions(best_params_list, class_names, n_srs, n_crs, out_dir)
    return study

#############################################
# Sequential categorization using Optuna
#############################################

def get_best_cut_params_using_optuna(n_categories, samples_input, out_dir, signal_class=3):
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
    os.makedirs(cat_path, exist_ok=True)
    
    best_cut_params_list = []  # Best parameters per category.
    best_sig_values = []       # Best significance values per category.
    sig_peak_list = []         # Weighted signal in peak region for each category.
    bkg_side_list = []         # Weighted background in sideband for each category.
    
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
            if bkg_side_val < 10:
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
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        
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

def store_categorization_events_with_score(base_path, best_cut_values):

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
                "VBFHHto2B2G_CV_1_C2V_1_C3_1",
                "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
                "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00"
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
            events["dijet_mass"] = events["nonRes_dijet_mass"]

            selected_events = events
            selected_scores = scores

            for i in range(2):

                # get mask for category
                score_cuts = best_cut_values[i]
                mask = selected_scores[:, 3] > score_cuts["th_signal"]

                for b in [0, 1, 2, 4]:
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
            for b in [0, 1, 4]:
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
            for b in [0, 2, 4]:
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

            for b in [0, 1, 2, 4]:
                mask = mask & (selected_scores[:, b] < score_cuts[f"th_bg_{b}"])

            # remove mass in 120-130
            #mask = mask & ((selected_events["mass"] < 120) | (selected_events["mass"] > 130))

            selected_events["dijet_mass"] = selected_events["nonRes_dijet_mass"]
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
        for b in [0, 1, 4]:
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
        for b in [0, 2, 4]:
            mask = mask & (scores[:, b] < 0.2)
        os.makedirs(f"{out_dir}/ttH_enriched/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched/{data_sample}/events.parquet")
        np.save(f"{out_dir}/ttH_enriched/{data_sample}/y.npy", scores[mask])
        mask = mask & (scores[:, 1] > 0.85)
        os.makedirs(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}", exist_ok=True)
        ak.to_parquet(events[mask], f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}/events.parquet")
        np.save(f"{out_dir}/ttH_enriched_ttH_score_gt_0p85/{data_sample}/y.npy", scores[mask])

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
                "VBFHHto2B2G_CV_1_C2V_1_C3_1",
                "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
                "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00"
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
        "VBFHHto2B2G_CV_1_C2V_1_C3_1",
        #"GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00"
    ]
    
    samples_input = load_samples(args.base_path, samples_list)

    #best_params, best_sig_values = get_best_cut_params_using_optuna(
    #    n_categories=args.n_categories,
    #    samples_input=samples_input,
    #    out_dir=args.base_path,
    #    signal_class=3  # adjust if necessary
    #)

    # load the best cut values
    with open(f"{args.base_path}/optuna_categorization/best_cut_params.json", "r") as f:
        best_cut_values = json.load(f)

    # store the events in each category
    #store_categorization_events_with_score(args.base_path, best_cut_values)

    folder_list = ["cat1", "cat2", "singleH_enriched", "ttH_enriched", "singleH_enriched_singleH_score_gt_0p85", "ttH_enriched_ttH_score_gt_0p85"]

    for folder in folder_list:
        sim_folder = f"{args.base_path}/optuna_categorization/{folder}"
        data_folder = f"{args.base_path}/optuna_categorization/{folder}"
        out_path = f"{args.base_path}/optuna_categorization/{folder}"
        variables = ["mass", "nonRes_dijet_mass"]
        print("INFO: Plotting the stacked histogram for ", folder)
        #plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100)
        convert_to_root(out_path)