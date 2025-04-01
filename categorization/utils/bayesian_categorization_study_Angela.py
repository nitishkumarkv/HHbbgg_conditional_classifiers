
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
        data_dijet_mass = selected_events[mask]["nonRes_dijet_mass"]

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
        study.optimize(objective, n_trials=150, show_progress_bar=False)
        
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
                "VBFHHto2B2G_CV_1_C2V_1_C3_1"
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
                "VBFHHto2B2G_CV_1_C2V_1_C3_1"
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
        #plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100)
        convert_to_root(out_path)