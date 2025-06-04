
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import optuna
import math
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import json
import mplhep
from matplotlib.backends.backend_pdf import PdfPages

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

def load_samples(base_path, samples):
    """Load predictions and weights, scaling weights by luminosity."""
    samples_input = {"score": [], "diphoton_mass": [], "dijet_mass": [], "weights": [], "labels": [], "sample": [], "arg_max_score": [], "nBLoose": [], "nBMedium": []}
    eras = ["preEE", "postEE", "preBPix", "postBPix"]
    #eras = ["preEE", "postEE"]
    #dijet_mass_key = "nonRes_mjj_regressed"
    #dijet_mass_key = "dijet_mass"
    dijet_mass_key = "Res_mjj_regressed"

    apply_selection = False
    def selection(events, apply_selection):
        # Apply selection criteria here
        # For example, you can filter events based on certain conditions
        # For now, let's just return all events
        if not apply_selection:
            return events
        else:
            print("Applying selection criteria...")
            #nBLoose_cut = (events["nBLoose"] >= 2)
            #events = events[nBLoose_cut]
            #nBMedium_cut = (events["nBMedium"] >= 2)
            #events = events[nBMedium_cut]

            return events

    for era in eras:
        for sample in samples:
            path = os.path.join(base_path, "individual_samples", era, sample)
            y_path = os.path.join(path, 'y.npy')
            #print("sample, y_shape[1]", sample, np.load(y_path).shape[1])
            w_path = os.path.join(path, 'rel_w.npy')
            events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=["mass", dijet_mass_key, "nBLoose", "nBMedium", "lead_genPartFlav", "sublead_genPartFlav"])  # Load events
            y = np.load(y_path)
            weights = np.load(w_path)


            if (("TTG_" in sample) or (sample == "TT") or (sample == "TTGG")):
                prompt_photon_bool = (events["lead_genPartFlav"] == 1) & (events["sublead_genPartFlav"] == 1)
                # Select only prompt photons
                events = events[prompt_photon_bool]
                y = y[prompt_photon_bool]
                weights = weights[prompt_photon_bool]

            diphoton_mass = events['mass']
            dijet_mass = events[dijet_mass_key]
            nBLoose = events['nBLoose']
            nBMedium = events['nBMedium']

            # Check if files exist
            if not (os.path.exists(y_path) and os.path.exists(w_path)):
                print(f"Missing y or weights for {path}. Skipping.")
                continue



            label = 1 if "GluGlutoHHto2B2G_kl" in sample else 0  # Signal = 1, Background = 0
            samples_input["score"].append(y)
            samples_input["weights"].append(weights)
            samples_input["labels"].append(np.full(y.shape[0], label))
            samples_input["diphoton_mass"].append(np.array(diphoton_mass))
            samples_input["dijet_mass"].append(np.array(dijet_mass))
            samples_input["sample"].append(np.full(y.shape[0], sample))
            samples_input["nBLoose"].append(np.array(nBLoose))
            samples_input["nBMedium"].append(np.array(nBMedium))

    # Concatenate all data
    samples_input["weights"] = np.concatenate(samples_input["weights"], axis=0)
    samples_input["labels"] = np.concatenate(samples_input["labels"], axis=0)
    samples_input["diphoton_mass"] = np.concatenate(samples_input["diphoton_mass"], axis=0)
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["nBLoose"] = np.concatenate(samples_input["nBLoose"], axis=0)
    samples_input["nBMedium"] = np.concatenate(samples_input["nBMedium"], axis=0)

    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["arg_max_score"] = np.argmax(scores, axis=1)
    samples_input["score"] = [row for row in scores]
    
    # concovert to pandas dataframe
    samples_input = pd.DataFrame(samples_input)

    # apply selection
    samples_input = selection(samples_input, apply_selection)

    # apply preselection
    #samples_input = samples_input[(samples_input["diphoton_mass"] > 100) & (samples_input["diphoton_mass"] < 180)]
    #samples_input = samples_input[(samples_input["dijet_mass"] > 70) & (samples_input["dijet_mass"] < 190)]

    # print sum of total background and signal weights
    print(f"Total background weight: {samples_input['weights'][samples_input['labels'] == 0].sum()}")
    print(f"Total signal weight: {samples_input['weights'][samples_input['labels'] == 1].sum()}")

    # print sum of total background and signal weights under the peak
    print(f"Total background weight under the peak: {samples_input['weights'][(samples_input['labels'] == 0) & (samples_input['diphoton_mass'] > 120) & (samples_input['diphoton_mass'] <130)].sum()}")
    print(f"Total signal weight under the peak: {samples_input['weights'][(samples_input['labels'] == 1) & (samples_input['diphoton_mass'] > 120) & (samples_input['diphoton_mass'] <130)].sum()}")


    return samples_input

def plot_stacked_histogram(sim_folder, data_folder, sim_samples, variables, out_path, bins=40, mass_window=(120, 130), signal_scale=100, include_2023=True, mask=True):
    """
    Load data first, then loop over variables to plot stacked histograms with MC and Data, including ratio plots.

    Parameters:
        sim_folder (str): Path to the directory containing MC samples.
        data_folder (str): Path to the directory containing data samples.
        sim_samples (list): List of MC sample names.
        variables (list): List of variables to plot.
        bins (int or array-like): Number of bins or bin edges.
        mass_window (tuple): Mass range to blind (default: (120, 130) GeV).
        signal_scale (int): Scale factor for signal visualization.
    """
    # create output directory if it does not exist
    out_path = os.path.join(out_path, "Data_MC_Plots_nonRes_first")
    os.makedirs(out_path, exist_ok=True)
    mc_colors = [
    "#FF8A50",  # Darker Peach
    "#FFB300",  # Golden Yellow
    "#66BB6A",  # Rich Green
    "#42A5F5",  # Deeper Sky Blue
    "#AB47BC",  # Strong Lavender Purple
    "#EC407A",  # Deeper Pink
    "#C0CA33",  # Darker Lime
    "#26A69A",  # Deep Teal
    "#1976D2",  # Lighter Blue
    "#EF5350",  # Lighter Red
    "#795548",  # Deep Brown
    "#757575",  # Medium Gray
    "#66BB6A",  # Light Green
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
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "ggHH",
        "VBFHHto2B2G_CV_1_C2V_1_C3_1": "VBFHH",
        "DDQCDGJET": "DDQCDGJets",
        "TTG_10_100": "TTG_10_100",
        "TTG_100_200": "TTG_100_200",
        "TTG_200": "TTG_200",
        "TT": "TT",
    }

    # Load MC and Signal Data First
    stack_mc_dict = {}
    signal_mc_dict = {}

    class_names = ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score", "VBFToHH_sig_score"]

    for sample in sim_samples:
        # Load MC events
        sample_preEE = ak.from_parquet(f"{sim_folder}/preEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
        sample_postEE = ak.from_parquet(f"{sim_folder}/postEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
        if include_2023:
            sample_preBPix = ak.from_parquet(f"{sim_folder}/preBPix/{sample}/events.parquet", columns=variables + ["weight_tot"])
            sample_postBPix = ak.from_parquet(f"{sim_folder}/postBPix/{sample}/events.parquet", columns=variables + ["weight_tot"])

        if os.path.exists(f"{sim_folder}/preEE/{sample}/y.npy"):
            score_preEE = np.load(f"{sim_folder}/preEE/{sample}/y.npy")
            score_postEE = np.load(f"{sim_folder}/postEE/{sample}/y.npy")
            if include_2023:
                score_preBPix = np.load(f"{sim_folder}/preBPix/{sample}/y.npy")
                score_postBPix = np.load(f"{sim_folder}/postBPix/{sample}/y.npy")

            num_classes = score_preEE.shape[1]

            for i, class_name in enumerate(class_names):
                if i < num_classes:
                    sample_preEE[class_name] = score_preEE[:, i]
                    sample_postEE[class_name] = score_postEE[:, i]
                    if include_2023:
                        sample_preBPix[class_name] = score_preBPix[:, i]
                        sample_postBPix[class_name] = score_postBPix[:, i]


        # Merge preEE and postEE
        if include_2023:
            sample_combined = ak.concatenate([sample_preEE, sample_postEE, sample_preBPix, sample_postBPix], axis=0)
        else:
            sample_combined = ak.concatenate([sample_preEE, sample_postEE], axis=0)
        if "minMVAID" in variables:
            sample_combined["minMVAID"] = np.min([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)
            sample_combined["maxMVAID"] = np.max([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)

        # Blind mass region
        #if "mass" in sample_combined.fields:
        #    sample_combined = sample_combined[(sample_combined["mass"] < mass_window[0]) | (sample_combined["mass"] > mass_window[1])]

        # Separate signal from background
        if sample in ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]:
            signal_mc_dict[label_dict[sample]] = sample_combined
        else:
            stack_mc_dict[label_dict[sample]] = sample_combined

    # Load Data First
    if include_2023:
        data_samples = ["2022_EraE", "2022_EraF", "2022_EraG", "2022_EraC", "2022_EraD", "2023_EraCv1to3", "2023_EraCv4", "2023_EraD"]
    else:
        data_samples = ["2022_EraE", "2022_EraF", "2022_EraG", "2022_EraC", "2022_EraD"]
    data_combined = None

    for data_sample in data_samples:
        data_part = ak.from_parquet(f"{data_folder}/{data_sample}/events.parquet", columns=variables)
        if os.path.exists(f"{data_folder}/{data_sample}/y.npy"):
            data_score = np.load(f"{data_folder}/{data_sample}/y.npy")
            for i, class_name in enumerate(class_names):
                if i < num_classes:
                    data_part[class_name] = data_score[:, i]

        if data_combined is None:
            data_combined = data_part
        else:
            data_combined = ak.concatenate([data_combined, data_part], axis=0)
    if "minMVAID" in variables:
        data_combined["minMVAID"] = np.min([data_combined.lead_mvaID, data_combined.sublead_mvaID], axis = 0)
        data_combined["maxMVAID"] = np.max([data_combined.lead_mvaID, data_combined.sublead_mvaID], axis = 0)

    # Blind mass region
    data_combined = data_combined[(data_combined["mass"] < mass_window[0]) | (data_combined["mass"] > mass_window[1])]

    # Blind data in mass window
    #if "mass" in data_combined.fields:
    #    data_combined = data_combined[(data_combined["mass"] < mass_window[0]) | (data_combined["mass"] > mass_window[1])]

    variables = variables + class_names

    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 30, "range": (100, 180), "log": True},
        "dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonRes_dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonRes_mjj_regressed": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "Res_mjj_regressed": {"label": r"Resonant $m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "Res_dijet_mass": {"label": r" Resonant $m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "non_resonant_bkg_score": {"label": "non_resonant_bkg_score", "bins": 30, "range": (0, 1), "log": True},
        "ttH_score": {"label": "ttH_score", "bins": 30, "range": (0, 1), "log": True},
        "other_single_H_score": {"label": "other_single_H_score", "bins": 30, "range": (0, 1), "log": True},
        "GluGluToHH_score": {"label": "GluGluToHH_score", "bins": 30, "range": (0, 1), "log": True},
        "VBFToHH_sig_score": {"label": "VBFToHH_sig_score", "bins": 30, "range": (0, 1), "log": True},
        "minMVAID": {"label": "minMVAID", "bins": 30, "range": (-0.7, 1), "log": True},
        "maxMVAID": {"label": "maxMVAID", "bins": 30, "range": (-0.7, 1), "log": True},
        "n_jets": {"label": "n_jets", "bins": 10, "range": (0, 10), "log": False},
        "sublead_eta": {"label": "sublead_eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "lead_eta": {"label": "lead_eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "sublead_pt": {"label": "sublead_pt [GeV]", "bins": 30, "range": (0, 200), "log": True},
        "lead_pt": {"label": "lead_pt [GeV]", "bins": 30, "range": (0, 200), "log": True},
        "pt": {"label": "Diphoton pt [GeV]", "bins": 30, "range": (0, 400), "log": True},
        "eta": {"label": "Diphoton eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "lead_mvaID": {"label": "lead_mvaID", "bins": 30, "range": (-0.7, 1), "log": True},
        "sublead_mvaID": {"label": "sublead_mvaID", "bins": 30, "range": (-0.7, 1), "log": True},
    }

    # Loop Over Variables and Create Plots
    for variable in variables:
        if variable not in data_combined.fields:
            print(f"Variable {variable} not found in data fields. Skipping...")
            continue
        print(f"Processing variable: {variable}")

        # Histogram binning
        print(var_config[variable]["range"])
        bin_edges = np.linspace(*var_config[variable]["range"], var_config[variable]["bins"] + 1)



        # Compute MC histograms with weights
        mc_hist = []
        mc_err = np.zeros(len(bin_edges) - 1)
        mc_labels = []
        mc_colors_used = []

        for i, (sample, data) in enumerate(stack_mc_dict.items()):
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            mc_hist.append(hist)
            mc_labels.append(sample)
            mc_colors_used.append(mc_colors[i % len(mc_colors)])

            # Sum of squared weights for uncertainty calculation
            hist_err, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"]))**2)
            mc_err += hist_err

        mc_total = np.sum(mc_hist, axis=0)
        mc_err = np.sqrt(mc_err)  # Statistical uncertainty

        data_hist, _ = np.histogram(ak.to_numpy((data_combined[variable])), bins=bin_edges)
        data_err = np.sqrt(data_hist)  # Poisson errors

        # Compute total MC and Data integrals
        total_mc_yield = np.sum([np.sum(hist) for hist in mc_hist])
        total_data_yield = np.sum(data_hist)
        yield_text = f"Data Int.: {total_data_yield:.2f}\nMC Int.: {total_mc_yield:.2f}"

        # Compute signal histograms with weights
        signal_histograms = {}
        for signal, data in signal_mc_dict.items():
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            if "ggHH" in signal:
                signal_histograms[signal] = hist * signal_scale
            else:
                signal_histograms[signal] = hist * signal_scale * 10

        # Plot
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, figsize=(10, 10), sharex=True )
        ax, ax_ratio = axs

        # set luminosity, CMS label, and legend
        #hep.cms.label(data=True, lumi=34.65, ax=ax, loc=0, fontsize=16, label="Private Work", com=13.6)
        if include_2023:
            hep.cms.label(data=True, lumi=61.90, ax=ax, loc=0, fontsize=16, label="Private Work", com=13.6)
        else:
            hep.cms.label(data=True, lumi=34.65, ax=ax, loc=0, fontsize=16, label="Private Work", com=13.6)
        

        # Stacked MC histograms
        hep.histplot(
            mc_hist,
            bin_edges,
            histtype="fill",
            stack=True,
            label=mc_labels,
            color=mc_colors_used,
            edgecolor="black",
            ax=ax,
            #alpha=0.7
        )

        # Uncertainty bars for MC sum
        #ax.errorbar(
        #    (bin_edges[:-1] + bin_edges[1:]) / 2,
        #    mc_total,
        #    yerr=mc_err,
        #    fmt="none",
        #    color="black",
        #    label="MC Stat. Unc.",
        #    capsize=3,
        #    linestyle="none",
        #)

        ax.fill_between(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            mc_total - mc_err,
            mc_total + mc_err,
            color="gray",
            alpha=0.5,
            step="mid",
            #label="MC Stat. Unc."
        )

        # Data points
        ax.errorbar(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            data_hist,
            yerr=data_err,
            fmt="o",
            color="black",
            label="Data",
            markersize=5,
        )

        # Plot signal as step histogram
        for signal, hist in signal_histograms.items():
            if "ggHH" in signal:
                print(signal)
                ax.step(
                    (bin_edges[:-1] + bin_edges[1:]) / 2,
                    hist,
                    where="mid",
                    linestyle="dashed",
                    color="red",
                    label=f"{signal} x {signal_scale}"
                )
            else:
                signal_scale_ = signal_scale*10
                ax.step(
                    (bin_edges[:-1] + bin_edges[1:]) / 2,
                    hist,
                    where="mid",
                    linestyle="dashed",
                    color="yellow",
                    label=f"{signal} x {signal_scale_}"
                )

        ax.legend(fontsize=12, ncol=2, title=r"$m_{\gamma\gamma}$ blinded in [120, 130] GeV")

        # Labels
        ax.set_ylabel("Events")
        ax.set_xlim(var_config[variable]["range"])

        if var_config[variable]["log"]:
            ax.set_yscale("log")
            ax.set_ylim(0.001, 1200 * np.max(data_hist))
        else:
            ax.set_ylim(0, 1.7 * np.max(data_hist))

        # Annotate Data and MC integrals on the plot
        #ax.text(
        #    0.3, 0.95, yield_text,
        #    transform=ax.transAxes,
        #    fontsize=13,
        #    verticalalignment='top',
        #    horizontalalignment='right',
        #    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
        #)
        # Ratio plot (Data / MC)
        ratio = abs(data_hist / mc_total)
        data_ratio_err = abs(data_err / mc_total)
        mc_ratio_err = abs(mc_err / mc_total)

        ax_ratio.errorbar(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            ratio,
            yerr=data_ratio_err,
            fmt="o",
            color="black",
            markersize=5,
        )

        ax_ratio.fill_between(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            1 - mc_ratio_err,
            1 + mc_ratio_err,
            color="gray",
            alpha=0.3,  # Keep it slightly transparent
            hatch="xx",  # Adds cross-hatched pattern
            edgecolor="black",  # Ensures visibility of hatch
            linewidth=0.0,  # Removes additional border
            step="mid",
        )

        ax_ratio.axhline(1, linestyle="dashed", color="gray")
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_xlabel(var_config[variable]["label"])
        plt.savefig(f"{out_path}/{variable}.png", dpi=300, bbox_inches="tight")
        plt.clf()
    print("output saved in ", out_path)

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
            f"{th_name_to_calss[k]}={v:.4f}" for k, v in thresholds_dict.items()
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
    return np.sqrt(2.0 * ((s + b + 1e-10) * np.log(1.0 + s / (b + 1e-10)) - s))

def asymptotic_significance(df):
    s = df[df["labels"] == 1].weights.sum()
    b = df[df["labels"] == 0].weights.sum()
    print(f"Signal: {s}, Background: {b}")
    z = np.sqrt(2 * ((s + b + 1e-10) * np.log(1 + (s / (b + 1e-10))) - s))
    #z = np.sqrt(2 * (((s + b + 1e-10) * np.log(1 + s / (b + 1e-10))) - s))
    print(f"Signal: {s}, Background: {b}, Significance: {z}")
    return z

#############################################
# Sequential categorization using Optuna
#############################################

def get_best_cut_params_using_optuna(n_categories, samples_input, out_dir, folder_name, signal_class=3, n_trials=150, side_band_threshold=10):
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
    cat_path = os.path.join(out_dir, f"{folder_name}")
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
            th_signal = trial.suggest_float("th_signal", 0, prev_signal_cut)
            mask = scores[:, signal_class] > th_signal
            # Background thresholds for each background class, search in (prev_bg_cut[b], 1)
            for b in bg_classes:
                th_bg = trial.suggest_float(f"th_bg_{b}", prev_bg_cut[b], 1)
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

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        #study = optuna.create_study(direction="maximize")
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

    filtered_df = samples_df[~remove_mask].copy()
    return filtered_df, remove_mask

def get_best_cut_params_for_cr(
    samples_input,
    out_dir,
    folder_name,
    cr_classes=[1, 2],
    cr_class_names=[["ttHtoGG_M_125"], ["BBHto2G_M_125", "GluGluHToGG_M_125"]],
    cr_name = ['ttH'],
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

    cr_path = os.path.join(out_dir, f"{folder_name}")
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
        samples = samples_remaining["sample"].values

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
            #side_mask = ((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130)) & (labels[mask] != cr_class_idx)
            side_mask = (((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130)) &
             (~np.isin(samples[mask], cr_class_names[i_cr])))
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
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        #study = optuna.create_study(direction="maximize")
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


def store_categorization_events_with_score(
    base_path,
    best_cut_values,
    folder_name,
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

    nsr = 3

    out_dir = f"{base_path}/{folder_name}/"
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
        "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
        "TTG_10_100",
        "TTG_100_200",
        "TTG_200",
        "TT"
    ]
    dijet_mass_key = "Res_mjj_regressed"
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
            #nBMedium_cut = (events["nBMedium"] >= 2)
            #events = events[nBMedium_cut]
            #scores = scores[nBMedium_cut]

            return events, scores

    for era in ["preEE", "postEE", "preBPix", "postBPix"]:
        for sample in samples:
            if not os.path.exists(f"{base_path}/individual_samples/{era}/{sample}"):
                print(f"Skipping {sample} in {era} as it does not exist.")
                continue
            inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
            print(f"Processing {inputs_path}")

            # Load events
            events = ak.from_parquet(inputs_path)
            scores = np.load(f"{inputs_path}/y.npy")

            # Load weights
            rel_w = np.load(f"{inputs_path}/rel_w.npy")

            if (("TTG_" in sample) or (sample == "TT") or (sample == "TTGG")):
                print("selecting prompt photons for TTG and TT samples")
                prompt_photon_bool = ((events.lead_genPartFlav == 1) | (events.sublead_genPartFlav == 1))
                events = events[prompt_photon_bool]
                rel_w = rel_w[prompt_photon_bool]
                scores = scores[prompt_photon_bool]
            
            events["weight_tot"] = rel_w
            events["dijet_mass"] = events[dijet_mass_key]

            # Apply selection if needed
            events, scores = selection(events, scores, apply_selection)

            # Keep track of leftover
            selected_events = events
            selected_scores = scores

            # ============= SR Categories (cat1, cat2, cat3) =============
            for i in range(nsr):
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
        for i in range(nsr):
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
    all_categories_ = set(cat for (_, cat) in yields_info.keys())
    # Sort them (e.g. cat1, cat2, CR_...) if you prefer alphabetical order
    all_categories_ = sorted(list(all_categories_))

    # Group by category
    with open(out_txt, "w") as f:
        for cat in all_categories_:
            f.write(f"{cat}:\n")
            # Loop over all (sample, category) pairs
            for (sample, this_cat), (sum_w, sum_w_sq) in yields_info.items():
                if this_cat == cat:
                    unc = math.sqrt(sum_w_sq)
                    f.write(f"{sample}, {sum_w}, {unc}\n")
            f.write("\n")  # Blank line after each category

    all_samples = sorted(set(sample for (sample, _) in yields_info.keys()))
    all_categories = sorted(set(cat for (_, cat) in yields_info.keys()))

    # Initialize DataFrame
    data = []

    for sample in all_samples:
        row = {"Sample": sample}
        for cat in all_categories:
            sum_w, sum_w_sq = yields_info.get((sample, cat), (0.0, 0.0))
            if sum_w == 0.0 and sum_w_sq == 0.0:
                row[cat] = ""
            else:
                unc = math.sqrt(sum_w_sq)
                row[cat] = f"{sum_w:.3f} ± {unc:.3f}"
        data.append(row)

    df = pd.DataFrame(data)
    df = df[["Sample"] + all_categories]  # Ensure column order

    # Save to txt that looks like a table when pasted into Keynote
    out_txt = os.path.join(out_dir, "categorization_yields.txt")
    df.to_csv(out_txt, sep="\t", index=False)

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
                "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
                "TTG_10_100", "TTG_100_200", "TTG_200", "TT"
                ]
    # combine preEE and postEE and convert to root
    for sample in samples:
        preEE_events = ak.from_parquet(f"{base_path}/preEE/{sample}/events.parquet")
        postEE_events = ak.from_parquet(f"{base_path}/postEE/{sample}/events.parquet")
        preBPix_events = ak.from_parquet(f"{base_path}/preBPix/{sample}/events.parquet")
        postBPix_events = ak.from_parquet(f"{base_path}/postBPix/{sample}/events.parquet")
        events = ak.concatenate([preEE_events, postEE_events, preBPix_events, postBPix_events], axis=0)
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
        "2022_EraE",
        "2022_EraF",
        "2022_EraG",
        "2022_EraC",
        "2022_EraD",
        "2023_EraCv1to3",
        "2023_EraCv4",
        "2023_EraD"
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

def save_table_as_pdf(df, output_pdf="event_yields.pdf", title="Event Yields Summary"):
    fig, ax = plt.subplots(figsize=(1.5 * len(df.columns), 0.5 * len(df) + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

def collect_event_yields(base_dir, category_list):
    import numpy as np
    import awkward as ak
    import pandas as pd
    import matplotlib.pyplot as plt
    
    samples = [
        "GGJets", "DDQCDGJET", "TTGG", "ttHtoGG_M_125", "BBHto2G_M_125",
        "GluGluHToGG_M_125", "VBFHToGG_M_125", "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        "TTG_10_100", "TTG_100_200", "TTG_200", "TT"
    ]
    
    data_samples = [
        "2022_EraE", "2022_EraF", "2022_EraG", "2022_EraC", "2022_EraD",
        "2023_EraCv1to3", "2023_EraCv4", "2023_EraD"
    ]

    folder_to_region_dict = {
        "cat1": "SR1",
        "cat2": "SR2",
        "cat3": "SR3",
        "ttH": "CR_ttH",
        # "bbH": "CR_bbH",
    }
    
    yield_dict = {}

    for category in category_list:
        total_mc = 0.0
        total_mc_err2 = 0.0

        for sample in samples:
            if not os.path.exists(f"{base_dir}/{category}"):
                print(f"Skipping {category} as it does not exist.")
                continue
            events = ak.concatenate([
                ak.from_parquet(f"{base_dir}/{category}/preEE/{sample}/events.parquet", columns=["weight_tot"]),
                ak.from_parquet(f"{base_dir}/{category}/postEE/{sample}/events.parquet", columns=["weight_tot"]),
                ak.from_parquet(f"{base_dir}/{category}/preBPix/{sample}/events.parquet", columns=["weight_tot"]),
                ak.from_parquet(f"{base_dir}/{category}/postBPix/{sample}/events.parquet", columns=["weight_tot"])
            ], axis=0)
            weight_sum = np.sum(events["weight_tot"])
            weight_err = np.sqrt(np.sum(events["weight_tot"]**2))
            if sample not in yield_dict:
                yield_dict[sample] = {}
            yield_dict[sample][folder_to_region_dict[category]] = f"{weight_sum:.2f} ± {weight_err:.2f}"
            total_mc += weight_sum
            total_mc_err2 += weight_err**2

        if "MC total" not in yield_dict:
            yield_dict["MC total"] = {}
        yield_dict["MC total"][folder_to_region_dict[category]] = f"{total_mc:.2f} ± {np.sqrt(total_mc_err2):.2f}"
        
        total_data = 0.0
        for data in data_samples:
            if not os.path.exists(f"{base_dir}/{category}/{data}"):
                print(f"Skipping {category}/{data} as it does not exist.")
                continue
            events = ak.from_parquet(f"{base_dir}/{category}/{data}/events.parquet", columns=["weight_tot"])
            total_data += len(events)

        total_data_err = np.sqrt(total_data)
        if "Data total" not in yield_dict:
            yield_dict["Data total"] = {}
        yield_dict["Data total"][folder_to_region_dict[category]] = f"{total_data:.2f} ± {total_data_err:.2f}"

    print(yield_dict)
    # Convert to DataFrame and export
    df = pd.DataFrame(yield_dict).fillna("N/A")
    df = df.T
    print(df)
    #df = df[category_list]  # Keep column order consistent
    output_pdf = f"{base_dir}/event_yields.pdf"

    fixed_first = 0.42
    # total number of table columns = 1 index + N data columns
    n_cols = len(df.columns) + 1
    # split the remaining width equally among the others
    rest = (1 - fixed_first) / (n_cols - 1)
    col_widths = [fixed_first] + [rest] * (n_cols - 1)

    # 2) Make the figure width scale with number of categories:
    #    e.g. give yourself 1.2 inches per category
    width_per_col = 2
    fig_width = width_per_col * n_cols
    fig_height = 0.4 * len(df) + 1.5   # tweak this so text doesn’t run together

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        tbl = ax.table(
            cellText=df.reset_index().values,
            colLabels=[''] + df.columns.tolist(),
            colWidths=col_widths,
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)

        # 3) Highlight the last two DATA rows
        #    Header is at row 0, data rows go 1..len(df)
        last = len(df)
        second_last = last - 1
        for row in (second_last, last):
            for col in range(n_cols):
                tbl[(row, col)].set_facecolor('#fffbcc')  # pale yellow, say

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    #save_table_as_pdf(df, output_pdf)
        
def collect_event_yields(base_dir, category_list, mass_range=None):
    import os
    import numpy as np
    import awkward as ak
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    samples = [
        "GGJets", "DDQCDGJET", "TTGG", "ttHtoGG_M_125", "BBHto2G_M_125",
        "GluGluHToGG_M_125", "VBFHToGG_M_125", "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        "TTG_10_100", "TTG_100_200", "TTG_200", "TT"
    ]
    data_samples = [
        "2022_EraE","2022_EraF","2022_EraG","2022_EraC","2022_EraD",
        "2023_EraCv1to3","2023_EraCv4","2023_EraD"
    ]
    folder_to_region = {
        "cat1":"SR1","cat2":"SR2","cat3":"SR3",
        "ttH":"CR_ttH",
        # "bbH":"CR_bbH",
    }
    yield_dict = {}

    # unpack mass range if given
    if mass_range is not None:
        m_low, m_high = mass_range

    for category in category_list:
        # ----- Monte Carlo -----
        total_mc, total_mc_err2 = 0.0, 0.0
        for sample in samples:
            path = f"{base_dir}/{category}"
            if not os.path.isdir(path):
                print(f"Skipping {category} as it does not exist.")
                continue

            # load both weight and mass
            ev = ak.concatenate([
                ak.from_parquet(f"{path}/preEE/{sample}/events.parquet",
                                columns=["weight_tot","mass"]),
                ak.from_parquet(f"{path}/postEE/{sample}/events.parquet",
                                columns=["weight_tot","mass"]),
                ak.from_parquet(f"{path}/preBPix/{sample}/events.parquet",
                                columns=["weight_tot","mass"]),
                ak.from_parquet(f"{path}/postBPix/{sample}/events.parquet",
                                columns=["weight_tot","mass"])
            ], axis=0)

            # apply the mass cut if requested
            if mass_range is not None:
                mask = (ev.mass >= m_low) & (ev.mass <= m_high)
                ev = ev[mask]

            wsum = np.sum(ev.weight_tot)
            werr = np.sqrt(np.sum(ev.weight_tot**2))

            yield_dict.setdefault(sample, {})[
                folder_to_region[category]
            ] = f"{wsum:.2f} ± {werr:.2f}"

            total_mc     += wsum
            total_mc_err2 += werr**2

        yield_dict.setdefault("MC total", {})[
            folder_to_region[category]
        ] = f"{total_mc:.2f} ± {np.sqrt(total_mc_err2):.2f}"

        # ----- Data -----
        total_data = 0
        for data in data_samples:
            dpath = f"{base_dir}/{category}/{data}"
            if not os.path.isdir(dpath):
                print(f"Skipping {category}/{data} as it does not exist.")
                continue

            ev = ak.from_parquet(f"{dpath}/events.parquet",
                                 columns=["weight_tot","mass"])
            if mass_range is not None:
                mask = (ev.mass >= m_low) & (ev.mass <= m_high)
                ev = ev[mask]
            total_data += len(ev)

        data_err = np.sqrt(total_data)
        yield_dict.setdefault("Data total", {})[
            folder_to_region[category]
        ] = f"{total_data:.2f} ± {data_err:.2f}"

    # build and transpose DF
    df = pd.DataFrame(yield_dict).T.fillna("N/A")

    # choose suffix
    suffix = f"_{m_low:.0f}_{m_high:.0f}" if mass_range else ""
    output_pdf = f"{base_dir}/event_yields{suffix}.pdf"

    # same plotting code as before
    fixed_first = 0.42
    n_cols = len(df.columns) + 1
    rest = (1-fixed_first)/(n_cols-1)
    col_widths = [fixed_first] + [rest]*(n_cols-1)
    width_per_col = 2
    fig_w = width_per_col * n_cols
    fig_h = 0.4*len(df) + 1.5

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText   = df.reset_index().values,
            colLabels  = [""] + df.columns.tolist(),
            colWidths  = col_widths,
            loc        = "center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)

        # highlight last two rows
        last = len(df); second_last = last-1
        for r in (second_last, last):
            for c in range(n_cols):
                tbl[(r,c)].set_facecolor("#fffbcc")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


#############################################
# Main execution
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize multiclass scores using Optuna with dynamic search ranges, sideband requirements, and summary plots.")
    parser.add_argument("--n_categories", type=int, default=6, help="Number of categories to optimize")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the input samples")
    args = parser.parse_args()

    optuna_folder = "optuna_categorization"

    #samples_list = [
    #    "GGJets",
    #    "DDQCDGJET",
    #    "TTGG",
    #    "ttHtoGG_M_125",
    #    "BBHto2G_M_125",
    #    "GluGluHToGG_M_125",
    #    "VBFHToGG_M_125",
    #    "VHtoGG_M_125",
    #    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    #    "TTG_10_100", "TTG_100_200", "TTG_200", "TT"
    #    #"VBFHHto2B2G_CV_1_C2V_1_C3_1"
    #]
    samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT"]
    
    samples_input = load_samples(args.base_path, samples_list)

    best_params, best_sig_values = get_best_cut_params_using_optuna(
        n_categories=args.n_categories,
        samples_input=samples_input,
        folder_name=optuna_folder,
        out_dir=args.base_path,
        signal_class=3,  # adjust if necessary
        n_trials=150,
        side_band_threshold=10.0,
    )

    # load the best cut values
    with open(f"{args.base_path}/{optuna_folder}/best_cut_params.json", "r") as f:
        best_cut_values = json.load(f)

    chosen_k = 3  # e.g. you discovered that 2 SR categories saturates your significance
    best_cats_for_sr = best_cut_values[:chosen_k]

    samples_for_cr, sr_mask = remove_selected_events(
    samples_df=samples_input,
    best_cut_params_list=best_cats_for_sr,
    signal_class=3,
    )


    #cr_classes = [1, 2]  # or whichever you desire
    #cr_class_names = [
    #    #["BBHto2G_M_125", "GluGluHToGG_M_125"]
    #    ["ttHtoGG_M_125"],
    #    ["BBHto2G_M_125"]
    #]
    #cr_name = ["ttH", 'bbH']

    cr_classes = [1]  # or whichever you desire
    cr_class_names = [
        #["BBHto2G_M_125", "GluGluHToGG_M_125"]
        ["ttHtoGG_M_125"],
    ]
    cr_name = ["ttH"]

    best_cut_params_cr, best_sig_cr = get_best_cut_params_for_cr(
        samples_input=samples_for_cr,
        out_dir=args.base_path,
        folder_name=optuna_folder,
        cr_classes=cr_classes,
        cr_class_names=cr_class_names,
        cr_name=cr_name,
        n_trials=150,
        sideband_threshold=5.0,
    )

    # load the best cut values for CR
    with open(f"{args.base_path}/{optuna_folder}/best_cr_cut_params.json", "r") as f:
        best_cr_cut_values = json.load(f)

    store_categorization_events_with_score(
    args.base_path,
    best_cut_values,
    folder_name=optuna_folder,
    best_cut_params_cr=best_cr_cut_values,
    cr_classes=cr_classes,
    cr_class_names=cr_class_names,
    cr_name=cr_name
    )

    folder_list = ["cat1", "cat2", "cat3", "ttH"]#, "ggH", "bbH"]
    #folder_list = ["cat1", "cat2", "ttH", "bbH"]
    #folder_list = ["ttH", "ggH_bbH"]

    for folder in folder_list:
        opt_cat_folder_name = f"{optuna_folder}"
        if not os.path.exists(f"{args.base_path}/{opt_cat_folder_name}/{folder}"):
            print(f"Folder {args.base_path}/{opt_cat_folder_name}/{folder} does not exist.")
            continue
        sim_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        data_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        out_path = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        variables = ["mass", "dijet_mass", "nonRes_dijet_mass", "nonRes_mjj_regressed", "Res_mjj_regressed", "Res_dijet_mass"]
        
        plot_stacked_histogram(sim_folder, data_folder, samples_list, variables, out_path, signal_scale=100)

    collect_event_yields(f"{args.base_path}/{opt_cat_folder_name}", folder_list[:-1])
    
    for folder in folder_list:
        opt_cat_folder_name = f"{optuna_folder}"
        if not os.path.exists(f"{args.base_path}/{opt_cat_folder_name}/{folder}"):
            print(f"Folder {args.base_path}/{opt_cat_folder_name}/{folder} does not exist.")
            continue
        sim_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        data_folder = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        out_path = f"{args.base_path}/{opt_cat_folder_name}/{folder}"
        variables = ["mass", "dijet_mass", "nonRes_dijet_mass", "nonRes_mjj_regressed", "Res_mjj_regressed", "Res_dijet_mass"]
        convert_to_root(out_path)

    
    #collect_event_yields(f"{args.base_path}/{opt_cat_folder_name}", folder_list[:-1], mass_range=(120, 130))