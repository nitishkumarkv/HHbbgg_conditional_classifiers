import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import yaml

# Apply CMS style
hep.style.use("CMS")

def plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, bins=40, mass_window=(120, 130), signal_scale=100, only_MC=False):
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
    if only_MC:
        out_path = os.path.join(out_path, "MC_Plots")
    else:
        out_path = os.path.join(out_path, "Data_MC_Plots")
    os.makedirs(out_path, exist_ok=True)
    mc_colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "gold", "brown", "pink", "lime"]
    mc_colors = [
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
        "DDQCDGJET": "DDQCDGJets"
    }

    # Load MC and Signal Data First
    stack_mc_dict = {}
    signal_mc_dict = {}

    def add_preselection(events):
        # Apply preselection criteria
        dijet_bool = (events["nonRes_mjj_regressed"] > 70) & (events["nonRes_mjj_regressed"] < 190)
        
        #events = events[dijet_bool]
        return events

    class_names = ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score", "VBFToHH_sig_score"]
    events_path = samples_info["samples_path"]
    eras = samples_info["eras"]
    for sample in sim_samples:
        sample_combined = []
        for era in eras:
            events_ = ak.from_parquet(f"{events_path}/{samples_info[era][sample]}", columns=variables)
            scores_ = np.load(f"{sim_folder}/{era}/{sample}/y.npy")
            rel_w_ = np.load(f"{sim_folder}/{era}/{sample}/rel_w.npy")
            events_["weight_tot"] = rel_w_
            for i, class_name in enumerate(class_names):
                if i < scores_.shape[1]:
                    num_classes = scores_.shape[1]
                    events_[class_name] = scores_[:, i]
            sample_combined.append(events_)
        sample_combined = ak.concatenate(sample_combined, axis=0)

        if "minMVAID" in variables:
            sample_combined["minMVAID"] = np.min([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)
            sample_combined["maxMVAID"] = np.max([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)

        # Apply preselection
        sample_combined = add_preselection(sample_combined)
        if sample == "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
            print("number of events in GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", sum(sample_combined["weight_tot"]))


        # Blind mass region
        #if "mass" in sample_combined.fields:
        #    sample_combined = sample_combined[(sample_combined["mass"] < mass_window[0]) | (sample_combined["mass"] > mass_window[1])]

        # Separate signal from background
        if sample in ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]:
            signal_mc_dict[label_dict[sample]] = sample_combined
        else:
            stack_mc_dict[label_dict[sample]] = sample_combined

    # Load Data First
    data_combined = None

    data_samples = training_config["samples_info"]["data"]

    for data_sample, path in data_samples.items():
        data_part = ak.from_parquet(f"{events_path}/{path}", columns=variables)
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

    # Apply preselection
    data_combined = add_preselection(data_combined)

    # Blind data in mass window
    #if "mass" in data_combined.fields:
    #    data_combined = data_combined[(data_combined["mass"] < mass_window[0]) | (data_combined["mass"] > mass_window[1])]

    variables = variables + class_names

    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 40, "range": (100, 180), "log": True},
        "nonRes_dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
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

        # Compute data histogram
        if variable == "mass":
            to_plot = data_combined[(data_combined["mass"] < mass_window[0]) | (data_combined["mass"] > mass_window[1])]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        elif variable == "nonRes_dijet_mass":
            to_plot = data_combined[(data_combined["nonRes_dijet_mass"] < 110) | (data_combined["nonRes_dijet_mass"] > 140)]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        elif variable == "dijet_mass":
            to_plot = data_combined[(data_combined["dijet_mass"] < 110) | (data_combined["dijet_mass"] > 140)]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        elif variable == "nonRes_mjj_regressed":
            to_plot = data_combined[(data_combined["nonRes_mjj_regressed"] < 110) | (data_combined["nonRes_mjj_regressed"] > 140)]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        elif variable == "Res_mjj_regressed":
            to_plot = data_combined[(data_combined["Res_mjj_regressed"] < 110) | (data_combined["Res_mjj_regressed"] > 140)]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        elif variable == "Res_dijet_mass":
            to_plot = data_combined[(data_combined["Res_dijet_mass"] < 110) | (data_combined["Res_dijet_mass"] > 140)]
            data_hist, _ = np.histogram(ak.to_numpy((to_plot[variable])), bins=bin_edges)
        else: 
            data_hist, _ = np.histogram(ak.to_numpy((data_combined[variable])), bins=bin_edges)
        data_err = np.sqrt(data_hist)  # Poisson errors

        # Compute signal histograms with weights
        signal_histograms = {}
        for signal, data in signal_mc_dict.items():
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            if "ggHH" in signal:
                signal_histograms[signal] = hist * signal_scale
            else:
                signal_histograms[signal] = hist * signal_scale * 10

        # Plot
        if only_MC:
            fig, ax = plt.subplots(figsize=(10, 10))
        else: 
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, figsize=(10, 10), sharex=True )
            ax, ax_ratio = axs

        luminosities = {
        "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
        "postEE": 26.67,  # Integrated luminosity for postEE in fb^-1
        "preBPix": 17.794,  # Integrated luminosity for preEE in fb^-1
        "postBPix": 9.451  # Integrated luminosity for postEE in fb^-1
        }

        lumi = 0
        for era in eras:
            if era in luminosities:
                lumi += luminosities[era]

        # set luminosity, CMS label, and legend
        hep.cms.label(data=True, lumi=lumi, ax=ax, loc=0, fontsize=16, label="Private Work", com=13.6)
        

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
        if not only_MC:
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

        ax.legend(fontsize=16, ncol=2)

        # Labels
        ax.set_ylabel("Events")
        ax.set_xlim(var_config[variable]["range"])

        if var_config[variable]["log"]:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 200 * np.max(data_hist))
        else:
            ax.set_ylim(0, 1.5 * np.max(data_hist))
        # Ratio plot (Data / MC)
        if not only_MC:

            ratio = data_hist / mc_total
            data_ratio_err = data_err / mc_total
            mc_ratio_err = mc_err / mc_total

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
        else:
            ax.set_xlabel(var_config[variable]["label"])
            plt.savefig(f"{out_path}/{variable}.png", dpi=300, bbox_inches="tight")
            plt.clf()
        
        
    print("output saved in ", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot stacked histograms for MC and Data.")
    parser.add_argument("--base-path", type=str, required=True, help="Path to the base directory containing MC and Data folders.")
    parser.add_argument("--training_config_path", type=str, required=True, help="Path to the training config file.")
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)
    
    samples_info = training_config["samples_info"]

    base_path = args.base_path
    sim_folder = f"{base_path}/individual_samples"
    data_folder = f"{base_path}/individual_samples_data"

    # sim_samples = ["GGJets", "DDQCDGJET", "TTGG", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "VBFHToGG_M_125", "VHtoGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]
    # sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1", "TTGG", "GGJets", "DDQCDGJET"]
    sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET"]
    variables = ["Res_mjj_regressed", "Res_dijet_mass", "nonRes_mjj_regressed", "mass", "nonRes_dijet_mass", "minMVAID", "maxMVAID", "n_jets", "sublead_eta", "lead_eta", "sublead_pt", "lead_pt", "pt", "eta", "lead_mvaID", "sublead_mvaID"]
    out_path = f"{base_path}/"

    plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000)
    plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000, only_MC=True)
