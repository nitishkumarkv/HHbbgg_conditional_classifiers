import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import yaml

# Apply CMS style
hep.style.use("CMS")


def plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, bins=40, mass_window=(120, 130), mjj_mass_window=(110, 140), signal_scale=100, only_MC=False):
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
        out_path = os.path.join(out_path, "MC_Plots_70_190")
    else:
        out_path = os.path.join(out_path, "Data_MC_Plots_70_190")
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
    "blue",  # Deep Brown
    "red",  # Vibrant Orange
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

    def deltaR(eta1, phi1, eta2, phi2, fill_none=True):
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

    def add_var(events):
        
        events["nonRes_lead_bjet_pt_over_M_regressed"] = events.nonRes_lead_bjet_pt / events.nonRes_mjj_regressed
        events["nonRes_sublead_bjet_ptover_M_regressed"] = events.nonRes_sublead_bjet_pt / events.nonRes_mjj_regressed

        #events["nonRes_lead_bjet_ptPNetCorr_over_M_regressed"] = events.nonRes_lead_bjet_ptPNetCorr / events.nonRes_mjj_regressed
        #events["nonRes_sublead_bjet_ptPNetCorr_over_M_regressed"] = events.nonRes_sublead_bjet_ptPNetCorr / events.nonRes_mjj_regressed

        events["nonRes_diphoton_PtOverM_ggjj"] = events.pt / events.nonRes_HHbbggCandidate_mass
        events["nonRes_dijet_PtOverM_ggjj"] = events.nonRes_dijet_pt / events.nonRes_HHbbggCandidate_mass
        #events["nonRes_dijet_PtPNetCorrOverM_ggjj"] = events.nonRes_dijet_ptPNetCorr / events.nonRes_HHbbggCandidate_mass
        

        events["Res_lead_bjet_pt_over_M_regressed"] = events.Res_lead_bjet_pt / events.Res_mjj_regressed
        events["Res_sublead_bjet_pt_over_M_regressed"] = events.Res_sublead_bjet_pt / events.Res_mjj_regressed

        #events["Res_lead_bjet_ptPNetCorr_over_M_regressed"] = events.Res_lead_bjet_ptPNetCorr / events.Res_mjj_regressed
        #events["Res_sublead_bjet_ptPNetCorr_over_M_regressed"] = events.Res_sublead_bjet_ptPNetCorr / events.Res_mjj_regressed

        events["Res_diphoton_PtOverM_ggjj"] = events.pt / events.Res_HHbbggCandidate_mass
        events["Res_dijet_PtOverM_ggjj"] = events.Res_dijet_pt / events.Res_HHbbggCandidate_mass

        # add deltaR between lead and sublead photon
        #events["deltaR_gg"] = np.sqrt((events.lead_eta - events.sublead_eta) ** 2 + (events.lead_phi - events.sublead_phi) ** 2)
        events["deltaR_gg"] = deltaR(events.lead_eta, events.lead_phi, events.sublead_eta, events.sublead_phi)
        # add deltaR between lead and sublead bje

        return events

    def add_preselection(events):
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        #dijet_mass_bool = ((events.Res_mjj_regressed > 80) & (events.Res_mjj_regressed < 180))
        dijet_mass_bool = ((events.nonResReg_dijet_mass_DNNreg > 70) & (events.nonResReg_dijet_mass_DNNreg < 190))
        
        #lead_mvaID_bool = ((events.lead_mvaID > 0.0439603) & (events.lead_isScEtaEB == True)) | ((events.lead_mvaID > -0.249526) & (events.lead_isScEtaEE == True))
        #sublead_mvaID_bool = ((events.sublead_mvaID > 0.0439603) & (events.sublead_isScEtaEB == True)) | ((events.sublead_mvaID > -0.249526) & (events.sublead_isScEtaEE == True))

        # add lead and sublead mvaID cut at -0.7
        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]
        #events = events[mass_bool & dijet_mass_bool]

        events = add_var(events)

        return events

    class_names = ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score", "VBFToHH_sig_score"]
    events_path = samples_info["samples_path"]

    # Load Data First
    data_combined = None

    data_samples = training_config["samples_info"]["data"]

    for data_sample, path in data_samples.items():
        if os.path.exists(f"{data_folder}/{data_sample}/events.parquet"):
            print(f"Loading data from {data_folder}/{data_sample}/events.parquet")
            data_part = ak.from_parquet(f"{data_folder}/{data_sample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"])
        else:
            data_part = ak.from_parquet(f"{events_path}/{path}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"])
        if os.path.exists(f"{data_folder}/{data_sample}/y.npy"):
            data_score = np.load(f"{data_folder}/{data_sample}/y.npy")
            num_classes = data_score.shape[1]
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
    print("before: ", len(data_combined))
    
    # get number of data events in sideband
    int_data_sideband = len(data_combined)

    eras = samples_info["eras"]
    for sample in sim_samples:
        sample_combined = []
        for era in eras:
            if os.path.exists(f"{sim_folder}/{era}/{sample}/events.parquet"):
                events_ = ak.from_parquet(f"{sim_folder}/{era}/{sample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav"])
            else:
                events_ = ak.from_parquet(f"{events_path}/{samples_info[era][sample]}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav"])

        
            scores_ = np.load(f"{sim_folder}/{era}/{sample}/y.npy")
            rel_w_ = np.load(f"{sim_folder}/{era}/{sample}/rel_w.npy")
            # select prompt photons for TTG and TT samples
            if (("TTG_" in sample) or (sample == "TT")):
                print("selecting prompt photons for TTG and TT samples")
                prompt_photon_bool = ((events_.lead_genPartFlav == 1) | (events_.sublead_genPartFlav == 1))
                events_ = events_[prompt_photon_bool]
                rel_w_ = rel_w_[prompt_photon_bool]
                scores_ = scores_[prompt_photon_bool]

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


        # Separate signal from background
        if sample in ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]:
            signal_mc_dict[label_dict[sample]] = sample_combined
        else:
            stack_mc_dict[label_dict[sample]] = sample_combined


    variables = class_names + variables

    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 40, "range": (100, 180), "log": True},
        "nonRes_dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonRes_mjj_regressed": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_DNNpair_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_DNNpair_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
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
        #print(f"Processing variable: {variable}")

        if variable not in var_config.keys():
            #get the min and max of the variable
            min_ = 0
            max_ = 0
            for i, (sample, data) in enumerate(stack_mc_dict.items()):
                ak_min = ak.min(data[variable])
                if ak_min != -999:
                    min_ = min(min_, ak.min(data[variable]))
                max_ = max(max_, ak.max(data[variable]))
            
            # check if a list of values is in the variable
            keywords = ["pt", "mass", "btag"]
            if any(key in variable for key in keywords):
            #if "pt" in variable:
                var_config[variable] = {"label": variable, "bins": 30, "range": (min_, max_), "log": True}
            else:
                var_config[variable] = {"label": variable, "bins": 30, "range": (min_, max_), "log": False}

        # Histogram binning
        bin_edges = np.linspace(*var_config[variable]["range"], var_config[variable]["bins"] + 1)



        # Compute MC histograms with weights
        mc_hist = []
        mc_err = np.zeros(len(bin_edges) - 1)
        mc_labels = []
        mc_colors_used = []

        for i, (sample, data) in enumerate(stack_mc_dict.items()):
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            print(variable, "sample: ", sample, "hist sum : ", sum(hist))
            mc_hist.append(hist)
            mc_labels.append(sample)
            mc_colors_used.append(mc_colors[i % len(mc_colors)])

            # Sum of squared weights for uncertainty calculation
            hist_err, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"]))**2)
            mc_err += hist_err

        # compute counts for

        mc_total = np.sum(mc_hist, axis=0)
        mc_err = np.sqrt(mc_err)  # Statistical uncertainty

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
            ax.set_ylim(0.1, 400 * np.max(data_hist))
        else:
            ax.set_ylim(0, 1.7 * np.max(data_hist))
        # Ratio plot (Data / MC)
        if not only_MC:

            ratio = abs(data_hist / mc_total)
            data_ratio_err = abs(data_err / mc_total)
            mc_ratio_err =abs( mc_err / mc_total)

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
    #sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT"]
    # variables_ = ["Res_mjj_regressed", "Res_dijet_mass", "nonRes_mjj_regressed", "mass", "nonRes_dijet_mass", "minMVAID", "maxMVAID", "n_jets", "sublead_eta", "lead_eta", "sublead_pt", "lead_pt", "pt", "eta", "lead_mvaID", "sublead_mvaID"]
    # extra_vars = ["mass", "nonRes_dijet_mass", "Res_dijet_mass", "weight", "pt", "nonRes_dijet_pt", "Res_dijet_pt", "Res_lead_bjet_pt", "Res_sublead_bjet_pt", "Res_lead_bjet_ptPNetCorr", "Res_sublead_bjet_ptPNetCorr", "nonRes_HHbbggCandidate_mass", "Res_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "Res_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "jet1_mass", "jet2_mass", "jet3_mass", "jet4_mass", "jet5_mass", "jet6_mass", "Res_lead_bjet_jet_idx", "Res_sublead_bjet_jet_idx", "jet1_index", "jet2_index", "jet3_index", "jet4_index", "jet5_index", "jet6_index",
    #                        "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "jet5_pt", "jet6_pt", "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta", "jet5_eta", "jet6_eta", "jet1_phi", "jet2_phi", "jet3_phi", "jet4_phi", "jet5_phi", "jet6_phi"]

    variables_ = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "pt", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi"]
    extra_vars = []
    

    input_vars_path = args.training_config_path.replace("training_config.yaml", "input_variables.yaml")
    with open(input_vars_path, "r") as f:
        input_vars = yaml.safe_load(f)
    
    variables = variables_ + extra_vars + input_vars["mlp"]["vars"]
    # remove duplicate variables in this
    variables = list(set(variables))


    out_path = f"{base_path}/"
    plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000)
    plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000, only_MC=True)
