
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


class OptunaCategorizer:
    def __init__(self,
                base_path,
                cat_folder=None,
                signal_class=3,
                signal_samples=None,
                samples_list=None,
                bkg_samples=None,
                n_categories=5,
                n_trials_optuna=150,
                n_runs=10,
                side_band_threshold=10,
                beta=0.1,
                gamma_strategy="linear",
                SR_strategy="sequential"
                ):

        self.base_path = base_path
        self.cat_folder = cat_folder
        self.signal_class = signal_class
        self.signal_samples = signal_samples
        self.bkg_samples = bkg_samples
        self.best_cut_params = []
        self.best_cut_params_cr = []
        self.samples_list = samples_list
        self.n_categories = n_categories
        self.n_trials_optuna = n_trials_optuna
        self.side_band_threshold = side_band_threshold
        self.n_runs = n_runs
        self.beta = beta
        self.gamma_strategy = gamma_strategy
        self.SR_strategy = SR_strategy

        if self.cat_folder is None:
            print("INFO: No output directory specified, using default: optuna_categorization")
            print("INFO: If there is a previous run with the same output directory, it will be overwritten.")
            self.cat_folder = "optuna_categorization"
        if self.samples_list is None:
            self.samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT"]
            self.samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"]
        if self.bkg_samples is None:
            # self.bkg_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "TTGG", "GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT"]
            self.bkg_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"]
        if self.signal_samples is None:
            self.signal_samples = ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"]

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
        """
        Plot the optimization history using Optuna's matplotlib interface.
        """
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig = ax.get_figure()  # Retrieve the parent Figure.
        fig.suptitle(f"Optuna Optimization History for Category {category}", fontsize=14)
        out_dir = os.path.join(out_dir, "optuna_history_plots")
        os.makedirs(out_dir, exist_ok=True)  # Ensure the output directory exists
        fig.savefig(os.path.join(out_dir, f"optuna_history_cat_{category}.png"))
        plt.clf()

    def plot_parallel_coordinates(self, study, out_dir, category):
        """
        Plot parallel coordinates using Optuna's matplotlib interface.
        """
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig = ax.get_figure()  # Retrieve the parent Figure.
        fig.suptitle(f"Parallel Coordinates for Category {category}", fontsize=14)
        out_dir = os.path.join(out_dir, "optuna_history_plots")
        fig.savefig(os.path.join(out_dir, f"parallel_coordinates_cat_{category}.png"))
        plt.clf()

    def preselection(self, events, scores):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.nonResReg_dijet_mass_DNNreg > 70) & (events.nonResReg_dijet_mass_DNNreg < 190))

        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        events = events[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool)]
        scores = scores[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool)]

        return events, scores
    
    def load_samples(self):
        """
        Load predictions and event weights across eras and samples.
        Skips missing files, copes with empty selections, and fails loudly
        only if *nothing* is collected.
        """
        data = {k: [] for k in (
            "score", "diphoton_mass", "dijet_mass",
            "weights", "labels", "sample"
        )}

        eras = ("preEE", "postEE", "preBPix", "postBPix")
        dijet_mass_key = "nonResReg_dijet_mass_DNNreg"

        for era in eras:
            for sample in self.samples_list:
                samp_dir = os.path.join(
                    self.base_path, "individual_samples", era, sample
                )
                y_file   = os.path.join(samp_dir, "y.npy")
                evt_file = os.path.join(samp_dir, "events.parquet")

                # Skip if either file is missing
                if not (os.path.exists(y_file) and os.path.exists(evt_file)):
                    print(f"[load_samples] WARNING: missing files for {samp_dir}, skipping.")
                    continue

                try:
                    y = np.load(y_file)
                    events = ak.from_parquet(
                        evt_file,
                        columns=[
                            "mass", dijet_mass_key,
                            "lead_genPartFlav", "sublead_genPartFlav",
                            "weight_tot",
                            "lead_mvaID", "sublead_mvaID",
                        ],
                    )
                    if self.apply_preselection:
                        events, y = self.preselection(events, y)
                except Exception as exc:
                    print(f"[load_samples] ERROR while reading {samp_dir}: {exc}")
                    continue

                # Prompt-photon requirement for tt̄γ‐like samples
                if sample.startswith("TTG_") or sample in {"TT", "TTGG"}:
                    sel = ((events["lead_genPartFlav"] == 1) &
                           (events["sublead_genPartFlav"] == 1))
                    events = events[sel]
                    y = y[sel]

                if len(y) == 0:  # Nothing survived the cuts
                    continue

                data["score"].append(y)
                data["diphoton_mass"].append(np.asarray(events["mass"]))
                data["dijet_mass"].append(np.asarray(events[dijet_mass_key]))
                data["weights"].append(np.asarray(events["weight_tot"]))
                data["labels"].append(
                    np.full(len(y), 1 if sample in self.signal_samples else 0, dtype=int)
                )
                data["sample"].append(np.repeat(sample, len(y)))

        if not data["score"]:
            raise RuntimeError("[load_samples] No events found in any input sample.")

        # Concatenate each list into one array
        for key in data:
            data[key] = np.concatenate(data[key], axis=0)

        scores = data.pop("score")          # shape (N, n_scores)
        argmax = np.argmax(scores, axis=1)  # safest when scores is 2-D

        # Assemble DataFrame
        df = pd.DataFrame(data)
        df["score"] = list(scores)          # store per-event score vectors
        df["arg_max_score"] = argmax

        # Quick bookkeeping
        in_peak = (df["diphoton_mass"] > 120) & (df["diphoton_mass"] < 130)
        print(f"Background weight: {df.loc[df['labels'] == 0, 'weights'].sum():.3g}")
        print(f"Signal weight:     {df.loc[df['labels'] == 1, 'weights'].sum():.3g}")
        print(f"Bkg weight 120-130 GeV: {df.loc[(df['labels'] == 0) & in_peak, 'weights'].sum():.3g}")
        print(f"Sig weight 120-130 GeV: {df.loc[(df['labels'] == 1) & in_peak, 'weights'].sum():.3g}")

        return df

    def plot_stacked_histogram(self, sim_folder, data_folder, sim_samples, variables, out_path, bins=40, mass_window=(120, 130), signal_scale=100, include_2023=True, mask=True):
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
        out_path = os.path.join(out_path, "Data_MC_plots")
        os.makedirs(out_path, exist_ok=True)
        mc_colors = [
        "#FF8A50",  # Darker Peach
        "#FFB300",  # Golden Yellow
        "#66BB6A",  # Rich Green
        "#42A5F5",  # Deeper Sky Blue
        "#AB47BC",  # Strong Lavender Purple
        #"#EC407A",  # Deeper Pink
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
            "TTGG_TTG": r"TTGG + TTG",
        }


        # Load MC and Signal Data First
        stack_mc_dict = {}
        signal_mc_dict = {}

        class_names = ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score", "VBFToHH_sig_score"]

        for sample in sim_samples:
            # Load MC events
            if not os.path.exists(f"{sim_folder}/preEE/{sample}/events.parquet"):
                print(f"samples doesn't exist: {sample}")
                sample_preEE = ak.from_parquet(f"{sim_folder}/postEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
                sample_preEE["weight_tot"] = sample_preEE["weight_tot"] * luminosities["preEE"] / luminosities["postEE"] # Adjust weight for preEE only
            else:
                sample_preEE = ak.from_parquet(f"{sim_folder}/preEE/{sample}/events.parquet", columns=variables + ["weight_tot"])


            luminosities = {
            "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
            "postEE": 26.67,  # Integrated luminosity for postEE in fb^-1
            "preBPix": 17.794,  # Integrated luminosity for preEE in fb^-1
            "postBPix": 9.451  # Integrated luminosity for postEE in fb^-1
            }

            sample_postEE = ak.from_parquet(f"{sim_folder}/postEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
            if include_2023:
                if not os.path.exists(f"{sim_folder}/preBPix/{sample}/events.parquet"):
                    print(f"samples doesn't exist: {sample} preBPix")
                    sample_preBPix = ak.from_parquet(f"{sim_folder}/postEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
                    sample_preBPix["weight_tot"] = sample_preBPix["weight_tot"] * luminosities["preBPix"] / luminosities["postEE"]
                else:
                    sample_preBPix = ak.from_parquet(f"{sim_folder}/preBPix/{sample}/events.parquet", columns=variables + ["weight_tot"])

                if not os.path.exists(f"{sim_folder}/postBPix/{sample}/events.parquet"):
                    print(f"samples doesn't exist: {sample} postBPix")
                    sample_postBPix = ak.from_parquet(f"{sim_folder}/postEE/{sample}/events.parquet", columns=variables + ["weight_tot"])
                    sample_postBPix["weight_tot"] = sample_postBPix["weight_tot"] * luminosities["postBPix"] / luminosities["postEE"]
                else:
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

        # combine a few samples into one
        stack_mc_dict["ttGG + ttG"] = ak.concatenate([stack_mc_dict["TTGG"], stack_mc_dict["TTG_100_200"], stack_mc_dict["TTG_200"]], axis=0)

        # Load Data First
        if include_2023:
            data_samples = ["2022_EraE", "2022_EraF", "2022_EraG", "2022_EraC", "2022_EraD", "2023_EraC", "2023_EraD"]
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
            "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 23, "range": (100, 180), "log": True},
            "dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
            "nonRes_dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
            "nonRes_mjj_regressed": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
            "nonResReg_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
            "Res_mjj_regressed": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 23, "range": (80, 180), "log": True},
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

            k=0
            for i, (sample, data) in enumerate(stack_mc_dict.items()):
                if ("TTG_" in sample) or (sample=="TTGG") or (sample=="TT"):
                    continue
                hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
                mc_hist.append(hist)
                mc_labels.append(sample)
                mc_colors_used.append(mc_colors[k % len(mc_colors)])
                k += 1

                # Sum of squared weights for uncertainty calculation
                hist_err, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"]))**2)
                mc_err += hist_err

            mc_total = np.sum(mc_hist, axis=0)
            mc_err = np.abs(np.sqrt(mc_err))  # Statistical uncertainty

            data_hist, _ = np.histogram(ak.to_numpy((data_combined[variable])), bins=bin_edges)
            data_err = np.sqrt(data_hist)  # Poisson errors

            # Compute total MC and Data integrals
            total_mc_yield = np.sum([np.sum(hist) for hist in mc_hist])
            total_data_yield = np.sum(data_hist)

            # Compute signal histograms with weights
            signal_histograms = {}
            for signal, data in signal_mc_dict.items():
                hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
                if "ggHH" in signal:
                    signal_histograms[signal] = hist * signal_scale
                else:
                    signal_histograms[signal] = hist * signal_scale * 10

            # Plot
            #fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, figsize=(10, 10), sharex=True )
            #ax, ax_ratio = axs
            fig, ax = plt.subplots(figsize=(10, 8))

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


            ax.fill_between(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                mc_total - mc_err,
                mc_total + mc_err,
                color="gray",
                alpha=0.3,
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

            ax.legend(fontsize=15, ncol=2, title=r"$m_{\gamma\gamma}$ blinded in [120, 130] GeV", title_fontsize=13)

            # set font size of the legend title


            # Labels
            ax.set_ylabel("Events")
            ax.set_xlim(var_config[variable]["range"])

            if var_config[variable]["log"]:
                ax.set_yscale("log")
                ax.set_ylim(0.001, 1200 * np.max(data_hist))
            else:
                ax.set_ylim(0, 1.7 * np.max(data_hist))

            ax.set_ylabel("Events", fontsize=14)
            ax.set_xlabel(var_config[variable]["label"], fontsize=14)
            plt.savefig(f"{out_path}/{variable}.png", dpi=300, bbox_inches="tight")
            plt.clf()
        print("output saved in ", out_path)

    def plot_category_summary_with_thresholds(
        self,
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
        plt.savefig(f"{save_path}category_summary_new.png")
        plt.close(fig)


    def asymptotic_significance(self, df):

        # s should be the samples in self.signal_samples
        s = df[df['sample'].isin(self.signal_samples)].weights.sum()
        b = df[df['sample'].isin(self.bkg_samples)].weights.sum()
        
        print(f"Signal: {s}, Background: {b}")
        z = np.sqrt(2 * ((s + b + 1e-10) * np.log(1 + (s / (b + 1e-10))) - s))
        #z = np.sqrt(2 * (((s + b + 1e-10) * np.log(1 + s / (b + 1e-10))) - s))
        print(f"Signal: {s}, Background: {b}, Significance: {z}")
        return z

    #############################################
    # Sequential categorization using Optuna
    #############################################
    
    def optmize_SR_sequential(self, samples_input):
        """
        For each category, use Optuna to find the best multidimensional cuts:
          - The signal score must be above a threshold.
          - Each background score must be below its respective threshold.
        In subsequent categories, the optimization search ranges are restricted:
          - Signal: (0, previous_best_signal_threshold)
          - Background (for each class): (previous_best_bg_threshold, 1)
        Additionally, the background in the sidebands (mass <120 or >130) must sum to at least 10.
        Selected events (based on a significance metric in the diphoton mass window 120-130 GeV)
        are removed from the DataFrame before optimizing the next category.
        For each category, the code stores:
          - The significance (Z)
          - The weighted signal in the peak (120-130 GeV)
          - The weighted background in the sidebands
        """
        cat_path = os.path.join(self.base_path, f"{self.cat_folder}")
        print(f"Creating output directory: {cat_path}")
        os.makedirs(cat_path, exist_ok=True)

        # Determine background classes using the first event.
        first_scores = np.stack(samples_input["score"].values)
        n_classes = first_scores.shape[1]
        bg_classes = [j for j in range(n_classes) if j != self.signal_class]

        run_significance_list = []  # Store significance for each run.
        run_best_params_list = []  # Store best parameters for each run.
        run_sig_peak_list = []  # Store signal in peak for each run.
        run_bkg_side_list = []  # Store background in sidebands for each run.
        for run in range(self.n_runs):

            # Begin with all events.
            samples_remaining = samples_input.copy()

            # Initialize dynamic search ranges.
            prev_signal_cut = 1.0  # For signal score: initial range is (0, 1).
            # For each background class, since we want to cut "less than" a threshold, we update the lower bound.
            prev_bg_cut = {b: 0.0 for b in bg_classes}

            # print numner of signal events
            n_signal_events = samples_remaining[samples_remaining["labels"] == 1].weights.sum()
            print(f"Run {run}: Number of signal events: {n_signal_events}")

            print(f"--- Run {run} ---")

            best_cut_params_list = []  # Best parameters per category.
            best_sig_values = []       # Best significance values per category.
            sig_peak_list = []         # Weighted signal in peak region for each category.
            bkg_side_list = []         # Weighted background in sideband for each category.
            mask_list = []           # Mask for selected events in each category.
            for cat in range(1, self.n_categories + 1):
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
                    th_signal = trial.suggest_float("th_signal", 0, 1)
                    mask = scores[:, self.signal_class] > th_signal
                    # Background thresholds for each background class, search in (prev_bg_cut[b], 1)
                    for b in bg_classes:
                        th_bg = trial.suggest_float(f"th_bg_{b}", 0, 1)
                        mask = mask & (scores[:, b] < th_bg)

                    if np.sum(mask) == 0:
                        return -1.0

                    # Enforce sideband requirement: background outside 120-130 GeV must have at least 10 (weighted).
                    side_mask = (((dipho_mass[mask] < 120) | (dipho_mass[mask] > 130)) & (labels[mask] == 0))
                    bkg_side_val = weights[mask][side_mask].sum()
                    if bkg_side_val < self.side_band_threshold:
                        return -1.0

                    # Select events in the diphoton mass window (120 < m_γγ < 130).
                    mass_mask = (dipho_mass[mask] > 120) & (dipho_mass[mask] < 130)
                    if np.sum(mass_mask) == 0:
                        return -1.0

                    selected_samples = samples[mask][mass_mask]
                    selected_weights = weights[mask][mass_mask]
                    df_temp = pd.DataFrame({"sample": selected_samples, "weights": selected_weights})
                    sig_val = self.asymptotic_significance(df_temp)
                    return sig_val

                # Create the sampler with your custom gamma:
                sampler = optuna.samplers.TPESampler(
                    gamma= self.gamma_fn()
                )
                study = optuna.create_study(direction="maximize", sampler=sampler)

                study.optimize(objective, n_trials=self.n_trials_optuna, show_progress_bar=False)

                best_params = study.best_params
                best_target = study.best_value
                best_cut_params_list.append(best_params)
                best_sig_values.append(best_target)
                print(f"Category {cat}: Best parameters: {best_params} with significance {best_target:.4f}")

                # Save optimization history and parallel coordinates plots.
                self.plot_optuna_history(study, cat_path, category=f"run_{run}_cat_{cat}")
                self.plot_parallel_coordinates(study, cat_path, category=f"run_{run}_cat_{cat}")

                # Use best parameters to select events for this category.
                mask = scores[:, self.signal_class] > best_params["th_signal"]
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

            self.plot_category_summary_with_thresholds(best_sig_values,
                sig_peak_list,
                bkg_side_list,
                best_cut_params_list,
                f"{cat_path}/run_{run}_")

            run_significance_list.append(best_sig_values)
            run_best_params_list.append(best_cut_params_list)
            run_sig_peak_list.append(sig_peak_list)
            run_bkg_side_list.append(bkg_side_list)

            run_sum_Z_quad = [np.sqrt(np.sum(np.array(sig)**2)) for sig in run_significance_list]

            # find the index of the maximum significance
            max_index = np.argmax(run_sum_Z_quad)

            best_sig_values = run_significance_list[max_index]
            best_cut_params_list = run_best_params_list[max_index]
            sig_peak_list = run_sig_peak_list[max_index]
            bkg_side_list = run_bkg_side_list[max_index]



        # Plot summary: cumulative Z, per-category Z, signal in peak, and bkg in sideband vs category.
        #plot_category_summary(best_sig_values, sig_peak_list, bkg_side_list, cat_path)
        # Plot summary with thresholds annotated on the significance plot.
        self.plot_category_summary_with_thresholds(best_sig_values,
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
            # also save the best run number
            f.write(f"Best run number: {max_index}\n")
            # also save the best significance value
            f.write(f"Best significance value: {best_sig_values}\n")

        # also save it as a JSON file
        best_params_json_path = os.path.join(cat_path, "best_cut_params.json")
        with open(best_params_json_path, "w") as f:
            json.dump(best_cut_params_list, f, indent=4)

        return best_cut_params_list, best_sig_values

    def store_categorization_events_with_score(
        self, 
        base_path,
        best_cut_values,
        folder_name,
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
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
            "TTG_10_100",
            "TTG_100_200",
            "TTG_200",
            "TT"
        ]
        dijet_mass_key = "nonResReg_dijet_mass_DNNreg"

        columns = ["mass", dijet_mass_key, "lead_genPartFlav", "sublead_genPartFlav", "lead_mvaID", "sublead_mvaID", "weight_tot"]

        for era in ["preEE", "postEE", "preBPix", "postBPix"]:
            for sample in samples:
                if not os.path.exists(f"{base_path}/individual_samples/{era}/{sample}"):
                    print(f"Skipping {sample} in {era} as it does not exist.")
                    continue
                inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
                print(f"Processing {inputs_path}")

                # Load events
                events = ak.from_parquet(f"{inputs_path}/events.parquet", columns=columns)
                scores = np.load(f"{inputs_path}/y.npy")
                # Apply selection if needed
                if self.apply_preselection:
                    events, scores = self.preselection(events, scores)


                if (("TTG_" in sample) or (sample == "TT") or (sample == "TTGG")):
                    print("selecting prompt photons for TTG and TT samples")
                    prompt_photon_bool = ((events.lead_genPartFlav == 1) & (events.sublead_genPartFlav == 1))
                    events = events[prompt_photon_bool]
                    scores = scores[prompt_photon_bool]

                events["dijet_mass"] = events[dijet_mass_key]

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

        # ----------------------
        # Process Data samples
        # ----------------------
        data_samples = [
            "2022_EraE",
            "2022_EraF",
            "2022_EraG",
            "2022_EraC",
            "2022_EraD",
            "2023_EraC",
            "2023_EraD"
        ]
        for data_sample in data_samples:
            inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
            print(f"Processing {inputs_path}")

            events = ak.from_parquet(f"{inputs_path}/events.parquet", columns=columns)
            scores = np.load(f"{inputs_path}/y.npy")

            # For data, weight_tot = 1
            events["weight_tot"] = ak.ones_like(events[dijet_mass_key])
            events["dijet_mass"] = events[dijet_mass_key]

            # Apply selection if needed
            if self.apply_preselection:
                events, scores = self.preselection(events, scores)

            selected_events = events
            selected_scores = scores

            # ============= SR (cat1, cat2, cat3) =============
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

    def collect_event_yields(self, base_dir, category_list, mass_range=None):
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
            "TTG_100_200", "TTG_200"
        ]
        data_samples = [
            "2022_EraE","2022_EraF","2022_EraG","2022_EraC","2022_EraD",
            "2023_EraC","2023_EraD"
        ]
        folder_to_region = {
            "cat1":"SR1","cat2":"SR2","cat3":"SR3",
            "ttH":"CR_ttH","bbH":"CR_bbH",
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

                if sample == "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
                    sample_ = "GluGlutoHH"
                else:
                    sample_ = sample

                yield_dict.setdefault(sample_, {})[
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


    def bootstrap_sample(self, nDim, w=None,nb=2000,seed=None):
        """
        nDim: dimension of the sample to be bootstrapped
        w:     For weighted events, np.array of dimension nDim with the weights.
               If w=None, the sample is unweighted.
        nb:    Number of bootstrap samples to generate
        seed:  Sets generator seeds (but I dont think it does anything, sigh,
               probably because internally it does not use np.random or it
               resets seed itself.  Dont feel like investigating at the moment.)


        Returns a list of dimension nb, each element of the list is a list of
        indeces of the given boptstrapped sample.


        Works (hopefully) with negative weights also.
        Inspired by chatgpt.

        Claudio 1 July 2025

        Example usage:

        x = np.array([10., 34., 3., 76.])  # data set to bootstrap
        w = np.array([1,   2, 0.5,  1.5  ]) # weight of each element of x
        idx = bootstrap_sample(len(x), w=w, nb=3)
        for i in range(3):
            print("sample ", i+1, "x = ", x[idx[i]])


        sample  1 x =  [34. 76. 34. 76.]
        sample  2 x =  [34. 34. 76. 76.]
        sample  3 x =  [76. 10. 76. 34.]
        """

        if seed != None: np.random.seed(seed)
        if w is not None:
            if len(w) != nDim:
                print("Error: Weight array should have dimension = ", nDim)
                return []
            probabilities = np.abs(w) / np.sum(np.abs(w))
        else:
            probabilities = None


        # Set up the weights
        outIndex = []
        for i in range(nb):
            outIndex.append(np.random.choice(nDim, size=nDim,
                                             replace=True, p=probabilities))
        return outIndex

    def weighted_pearson_correlation(self, x, y, weights):
        """
        Calculates the weighted Pearson correlation coefficient between two variables.


        Args:
            x (np.array): First variable.
            y (np.array): Second variable.
            weights (np.array): Weights for each observation.


        Returns:
            float: Weighted Pearson correlation coefficient.


        Claudio 1 July 2025.
        Got this courtesy of chatgpt.
        Seems to give same answer as WeightedCorr from wcorr package.
        """
        if len(x) != len(y) or len(x) != len(weights):
            raise ValueError("All input arrays must have the same length.")


        weighted_mean_x = np.sum(x * weights) / np.sum(weights)
        weighted_mean_y = np.sum(y * weights) / np.sum(weights)


        weighted_cov_xy = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y)) / np.sum(weights)


        weighted_std_x = np.sqrt(np.sum(weights * (x - weighted_mean_x)**2) / np.sum(weights))
        weighted_std_y = np.sqrt(np.sum(weights * (y - weighted_mean_y)**2) / np.sum(weights))


        if weighted_std_x == 0 or weighted_std_y == 0:
            return 0  # Avoid division by zero if a variable has no variance


        return weighted_cov_xy / (weighted_std_x * weighted_std_y)


    def test_mass_sculpting(self, folder, cat_list, cat_folder):

        path_for_plots = f"{folder}/{cat_folder}/mass_sculpting_plots/"
        os.makedirs(path_for_plots, exist_ok=True)

        columns_to_load = ["mass", "nonResReg_dijet_mass_DNNreg", "weight_tot"]

        preEE = ak.from_parquet(f"{folder}/individual_samples/preEE/GGJets/events.parquet", columns=columns_to_load)
        postEE = ak.from_parquet(f"{folder}/individual_samples/postEE/GGJets/events.parquet", columns=columns_to_load)
        preBPix = ak.from_parquet(f"{folder}/individual_samples/preBPix/GGJets/events.parquet", columns=columns_to_load)
        postBPix = ak.from_parquet(f"{folder}/individual_samples/postBPix/GGJets/events.parquet", columns=columns_to_load)
        # concatenate the samples
        presel_GGjets = ak.concatenate([preEE, postEE, preBPix, postBPix], axis=0)

        # load TTGG
        preEE = ak.from_parquet(f"{folder}/individual_samples/preEE/TTGG/events.parquet", columns=columns_to_load)
        postEE = ak.from_parquet(f"{folder}/individual_samples/postEE/TTGG/events.parquet", columns=columns_to_load)
        preBPix = ak.from_parquet(f"{folder}/individual_samples/preBPix/TTGG/events.parquet", columns=columns_to_load)
        postBPix = ak.from_parquet(f"{folder}/individual_samples/postBPix/TTGG/events.parquet", columns=columns_to_load)
        # concatenate the samples
        presel = ak.concatenate([preEE, postEE, preBPix, postBPix, presel_GGjets], axis=0)

        cat_to_label = {
            "cat1": "SR1",
            "cat2": "SR2",
            "cat3": "SR3",
            "cat4": "SR4",
        }

        cat_events = {}
        # loop over the categories
        for cat in cat_list:
            preEE = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/preEE/GGJets/events.parquet", columns=columns_to_load)
            postEE = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/postEE/GGJets/events.parquet", columns=columns_to_load)
            preBPix = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/preBPix/GGJets/events.parquet", columns=columns_to_load)
            postBPix = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/postBPix/GGJets/events.parquet", columns=columns_to_load)

            # load TTGG
            preEE_ttgg = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/preEE/TTGG/events.parquet", columns=columns_to_load)
            postEE_ttgg = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/postEE/TTGG/events.parquet", columns=columns_to_load)
            preBPix_ttgg = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/preBPix/TTGG/events.parquet", columns=columns_to_load)
            postBPix_ttgg = ak.from_parquet(f"{folder}/{cat_folder}/{cat}/postBPix/TTGG/events.parquet", columns=columns_to_load)

            # concatenate the samples
            events_nonRes = ak.concatenate([preEE, postEE, preBPix, postBPix, preEE_ttgg, postEE_ttgg, preBPix_ttgg, postBPix_ttgg], axis=0)
            cat_events[cat] = events_nonRes



        def plot_with_errorbars(sample, var, range_, label, ax):
            hist, bin_edges = np.histogram(np.array(sample[var]), bins=20, range=range_, weights=np.array(sample["weight_tot"]))
            sumw2, _ = np.histogram(np.array(sample[var]), bins=20, range=range_, weights=np.array(sample["weight_tot"])**2)

            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_widths = np.diff(bin_edges)

            # Normalize to density if requested
            norm_factor = np.sum(hist * bin_widths)
            if norm_factor > 0:
                hist /= norm_factor
                sumw2 /= norm_factor**2

            errors = np.sqrt(sumw2)
            hep.histplot(
                hist,
                bin_edges,
                yerr=errors,
                label=label,
                histtype='step',
                ax=ax,
                linewidth=2
            )

        # plot the mass sculpting for each category
        fig, ax = plt.subplots()
        plot_with_errorbars(presel, "mass", [100, 180], "Pre-selection", ax)
        for cat in cat_list:
            plot_with_errorbars(cat_events[cat], "mass", [100, 180], cat_to_label[cat], ax)
        ax.set_xlabel("Di-photon mass (GeV)")
        ax.set_ylabel("Normalized events")
        ax.legend()
        plt.title("GGJets+TTGG")
        plt.tight_layout()
        fig.savefig(f"{path_for_plots}/mass.png")
        plt.clf()

        fig, ax = plt.subplots()
        plot_with_errorbars(presel, "nonResReg_dijet_mass_DNNreg", [70, 190], "Pre-selection", ax)
        for cat in cat_list:
            plot_with_errorbars(cat_events[cat], "nonResReg_dijet_mass_DNNreg", [70, 190], cat_to_label[cat], ax)
        ax.set_xlabel("Dijet mass (GeV)")
        ax.set_ylabel("Normalized events")
        ax.legend()
        plt.title("GGJets+TTGG")
        plt.tight_layout()
        fig.savefig(f"{path_for_plots}/nonResReg_dijet_mass_DNNreg.png")
        plt.clf()

        # get correlation
        cat_corr_dict = {}
        for cat in cat_list:
            cat_corr_dict[cat] = {"mean": 0, "std": 0}
            cat_events_ = cat_events[cat]
            x = ak.to_numpy(cat_events_["mass"])
            y = ak.to_numpy(cat_events_["nonResReg_dijet_mass_DNNreg"])
            weights = ak.to_numpy(cat_events_["weight_tot"])

            # first get the index for bootstrap sampling
            idxs = self.bootstrap_sample(len(x), w=weights, nb=200, seed=42)

            cat_corr = []
            for i, idx in enumerate(idxs):
                x_sample = x[idx]
                y_sample = y[idx]
                weights_sample = weights[idx]

                # calculate the weighted pearson correlation
                corr = self.weighted_pearson_correlation(x_sample, y_sample, weights_sample)
                cat_corr.append(corr)
            cat_corr = np.array(cat_corr)
            cat_corr_dict[cat]["mean"] = np.mean(cat_corr)
            cat_corr_dict[cat]["std"] = np.std(cat_corr)

        for cat in cat_list:
            cat_events_ = cat_events[cat]
            fig, ax = plt.subplots()
            plot_with_errorbars(cat_events_, "nonResReg_dijet_mass_DNNreg", [70, 190], r"$m_{\gamma \gamma} = [100, 180] GeV$", ax)
            plot_with_errorbars(cat_events_[cat_events_.mass < 125], "nonResReg_dijet_mass_DNNreg", [70, 190], r"$m_{\gamma \gamma} < 125 GeV$", ax)
            plot_with_errorbars(cat_events_[cat_events_.mass > 125], "nonResReg_dijet_mass_DNNreg", [70, 190], r"$m_{\gamma \gamma} > 125 GeV$", ax)
            ax.set_xlabel("Dijet regressed mass (GeV)")
            ax.set_ylabel("Normalized events")
            ax.legend(title=f"Weigted Pea. Corr.: {cat_corr_dict[cat]['mean']:.2f} $\pm$ {cat_corr_dict[cat]['std']:.2f}")
            plt.title(f"GGJets+TTGG - {cat}")
            plt.tight_layout()
            fig.savefig(f"{path_for_plots}/nonResReg_dijet_mass_DNNreg_diff_mass_{cat}.png")
            plt.clf()


    def run_categorisation(self):
    
        # load the samples
        samples_input = self.load_samples()

        if self.SR_strategy == "sequential":
            best_params, best_sig_values = self.optmize_SR_sequential(samples_input)

        # load the best cut values
        with open(f"{self.base_path}/{self.cat_folder}/best_cut_params.json", "r") as f:
            best_cut_values = json.load(f)

        self.store_categorization_events_with_score(
            self.base_path,
            best_cut_values,
            folder_name=self.cat_folder,
            )
        
        folder_list = [f"cat{i}" for i in range(1, 4)]

        # plots for sculpting test
        print("Testing mass sculpting...")
        self.test_mass_sculpting(f"{self.base_path}/", folder_list, self.cat_folder)

        # plot data-MC for SRs
        sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"]
        variables = ["mass", "nonResReg_dijet_mass_DNNreg"]
        for folder in folder_list:
            sim_folder = f"{self.base_path}/{self.cat_folder}/{folder}"
            data_folder = f"{self.base_path}/{self.cat_folder}/{folder}"
            out_path = f"{self.base_path}/{self.cat_folder}/{folder}"
            self.plot_stacked_histogram(sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=100)

        # collect event yields
        self.collect_event_yields(
            f"{self.base_path}/{self.cat_folder}/",
            folder_list,
            mass_range=(100, 180)  # Example mass range, adjust as needed
        )



#############################################
# Main execution
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize multiclass scores using Optuna with dynamic search ranges, sideband requirements, and summary plots.")
    parser.add_argument("--n_categories", type=int, default=5, help="Number of categories to optimize")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the input samples")
    parser.add_argument("--optuna_folder", type=str, default="optuna_categorization", help="Folder name for Optuna results")
    parser.add_argument("--n_trials", type=int, default=150, help="Number of trials for Optuna optimization")
    parser.add_argument("--SR_strategy", type=str, choices=["sequential", "simultaneous"], default="sequential", help="Strategy for SR categorization")
    parser.add_argument("--n_runs", type=int, default=15, help="Number of complete runs for the categorization")
    parser.add_argument("--gamma_strategy", type=str, choices=["sqrt", "linear"], default="linear", help="Gamma strategy for TPE sampler")
    parser.add_argument("--side_band_threshold", type=int, default=10, help="Threshold for sideband requirements")

    args = parser.parse_args()

    categoriser = OptunaCategorizer(base_path=args.base_path,
                                    cat_folder=args.optuna_folder,
                                    n_categories=args.n_categories,
                                    n_trials_optuna=args.n_trials,
                                    n_runs=args.n_runs,
                                    SR_strategy=args.SR_strategy)
    categoriser.run_categorisation()

    