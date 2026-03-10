#!/usr/bin/env python3

import os
import awkward as ak
import numpy as np
import yaml
import matplotlib.pyplot as plt

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import json
import mplhep
from matplotlib.backends.backend_pdf import PdfPages



# ============================================================
#  SR cut definition (hard-coded, frozen)
# ============================================================

def get_SR_cuts():
    """
    Fixed SR cuts (sequential).
    """
    return {
        "inclusive":{
            "nonResReg_dijet_mass_DNNreg": (80, 190),
        },
        # "lowMHH":{
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (0, 350),
        # },
        # "highMHH":{
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (350, "inf"),
        # },
        # "SR11": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (0, 350),
        #     "D_sig_vs_ttH": 0.286,
        #     "D_sig_vs_nonres": (0.9938, 1.0),
        # },
        # "SR12": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (0, 350),
        #     "D_sig_vs_ttH": 0.297,
        #     "D_sig_vs_nonres": (0.9883, 0.9938),
        # },
        # "SR13": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (0, 350),
        #     "D_sig_vs_ttH": 0.286,
        #     "D_sig_vs_nonres": (0.9819, 0.9883),
        # },
        # "SR14": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (0, 350),
        #     "D_sig_vs_ttH": 0.286,
        #     "D_sig_vs_nonres": (0.96815, 0.9819),
        # },
        # "SR21": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (350, "inf"),
        #     "D_sig_vs_ttH": 0.904,
        #     "D_sig_vs_nonres": (0.99525, 1.0),
        # },
        # "SR22": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (350, "inf"),
        #     "D_sig_vs_ttH": 0.898,
        #     "D_sig_vs_nonres": (0.993, 0.99525),
        # },
        # "SR23": {
        #     "nonResReg_dijet_mass_DNNreg": (80, 190),
        #     "nonResReg_vbfpair_HHbbggCandidate_mass": (350, "inf"),
        #     "D_sig_vs_ttH": 0.673,
        #     "D_sig_vs_nonres": (0.9871, 0.993),
        # },
        # "ttH_CR": {
        #     "ttH_score": 0.98,  # using ttH_score
        # },
        # "SR1": {
        #     "D_sig_vs_ttH": 0.89,
        #     "D_sig_vs_nonres": (0.9982, 1.0),
        # },
        # "SR2": {
        #     "D_sig_vs_ttH": 0.761,
        #     "D_sig_vs_nonres": (0.9932, 0.9982),
        # },
        # "SR3": {
        #     "D_sig_vs_ttH": 0.72,
        #     "D_sig_vs_nonres": (0.984, 0.9932),
        # },
        # "SR4": {
        #     "D_sig_vs_ttH": 0.905,
        #     "D_sig_vs_nonres": (0.9628, 0.984),
        # },
        # "SR5": {
        #     "D_sig_vs_ttH": 0.715,
        #     "D_sig_vs_nonres": (0.948, 0.9628),
        # },
        # "ttH_CR": {
        #     "ttH_score": 0.98,  # using ttH_score
        # },
    }
def plot_stacked_histogram_from_events(
    mc_events_dict,     # dict: sample -> awkward array (already SR-selected)
    data_events,        # awkward array (already SR-selected)
    variables,
    out_path,
    signal_scale=1000,
    mass_window=(120, 130),
):
    only_MC = True
    """
    Plot stacked MC vs Data histograms from SR-selected events.
    - SR cuts are assumed to be already applied
    - Data is blinded in the diphoton mass window
    - Plotting style is kept identical to the original function
    """

    import os
    import numpy as np
    import awkward as ak
    import matplotlib.pyplot as plt
    import mplhep as hep

    os.makedirs(out_path, exist_ok=True)

    # --------------------------------------------------
    # SAME color palette as original code
    # --------------------------------------------------
    mc_colors = [
        "#FF8A50",  # Darker Peach
        "#FFB300",  # Golden Yellow
        "#66BB6A",  # Rich Green
        # "#42A5F5",  # Deeper Sky Blue
        "#AB47BC",  # Strong Lavender Purple
        # "#EC407A",  # Deeper Pink
        "#C0CA33",  # Darker Lime
        "#26A69A",  # Deep Teal
        "#1976D2",  # Lighter Blue
        "#EF5350",  # Lighter Red
        "#795548",  # Deep Brown
        "#757575",  # Medium Gray
        "#66BB6A",  # Light Green
    ]

    # --------------------------------------------------
    # Variable config (keep identical ranges / bins)
    # --------------------------------------------------
    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 23, "range": (100, 180), "log": True},
        "nonResReg_vbfpair_HHbbggCandidate_mass": {"label": r"$m_{\mathrm{HHbbgg}}^{reg}$ [GeV]", "bins": 40, "range": (0, 4000), "log": True},
        "nonResReg_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 190), "log": True},
        "D_sig_vs_ttH": {"label": r"$D_{\mathrm{sig/ttH}}$", "bins": 30, "range": (0, 1), "log": True},
        "D_sig_vs_nonres": {"label": r"$D_{\mathrm{sig/nonres}}$", "bins": 30, "range": (0, 1), "log": True},
        "non_resonant_bkg_score": {"label": "non_resonant_bkg_score", "bins": 30, "range": (0.95, 1), "log": True},
        "ttH_score": {"label": "ttH_score", "bins": 30, "range": (0, 1), "log": True},
        "other_single_H_score": {"label": "other_single_H_score", "bins": 30, "range": (0, 1), "log": True},
        "GluGluToHH_score": {"label": "GluGluToHH_score", "bins": 30, "range": (0, 1), "log": True},
        "lead_mvaID": {"label": "lead photon MVA ID", "bins": 30, "range": (-0.7, 1), "log": True},
        "sublead_mvaID": {"label": "sublead photon MVA ID", "bins": 30, "range": (-0.7, 1), "log": True},
    }

    label_map = {
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": r"ggHH SM",
        "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": r"ggHH $\kappa_\lambda=0.00$",
        "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": r"ggHH $\kappa_\lambda=2.45$",
        "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": r"ggHH $\kappa_\lambda=5.00$",
        "VBFHHto2B2G_CV_1_C2V_1_C3_1": r"VBF HH",
    }
    # --------------------------------------------------
    # Split signal / background (same logic as before)
    # --------------------------------------------------
    stack_mc_dict = {}
    signal_mc_dict = {}

    for sample, events in mc_events_dict.items():
        if events is None or len(events) == 0:
            continue
        if "HH" in sample:
            signal_mc_dict[sample] = events
        else:
            stack_mc_dict[sample] = events

    # --------------------------------------------------
    # Blind DATA in mgg window (ONLY data)
    # --------------------------------------------------
    if "mass" in data_events.fields:
        data_events = data_events[
            (data_events.mass < mass_window[0]) |
            (data_events.mass > mass_window[1])
        ]

    # --------------------------------------------------
    # Loop over variables
    # --------------------------------------------------
    for variable in variables:
        if variable not in data_events.fields:
            print(f"[skip] variable {variable} not in data")
            continue

        cfg = var_config[variable]
        bin_edges = np.linspace(*cfg["range"], cfg["bins"] + 1)

        mc_hist = []
        mc_err = np.zeros(len(bin_edges) - 1)
        mc_labels = []
        mc_colors_used = []

        # ----------------------------
        # MC histograms
        # ----------------------------
        k = 0
        for sample, events in stack_mc_dict.items():
            values = ak.to_numpy(events[variable])
            weights = ak.to_numpy(events["weight_tot"])

            hist, _ = np.histogram(values, bins=bin_edges, weights=weights)
            err2, _ = np.histogram(values, bins=bin_edges, weights=weights**2)

            mc_hist.append(hist)
            mc_err += err2
            mc_labels.append(sample)
            mc_colors_used.append(mc_colors[k % len(mc_colors)])
            k += 1

        mc_total = np.sum(mc_hist, axis=0)
        mc_err = np.sqrt(mc_err)

        # ----------------------------
        # Data histogram
        # ----------------------------
        data_vals = ak.to_numpy(data_events[variable])
        data_hist, _ = np.histogram(data_vals, bins=bin_edges)
        data_err = np.sqrt(data_hist)
        
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     ratio = data_hist / mc_total
        #     ratio_err = data_err / mc_total
        #     mc_ratio_err = mc_err / mc_total

        # template solution for those bins where mc_total <=0
        ####### begin ########
        ratio = np.zeros_like(mc_total, dtype=float)
        ratio_err = np.zeros_like(mc_total, dtype=float)
        mc_ratio_err = np.zeros_like(mc_total, dtype=float)

        valid = mc_total > 0
        ratio[valid] = data_hist[valid] / mc_total[valid]
        ratio_err[valid] = data_err[valid] / mc_total[valid]
        mc_ratio_err[valid] = mc_err[valid] / mc_total[valid]
        ####### end #######

        # ----------------------------
        # Signal histograms (same style)
        # ----------------------------
        signal_hists = {}
        for sample, events in signal_mc_dict.items():
            values = ak.to_numpy(events[variable])
            weights = ak.to_numpy(events["weight_tot"])
            hist, _ = np.histogram(values, bins=bin_edges, weights=weights)
            signal_hists[sample] = hist * signal_scale

        # ----------------------------
        # Plot (KEEP ORIGINAL STYLE)
        # ----------------------------
        if only_MC:
            fig, ax = plt.subplots(figsize=(10, 8))
        else: 
            fig, axs = plt.subplots(
                2, 1,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
                figsize=(10, 10),
                sharex=True
            )
            ax, ax_ratio = axs

        hep.cms.label(
            data=True,
            lumi=None,
            ax=ax,
            label="Private Work",
            com=13.6,
        )

        hep.histplot(
            mc_hist,
            bin_edges,
            histtype="fill",
            stack=True,
            label=mc_labels,
            color=mc_colors_used,
            # edgecolor="black",
            ax=ax,
        )

        ax.fill_between(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            mc_total - mc_err,
            mc_total + mc_err,
            color="gray",
            alpha=0.3,
            step="mid",
        )
        
        print("mc_total min =", np.nanmin(mc_total))
        print("mc_total <= 0 bins =", np.where(mc_total <= 0))
        print("mc_total values =", mc_total)
        print("ratio_err min =", np.nanmin(ratio_err))
        print("mc_ratio_err min =", np.nanmin(mc_ratio_err))

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

        for sample, hist in signal_hists.items():
            label = label_map.get(sample, sample)

            ax.step(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                hist,
                where="mid",
                linestyle="dashed",
                linewidth=2,
                label=f"{label} × {signal_scale}",
            )

        ax.set_xlim(cfg["range"])
        ax.set_xlabel(cfg["label"], fontsize=14)
        ax.set_ylabel("Events", fontsize=14)
        ax.set_ylim(0.001, 5000 * max(np.max(mc_total), np.max(data_hist)))

        if cfg["log"]:
            ax.set_yscale("log")
            ax.set_ylim(0.1, max(5000 * np.max(data_hist), 10))

        ax.legend(
            fontsize=13,
            ncol=2,
            title=r"$m_{\gamma\gamma}$ blinded in [120,130] GeV",
            title_fontsize=12,
        )

        if not only_MC:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Data / MC points
            ax_ratio.errorbar(
                centers,
                ratio,
                yerr=ratio_err,
                fmt="o",
                color="black",
                markersize=5,
            )

            # MC uncertainty band
            ax_ratio.fill_between(
                centers,
                1 - mc_ratio_err,
                1 + mc_ratio_err,
                color="gray",
                alpha=0.3,
                step="mid",
            )

            ax_ratio.axhline(1.0, linestyle="--", color="gray")
            ax_ratio.set_ylim(0, 2)
            ax_ratio.set_ylabel("Data / MC")
            ax_ratio.set_xlabel(var_config[variable]["label"])
        
        plt.tight_layout()
        plt.savefig(f"{out_path}/{variable}.png", dpi=300)
        plt.close()

    print(f"[done] plots saved in {out_path}")
# ============================================================
#  SR Plotter (lightweight, no Optuna)
# ============================================================

class SRPlotter:
    def __init__(self, base_path, training_config_path):
        self.base_path = base_path
        self.SR_cuts = get_SR_cuts()

        with open(training_config_path, "r") as f:
            self.training_config = yaml.safe_load(f)

        self.samples_info = self.training_config["samples_info"]
        self.eras = self.samples_info["eras"] 
        # self.eras = ["preEE", "postEE", "preBPix", "postBPix", "2024"]

        self.signal_samples = [
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00"
        ]

        self.bkg_samples = [
            "GGJets", "DDQCDGJET", "TTGG",
            "ttHtoGG_M_125", 
            # "BBHto2G_M_125",
            "GluGluHToGG_M_125",
            "VBFHToGG_M_125", "VHtoGG_M_125",
        ]
        
        self.class_names = [
            "non_resonant_bkg_score",
            "ttH_score",
            "other_single_H_score",
            "GluGluToHH_score",
            # "VBFToHH_sig_score",
        ]

    # --------------------------------------------------------
    # Preselection
    # --------------------------------------------------------
    def preselection(self, events):
        mask = (
            (events.mass > 100) & (events.mass < 180) &
            (events.nonResReg_dijet_mass_DNNreg > 70) &
            (events.nonResReg_dijet_mass_DNNreg < 190) &
            (events.lead_mvaID > -0.7) &
            (events.sublead_mvaID > -0.7)
        )
        return events[mask]

    # --------------------------------------------------------
    # Add discriminators
    # --------------------------------------------------------
    def add_discriminators(self, events):
        events["D_sig_vs_ttH"] = (
            events.GluGluToHH_score /
            (events.GluGluToHH_score + events.ttH_score)
        )
        events["D_sig_vs_nonres"] = (
            events.GluGluToHH_score /
            (events.GluGluToHH_score +
             events.non_resonant_bkg_score +
             events.other_single_H_score)
        )
        return events

    # --------------------------------------------------------
    # Load one MC sample (all eras)
    # --------------------------------------------------------
    def load_mc_sample(self, sample, variables):
        out = []

        for era in self.eras:
            path = f"{self.base_path}/individual_samples/{era}/{sample}"
            if not os.path.exists(path):
                continue

            events = ak.from_parquet(
                f"{path}/events.parquet",
                columns=variables + ["weight_tot"]
            )

            scores = np.load(f"{path}/y.npy")
            for i, name in enumerate(self.class_names):
                events[name] = scores[:, i]

            events = self.preselection(events)
            events = self.add_discriminators(events)

            out.append(events)

        return ak.concatenate(out) if out else None

    # --------------------------------------------------------
    # Load data (all eras)
    # --------------------------------------------------------
    def load_data(self, variables):
        data = None
        data_samples = self.training_config["samples_info"]["data"]

        for sample, rel_path in data_samples.items():
            path = f"{self.base_path}/individual_samples_data/{sample}"
            if not os.path.exists(path):
                continue
            if ("16" in sample) & ("17" in sample) & ("18" in sample):
                continue

            events = ak.from_parquet(
                f"{path}/events.parquet",
                columns=variables
            )

            if os.path.exists(f"{path}/y.npy"):
                scores = np.load(f"{path}/y.npy")
                for i, name in enumerate(self.class_names):
                    events[name] = scores[:, i]

            events["weight_tot"] = ak.ones_like(events.mass)
            events = self.preselection(events)
            events = self.add_discriminators(events)

            data = events if data is None else ak.concatenate([data, events])

        return data

    # --------------------------------------------------------
    # Apply sequential SR
    # --------------------------------------------------------
    def apply_SR(self, events):
        out = {}
        remaining = ak.ones_like(events.mass, dtype=bool)

        for sr, cuts in self.SR_cuts.items():
            mask = remaining
            if "ttH_score" in cuts:
                mask = mask & (events.ttH_score > cuts["ttH_score"])
            if "D_sig_vs_nonres" in cuts:
                lo, hi = cuts["D_sig_vs_nonres"]
                mask = mask & (events.D_sig_vs_nonres > lo) & (events.D_sig_vs_nonres <= hi)
            if "D_sig_vs_ttH" in cuts:
                mask = mask & (events.D_sig_vs_ttH > cuts["D_sig_vs_ttH"])
            if "nonResReg_vbfpair_HHbbggCandidate_mass" in cuts:
                lo, hi = cuts["nonResReg_vbfpair_HHbbggCandidate_mass"]
                if hi == "inf":
                    mask = mask & (events.nonResReg_vbfpair_HHbbggCandidate_mass > lo)
                else:
                    mask = mask & (events.nonResReg_vbfpair_HHbbggCandidate_mass > lo) & (events.nonResReg_vbfpair_HHbbggCandidate_mass <= hi)
            if "nonResReg_dijet_mass_DNNreg" in cuts:
                lo, hi = cuts["nonResReg_dijet_mass_DNNreg"]
                mask = mask & (events.nonResReg_dijet_mass_DNNreg > lo) & (events.nonResReg_dijet_mass_DNNreg <= hi)

            out[sr] = events[mask]
            remaining = remaining & ~mask

        return out

    # --------------------------------------------------------
    # Main plotting interface
    # --------------------------------------------------------
    def plot_SR(self, variables, out_dir, plot_func, signal_scale=1000):
        os.makedirs(out_dir, exist_ok=True)

        # load data once
        data_all = self.load_data(variables)
        data_SR  = self.apply_SR(data_all)
        
        # self.sample_order = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET"]
        self.sample_order = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET"]

        mc_all = {
            s: self.load_mc_sample(s, variables)
            for s in self.sample_order
            if s in self.signal_samples + self.bkg_samples
        }

        mc_SR = {
            sr: {
                s: self.apply_SR(mc_all[s])[sr]
                for s in mc_all
            }
            for sr in self.SR_cuts
        }

        for sr in self.SR_cuts:
            print(f"Plotting {sr}")

            sim_folder = f"{self.base_path}/individual_samples"
            data_folder = f"{self.base_path}/individual_samples_data"

            plot_stacked_histogram_from_events(
                mc_events_dict = mc_SR[sr],
                data_events    = data_SR[sr],
                variables      = variables,
                out_path       = f"{out_dir}/{sr}",
                signal_scale   = signal_scale,
            )


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--out-dir", default="SR_plots")

    args = parser.parse_args()

    variables = [
        "mass",
        "nonResReg_dijet_mass_DNNreg",
        "nonResReg_vbfpair_HHbbggCandidate_mass",
        "D_sig_vs_ttH",
        "D_sig_vs_nonres",
        "non_resonant_bkg_score",
        "ttH_score",
        "other_single_H_score",
        "GluGluToHH_score",
        "lead_mvaID",
        "sublead_mvaID",
    ]

    plotter = SRPlotter(
        base_path=args.base_path,
        training_config_path=args.training_config,
    )

    plotter.plot_SR(
        variables=variables,
        out_dir=args.out_dir,
        plot_func=plot_stacked_histogram_from_events,
        signal_scale=1000,
    )
