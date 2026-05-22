import numpy as np
import awkward as ak
import pyarrow as pa
import matplotlib.pyplot as plt
import argparse
import json
import os
import yaml
import mplhep as hep

if not hasattr(pa.lib, "PyExtensionType") and hasattr(pa.lib, "ExtensionType"):
    pa.lib.PyExtensionType = pa.lib.ExtensionType

plt.style.use(hep.style.CMS)  # Apply mlhep CMS style


# ============================================================
#  SR cut definition (hard-coded, frozen)
# ============================================================

def get_SR_cuts():
    """
    Fixed SR cuts (sequential).
    """
    return {
        # "inclusive":{
        #     "nonResReg_vbfpair_dijet_mass": (80, 190),
        # },
        # "ggHH-lowMhh-1": {
        #     "D_sig_vs_nonres": (0.99385, 1),
        #     "D_sig_vs_ttH": 0.471
        # },
        # "ggHH-lowMhh-2": {
        #     "D_sig_vs_nonres": (0.98775, 0.99385),
        #     "D_sig_vs_ttH": 0.461,
        # },
        # "ggHH-highMhh-1": {
        #     "D_sig_vs_nonres": (0.9922, 1),
        #     "D_sig_vs_ttH": 0.8
        # },
        # "ggHH-highMhh-2": {
        #     "D_sig_vs_nonres": (0.9862, 0.9922),
        #     "D_sig_vs_ttH": 0.49,
        # },
        # "ggHH-highMhh-3": {
        #     "D_sig_vs_nonres": (0.9723, 0.9862),
        #     "D_sig_vs_ttH": 0.493,
        # },
        
        # "ggHH-lowMhh-1": {
        #     "D_sig_vs_nonres": (0.99545, 1),
        #     "D_sig_vs_ttH": 0.633
        # },
        # "ggHH-lowMhh-2": {
        #     "D_sig_vs_nonres": (0.99419, 0.99545),
        #     "D_sig_vs_ttH": 0.499,
        # },
        # "ggHH-lowMhh-3": {
        #     "D_sig_vs_nonres": (0.99251, 0.99419),
        #     "D_sig_vs_ttH": 0.474,
        # },
        # "ggHH-lowMhh-4": {
        #     "D_sig_vs_nonres": (0.99009, 0.99251),
        #     "D_sig_vs_ttH": 0.46,
        # },
        
        # 100 epochs
        # "ggHH-lowMhh-1": {
        #     "D_sig_vs_nonres": (0.99335, 1),
        #     "D_sig_vs_ttH": 0.481
        # },
        # "ggHH-lowMhh-2": {
        #     "D_sig_vs_nonres": (0.987225, 0.99335),
        #     "D_sig_vs_ttH": 0.48,
        # },
        # "ggHH-lowMhh-3": {
        #     "D_sig_vs_nonres": (0.9799, 0.987225),
        #     "D_sig_vs_ttH": 0.476,
        # },
        # "ggHH-highMhh-1": {
        #     "D_sig_vs_nonres": (0.9942, 1),
        #     "D_sig_vs_ttH": 0.845
        # },
        # "ggHH-highMhh-2": {
        #     "D_sig_vs_nonres": (0.98885, 0.9942),
        #     "D_sig_vs_ttH": 0.595,
        # },
        # "ggHH-highMhh-3": {
        #     "D_sig_vs_nonres": (0.9832, 0.98885),
        #     "D_sig_vs_ttH": 0.507,
        # },
        # "ggHH-highMhh-4": {
        #     "D_sig_vs_nonres": (0.9744, 0.9832),
        #     "D_sig_vs_ttH": 0.499,
        # },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for correlation between mass and ggFHH score')
    parser.add_argument('--input_path', type=str, help='Path to the directory containing the scores and parquet files')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    args = parser.parse_args()

    # load the configuration yaml files
    training_config_path = f"{args.config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    samples_in_config = training_config["samples_info"]["preEE"].keys()
    eras = training_config["samples_info"]["eras"]
    events_path = training_config["samples_info"]["samples_path"]

    # non_resonant_samples = ["TTGG", "GGJets"]
    # sample = [s for s in samples_in_config if s in non_resonant_samples]
    #sample = ["GGJets"]
    sample_groups = {
        "GGJets+TTGG": ["TTGG", "GGJets"],
        "DDQCDGJet":  ["DDQCDGJET"],
        "GGJets+TTGG+DDQCDGJet": ["TTGG", "GGJets", "DDQCDGJET"],
        "ttHToGG":    ["ttHtoGG_M_125"],
        "ggHToGG":    ["GluGluHToGG_M_125"],
    }
    path = args.input_path

    bins = 30

    for group_name, target_samples in sample_groups.items():
        sample = [s for s in samples_in_config if s in target_samples]
        if len(sample_groups) > 1:
            out_path = args.input_path + "/cor_plots/" + group_name
        else:
            out_path = args.input_path + "/cor_plots/"
        os.makedirs(out_path, exist_ok=True)
    
        # Load scores
        y = []
        rel_w = []
        for era in eras:
            for s in sample:
                y.append(np.load(f"{path}/individual_samples/{era}/{s}/y.npy"))
                rel_w.append(np.load(f"{path}/individual_samples/{era}/{s}/rel_w.npy"))
        y = np.concatenate(y)
        rel_w = np.concatenate(rel_w)
        
        # class index: 0=non_res_bkg, 1=ttH, 2=other_singleH, 3=GluGluToHH
        # D_sig_vs_nonres = y[:, 3] / (y[:, 3] + y[:, 0] + y[:, 2])
        # D_sig_vs_ttH    = y[:, 3] / (y[:, 3] + y[:, 1])


        # Load parquet files
        events = []
        for era in eras:
            for s in sample:
                if os.path.exists(f"{path}/individual_samples/{era}/{s}/events.parquet"):
                    events.append(ak.from_parquet(f"{path}/individual_samples/{era}/{s}/events.parquet", columns=["mass", "nonResReg_vbfpair_dijet_mass"]))
                else:
                    events.append(ak.from_parquet(f"{events_path}/{training_config['samples_info'][era][s]}", columns=["mass", "nonResReg_vbfpair_dijet_mass"]))
        events = ak.concatenate(events)

        # Apply common preselection to events, scores and weights together
        mjj = np.asarray(events.nonResReg_vbfpair_dijet_mass)

        presel_mask = (mjj > 80) & (mjj < 190)

        events = events[presel_mask]

        y = y[presel_mask]

        rel_w = rel_w[presel_mask]

        # D_sig_vs_nonres = D_sig_vs_nonres[presel_mask]

        # D_sig_vs_ttH = D_sig_vs_ttH[presel_mask]

        def plot_with_errorbars(data, weights, bins, range_, label, ax, inclusive=False):
            hist, bin_edges = np.histogram(data, bins=bins, range=range_, weights=weights)
            sumw2, _ = np.histogram(data, bins=bins, range=range_, weights=weights**2)

            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_widths = np.diff(bin_edges)

            # Normalize to density if requested
            norm_factor = np.sum(hist * bin_widths)
            if norm_factor > 0:
                hist /= norm_factor
                sumw2 /= norm_factor**2

            if inclusive:
                linewidth = 4
            else:
                linewidth = 2

            errors = np.sqrt(sumw2)
            hep.histplot(
                hist,
                bin_edges,
                yerr=errors,
                label=label,
                histtype='step',
                ax=ax,
                linewidth=linewidth,
            )

        # Di-photon mass plot
        fig, ax = plt.subplots()
        for cut in [0, 0.6, 0.9, 0.95]:
            mask = y[:, 3] > cut
            # mask = y[:, 2] > cut
            plot_with_errorbars(
                data=np.array(events.mass)[mask],
                weights=rel_w[mask],
                bins=bins,
                range_=(100, 180),
                label=f"ggFHH score > {cut}",
                ax=ax
            )
        ax.set_xlabel("di-photon mass [GeV]")
        ax.set_ylabel("Normalized events")
        ax.legend()
        #hep.cms.text("Private Work", ax=ax)
        plt.title(f"{group_name}")
        plt.tight_layout()
        fig.savefig(f"{out_path}/nonResSamples_diphoton_mass_ggFHH_score_cuts.png")
        plt.clf()

        print("events:", events.fields)
        # Dijet mass plot
        fig, ax = plt.subplots()
        for cut in [0, 0.6, 0.9, 0.95]:
            mask = y[:, 3] > cut
            # mask = y[:, 2] > cut
            plot_with_errorbars(
                data=np.array(events.nonResReg_vbfpair_dijet_mass)[mask],
                weights=rel_w[mask],
                bins=bins,
                range_=(70, 190),
                label=f"ggFHH score > {cut}",
                ax=ax
            )
        ax.set_xlabel("di-jet mass DNNreg [GeV]")
        ax.set_ylabel("Normalized events")
        ax.legend()
        #hep.cms.text("Private Work", ax=ax)
        plt.title(f"{group_name}")
        plt.tight_layout()
        fig.savefig(f"{out_path}/nonResSamples_nonResReg_dijet_mass_DNNreg_score_cuts.png")
        plt.clf()



        SR_cuts = get_SR_cuts()

        # Di-photon mass sculpting check
        fig, ax = plt.subplots()
        # inclusive
        plot_with_errorbars(
            data=np.array(events.mass),
            weights=rel_w,
            bins=bins, range_=(100, 180),
            label="Inclusive", ax=ax,
            inclusive=True
        )

        remaining = np.ones(len(y), dtype=bool)
        for sr_name, cuts in SR_cuts.items():
            mask = remaining.copy()
            if "D_sig_vs_nonres" in cuts:
                lo, hi = cuts["D_sig_vs_nonres"]
                mask &= (D_sig_vs_nonres > lo) & (D_sig_vs_nonres <= hi)
            if "D_sig_vs_ttH" in cuts:
                mask &= (D_sig_vs_ttH > cuts["D_sig_vs_ttH"])
            
            plot_with_errorbars(
                data=np.array(events.mass)[mask],
                weights=rel_w[mask],
                bins=bins, range_=(100, 180),
                label=sr_name, ax=ax
            )
            remaining &= ~mask   # sequential

        ax.set_xlabel("di-photon mass [GeV]")
        ax.set_ylabel("Normalized events")
        ax.legend()
        plt.title(f"{group_name} — sculpting check by SR")
        plt.tight_layout()
        fig.savefig(f"{out_path}/nonResSamples_diphoton_mass_SR_sculpting.png")
        plt.clf()

        # Dijet mass sculpting check
        fig, ax = plt.subplots()
        plot_with_errorbars(
            data=np.array(events.nonResReg_vbfpair_dijet_mass),
            weights=rel_w,
            bins=bins, range_=(70, 190),
            label="Inclusive", ax=ax,
            inclusive=True
        )

        remaining = np.ones(len(y), dtype=bool)
        for sr_name, cuts in SR_cuts.items():
            mask = remaining.copy()
            if "D_sig_vs_nonres" in cuts:
                lo, hi = cuts["D_sig_vs_nonres"]
                mask &= (D_sig_vs_nonres > lo) & (D_sig_vs_nonres <= hi)
            if "D_sig_vs_ttH" in cuts:
                mask &= (D_sig_vs_ttH > cuts["D_sig_vs_ttH"])
            
            plot_with_errorbars(
                data=np.array(events.nonResReg_vbfpair_dijet_mass)[mask],
                weights=rel_w[mask],
                bins=bins, range_=(70, 190),
                label=sr_name, ax=ax
            )
            remaining &= ~mask

        ax.set_xlabel("di-jet mass DNNreg [GeV]")
        ax.set_ylabel("Normalized events")
        ax.legend()
        plt.title(f"{group_name} — sculpting check by SR")
        plt.tight_layout()
        fig.savefig(f"{out_path}/nonResSamples_dijet_mass_SR_sculpting.png")
        plt.clf()
        
        remaining = np.ones(len(y), dtype=bool)
        for sr_name, cuts in SR_cuts.items():
            mask = remaining.copy()
            if "D_sig_vs_nonres" in cuts:
                lo, hi = cuts["D_sig_vs_nonres"]
                mask &= (D_sig_vs_nonres > lo) & (D_sig_vs_nonres <= hi)
            if "D_sig_vs_ttH" in cuts:
                mask &= (D_sig_vs_ttH > cuts["D_sig_vs_ttH"])
            remaining &= ~mask

            for var, xlabel, range_ in [
                ("mass",                          "di-photon mass [GeV]",      (100, 180)),
                ("nonResReg_vbfpair_dijet_mass",  "di-jet mass DNNreg [GeV]", (70, 190)),
            ]:
                fig, ax = plt.subplots()
                plot_with_errorbars(np.array(events[var]),        rel_w,        bins, range_, "Inclusive", ax, inclusive=True)
                plot_with_errorbars(np.array(events[var])[mask],  rel_w[mask],  bins, range_, sr_name,     ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Normalized events")
                ax.legend()
                plt.title(f"{group_name} — {sr_name}")
                plt.tight_layout()
                fig.savefig(f"{out_path}/{sr_name}_{var}_sculpting.png")
                plt.close()