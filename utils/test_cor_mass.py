import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse
import json
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)  # Apply mlhep CMS style

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

    non_resonant_samples = ["TTGG", "GGJets"]
    sample = [s for s in samples_in_config if s in non_resonant_samples]
    path = args.input_path
    out_path = args.input_path + "/cor_plots/"
    os.makedirs(out_path, exist_ok=True)

    # Load scores
    y = []
    rel_w = []
    for s in sample:
        for era in eras:
            y.append(np.load(f"{path}/individual_samples/{era}/{s}/y.npy"))
            rel_w.append(np.load(f"{path}/individual_samples/{era}/{s}/rel_w.npy"))
    y = np.concatenate(y)
    rel_w = np.concatenate(rel_w)

    # Load parquet files
    events = []
    for era in eras:
        for s in sample:
            if os.path.exists(f"{path}/individual_samples/{era}/{s}/events.parquet"):
                events.append(ak.from_parquet(f"{path}/individual_samples/{era}/{s}/events.parquet", columns=["mass", "Res_mjj_regressed"]))
            else:
                events.append(ak.from_parquet(f"{events_path}/{training_config['samples_info'][era][s]}", columns=["mass", "Res_mjj_regressed"]))
    events = ak.concatenate(events)

    # Define SR cuts from the table
    sr_cuts = {
        "SR1": {
            "GluGluToHH_score": 0.9598,
            "non_resonant_bkg_score": 0.009,
            "ttH_score": 0.4669,
            "other_single_H_score": 0.6068
        },
        "SR2": {
            "GluGluToHH_score": 0.4168,
            "non_resonant_bkg_score": 0.0017,
            "ttH_score": 0.5362,
            "other_single_H_score": 0.7904
        },
        "SR3": {
            "GluGluToHH_score": 0.2217,
            "non_resonant_bkg_score": 0.0089,
            "ttH_score": 0.8204,
            "other_single_H_score": 0.8888
        }
    }

    def plot_with_errorbars(data, weights, bins, range_, label, ax):
        hist, bin_edges = np.histogram(data, bins=bins, range=range_, weights=weights)
        sumw2, _ = np.histogram(data, bins=bins, range=range_, weights=weights**2)

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

    # Di-photon mass plot for SRs
    fig, ax = plt.subplots(figsize=(10, 8))
    for sr_name, cuts in sr_cuts.items():
        print(sr_name)
        mask = (
            (y[:, 3] > cuts["GluGluToHH_score"]) &  # GluGluToHH_score is the 4th column (index 3)
            (y[:, 0] < cuts["non_resonant_bkg_score"]) &  # non_resonant_bkg_score is the 1st column (index 0)
            (y[:, 1] < cuts["ttH_score"]) &  # ttH_score is the 2nd column (index 1)
            (y[:, 2] < cuts["other_single_H_score"])  # other_single_H_score is the 3rd column (index 2)
        )
        plot_with_errorbars(
            data=np.array(events.mass)[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(100, 180),
            label=sr_name,
            ax=ax
        )
    ax.set_xlabel("di-photon mass [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    plt.title("Sculpting check (GGJets+TTGG)")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_diphoton_mass_SR_comparison.png")
    plt.clf()

    # Dijet mass plot for SRs
    fig, ax = plt.subplots(figsize=(10, 8))
    for sr_name, cuts in sr_cuts.items():
        mask = (
            (y[:, 3] > cuts["GluGluToHH_score"]) &
            (y[:, 0] < cuts["non_resonant_bkg_score"]) &
            (y[:, 1] < cuts["ttH_score"]) &
            (y[:, 2] < cuts["other_single_H_score"])
        )
        plot_with_errorbars(
            data=np.array(events.Res_mjj_regressed)[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(70, 190),
            label=sr_name,
            ax=ax
        )
    ax.set_xlabel("Res_mjj_regressed [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    plt.title("Sculpting check (GGJets+TTGG)")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_Res_mjj_regressed_SR_comparison.png")
    plt.clf()