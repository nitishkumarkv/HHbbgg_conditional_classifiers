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
    #sample = ["GGJets"]
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
                events.append(ak.from_parquet(f"{path}/individual_samples/{era}/{s}/events.parquet", columns=["mass", "nonResReg_dijet_mass_DNNreg"]))
            else:
                events.append(ak.from_parquet(f"{events_path}/{training_config['samples_info'][era][s]}", columns=["mass", "nonResReg_dijet_mass_DNNreg"]))
    events = ak.concatenate(events)


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

    # Di-photon mass plot
    fig, ax = plt.subplots()
    for cut in [0, 0.6, 0.9, 0.95]:
        mask = y[:, 3] > cut
        plot_with_errorbars(
            data=np.array(events.mass)[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(100, 180),
            label=f"ggFHH score > {cut}",
            ax=ax
        )
    ax.set_xlabel("di-photon mass [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    #hep.cms.text("Private Work", ax=ax)
    plt.title("GGJets+TTGG")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_diphoton_mass_ggFHH_score_cuts.png")
    plt.clf()

    # Dijet mass plot
    fig, ax = plt.subplots()
    for cut in [0, 0.6, 0.9, 0.95]:
        mask = y[:, 3] > cut
        plot_with_errorbars(
            data=np.array(events.nonResReg_dijet_mass_DNNreg)[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(70, 190),
            label=f"ggFHH score > {cut}",
            ax=ax
        )
    ax.set_xlabel("di-jet mass DNNreg [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    #hep.cms.text("Private Work", ax=ax)
    plt.title("GGJets+TTGG")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_nonResReg_dijet_mass_DNNreg_score_cuts.png")
    plt.clf()
