import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse
import json
import os
import yaml
import mplhep as hep
plt.style.use(hep.style.CMS)  # Apply mlhep CMS style


def find_signal_class(class_names, signal_type):
    """
    Find the signal class by name or pattern.

    Args:
        class_names: List of class names from config
        signal_type: Either a shorthand ('ggF', 'VBF'), an exact class name,
                     or a substring to match

    Returns:
        Tuple of (class_name, index) or None if not found
    """
    # Define pattern mappings for common shorthands
    pattern_map = {
        'ggf': ['gluglutohh', 'ggfhh', 'gghh'],
        'vbf': ['vbfhh', 'vbftohh'],
    }

    signal_type_lower = signal_type.lower()

    # First, check if it's an exact match
    for i, class_name in enumerate(class_names):
        if class_name == signal_type:
            return (class_name, i)

    # Second, check if it's a known shorthand pattern
    if signal_type_lower in pattern_map:
        patterns = pattern_map[signal_type_lower]
        for i, class_name in enumerate(class_names):
            if any(p in class_name.lower() for p in patterns):
                return (class_name, i)

    # Third, try substring matching (case-insensitive)
    for i, class_name in enumerate(class_names):
        if signal_type_lower in class_name.lower():
            return (class_name, i)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for correlation between mass and signal score')
    parser.add_argument('--input_path', type=str, help='Path to the directory containing the scores and parquet files')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--signal_type', type=str, default='sig',
                        help='Signal class to use. Can be: exact class name (e.g., "is_sig"), '
                             'shorthand ("ggF", "VBF"), or substring match (e.g., "sig"). Default: sig')
    args = parser.parse_args()

    # load the configuration yaml files
    training_config_path = f"{args.config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    # Get class names and find the signal class
    class_names = training_config["classes"]
    print(f"Class names from config: {class_names}")

    # Find the signal class
    signal_result = find_signal_class(class_names, args.signal_type)

    if signal_result is None:
        raise ValueError(f"Could not find signal class matching '{args.signal_type}' in config. "
                         f"Available classes: {class_names}")

    signal_class_name, signal_class_idx = signal_result
    print(f"Using signal class: {signal_class_name} at index {signal_class_idx}")

    # Get var_prefix from training config, default to "nonResReg" for backwards compatibility
    var_prefix = training_config.get("var_prefix", "nonResReg")
    dijet_mass_var = f"{var_prefix}_dijet_mass_DNNreg"
    print(f"Using dijet mass variable: {dijet_mass_var}")

    # Get first available era for sample list
    eras = training_config["samples_info"]["eras"]
    first_era = eras[0]
    samples_in_config = training_config["samples_info"][first_era].keys()
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
    for era in eras:
        for s in sample:
            y.append(np.load(f"{path}/individual_samples/{era}/{s}/y.npy"))
            rel_w.append(np.load(f"{path}/individual_samples/{era}/{s}/rel_w.npy"))
    y = np.concatenate(y)
    rel_w = np.concatenate(rel_w)

    # Load parquet files
    events = []
    for era in eras:
        for s in sample:
            if os.path.exists(f"{path}/individual_samples/{era}/{s}/events.parquet"):
                events.append(ak.from_parquet(f"{path}/individual_samples/{era}/{s}/events.parquet", columns=["mass", dijet_mass_var]))
            else:
                events.append(ak.from_parquet(f"{events_path}/{training_config['samples_info'][era][s]}", columns=["mass", dijet_mass_var]))
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
    for cut in [0, 0.6, 0.9, 0.95, 0.98, 0.99]:
        mask = y[:, signal_class_idx] > cut
        plot_with_errorbars(
            data=np.array(events.mass)[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(100, 180),
            label=f"{signal_class_name} score > {cut}",
            ax=ax
        )
    ax.set_xlabel("di-photon mass [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    #hep.cms.text("Private Work", ax=ax)
    plt.title("GGJets+TTGG")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_diphoton_mass_{signal_class_name}_score_cuts.png")
    plt.clf()

    # Dijet mass plot
    fig, ax = plt.subplots()
    for cut in [0, 0.6, 0.9, 0.95, 0.98, 0.99]:
        mask = y[:, signal_class_idx] > cut
        plot_with_errorbars(
            data=np.array(events[dijet_mass_var])[mask],
            weights=rel_w[mask],
            bins=30,
            range_=(70, 190),
            label=f"{signal_class_name} score > {cut}",
            ax=ax
        )
    ax.set_xlabel(f"{dijet_mass_var} [GeV]")
    ax.set_ylabel("Normalized events")
    ax.legend()
    #hep.cms.text("Private Work", ax=ax)
    plt.title("GGJets+TTGG")
    plt.tight_layout()
    fig.savefig(f"{out_path}/nonResSamples_{dijet_mass_var}_{signal_class_name}_score_cuts.png")
    plt.clf()
