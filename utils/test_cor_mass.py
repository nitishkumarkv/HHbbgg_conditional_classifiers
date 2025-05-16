import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse
import json
import os
import yaml

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

    non_resonant_samples = ["TTGG", "GGJets", "DDQCDGJET"]
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
            events.append(ak.from_parquet(f"{events_path}/{training_config['samples_info'][era][s]}", columns=["mass", "Res_mjj_regressed"]))
    events = ak.concatenate(events)

    # plot di_photon and dijet mass for various ggFHH score cuts
    for cut in [0, 0.6, 0.9, 0.95]:
        plt.hist(np.array(events.mass)[y[:, 3] > cut], bins=30, range=(100, 180), weights=rel_w[y[:, 3] > cut], histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.xlabel("di-photon mass")
    plt.savefig(f"{out_path}/nonResSamples_diphoton_mass_ggFHH_score_cuts.png")
    plt.clf()

    for cut in [0, 0.6, 0.9, 0.95]:
        plt.hist(np.array(events.Res_mjj_regressed)[y[:, 3] > cut], bins=30, range=(70, 190), weights=rel_w[y[:, 3] > cut], histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.xlabel("Res_mjj_regressed")
    plt.savefig(f"{out_path}/nonResSamples_Res_mjj_regressed_score_cuts.png")
    plt.clf()
