import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check for correlation between mass and ggFHH score')
    parser.add_argument('--input_path', type=str, help='Path to the directory containing the scores and parquet files')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    args = parser.parse_args()

    with open(f"{args.config_path}/samples_and_classes.json", 'r') as f:
        samples_and_classes = json.load(f)
    samples_in_config = samples_and_classes["sample_to_class"].keys()
    non_resonant_samples = ["TTGG", "GGJets", "DDQCDGJET"]
    sample = [s for s in samples_in_config if s in non_resonant_samples]
    #sample = ["GGJets"]
    path = args.input_path
    out_path = args.input_path + "/cor_plots/"
    os.makedirs(out_path, exist_ok=True)

    # Load scores
    y = []
    for s in sample:
        y_preEE = np.load(f"{path}/individual_samples/preEE/{s}/y.npy")
        y_postEE = np.load(f"{path}/individual_samples/postEE/{s}/y.npy")
        y_preBPix = np.load(f"{path}/individual_samples/preBPix/{s}/y.npy")
        y_postBPix = np.load(f"{path}/individual_samples/postBPix/{s}/y.npy")
        #y.append(np.concatenate((y_preEE, y_postEE)))
        y.append(np.concatenate((y_preEE, y_postEE, y_preBPix, y_postBPix)))
    y = np.concatenate(y)

    # Load parquet files
    events = []
    for s in sample:
        preEE = ak.from_parquet(f"{path}/individual_samples/preEE/{s}/events.parquet")
        postEE = ak.from_parquet(f"{path}/individual_samples/postEE/{s}/events.parquet")
        preBPix = ak.from_parquet(f"{path}/individual_samples/preBPix/{s}/events.parquet")
        postBPix = ak.from_parquet(f"{path}/individual_samples/postBPix/{s}/events.parquet")
        #events.append(ak.concatenate([preEE, postEE]))
        events.append(ak.concatenate([preEE, postEE, preBPix, postBPix]))
    events = ak.concatenate(events)

    # Load rel_w
    rel_w = []
    for s in sample:
        rel_w_preEE = np.load(f"{path}/individual_samples/preEE/{s}/rel_w.npy")
        rel_w_postEE = np.load(f"{path}/individual_samples/postEE/{s}/rel_w.npy")
        rel_w_preBPix = np.load(f"{path}/individual_samples/preBPix/{s}/rel_w.npy")
        rel_w_postBPix = np.load(f"{path}/individual_samples/postBPix/{s}/rel_w.npy")
        #rel_w.append(np.concatenate((rel_w_preEE, rel_w_postEE)))
        rel_w.append(np.concatenate((rel_w_preEE, rel_w_postEE, rel_w_preBPix, rel_w_postBPix)))
    rel_w = np.concatenate(rel_w)

    # plot 2d histogram of diphoton_mass vs score
    plt.hist2d(np.array(events.mass), y[:, 3], bins=(50, 50), range=((100, 180), (0, 0.2)), weights=rel_w, cmap=plt.cm.viridis)
    plt.yscale('log')
    # the z axis should be in log scale
    #plt.scatter(np.array(events.mass), y[:, 3], s=1, c=rel_w, cmap=plt.cm.viridis, alpha=0.5)
    plt.savefig(f"{out_path}/nonResSamples_diphoton_mass_vs_ggFHH_score.png")
    plt.clf()

    # plot di_photon and dijet mass for various ggFHH score cuts
    for cut in [0, 0.6, 0.9]:
        plt.hist(np.array(events.mass)[y[:, 3] > cut], bins=30, range=(100, 180), weights=rel_w[y[:, 3] > cut], histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.savefig(f"{out_path}/nonResSamples_diphoton_mass_ggFHH_score_cuts.png")
    plt.clf()

    for cut in [0, 0.6, 0.9]:
        plt.hist(np.array(events.dijet_mass)[y[:, 3] > cut], bins=30, range=(100, 180), weights=rel_w[y[:, 3] > cut], histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.savefig(f"{out_path}/nonResSamples_dijet_mass_ggFHH_score_cuts.png")
    plt.clf()


    sample = ["Data_EraE", "Data_EraF", "Data_EraG", "DataC_2022", "DataD_2022"]
    #sample = ["GGJets"]
    path = "MLP_inputs_20250218_with_mjj_mass/individual_samples_data/"
    # Load scores
    y = []
    for s in sample:
        #y_preEE = np.load(f"{path}/individual_samples/preEE/{s}/y.npy")
        y_postEE = np.load(f"{path}/{s}/y.npy")
        #print(y_postEE.shape)
        y.append(y_postEE)
    y = np.concatenate(y)
    #print(y.shape)
    #print()

    # Load parquet files
    events = []
    for s in sample:
        #preEE = ak.from_parquet(f"{path}/individual_samples/preEE/{s}/events.parquet")
        postEE = ak.from_parquet(f"{path}/{s}/events.parquet")
        #print(len(postEE))
        events.append(postEE)
    events = ak.concatenate(events)
    #print(len(events))

    # plot di_photon and dijet mass for various ggFHH score cuts
    for cut in [0, 0.2, 0.7, 0.9, 0.95]:
        plt.hist(np.array(events.mass)[y[:, 3] > cut], bins=25, range=(100, 180), histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.savefig(f"{out_path}/data_mgg_ggFHH_score_cuts.png")
    plt.clf()

    for cut in [0, 0.2, 0.7, 0.9, 0.95]:
        plt.hist(np.array(events.nonRes_dijet_mass)[y[:, 3] > cut], bins=25, range=(100, 180), histtype='step', label=f"ggFHH score > {cut}", density=True)
    plt.legend()
    plt.savefig(f"{out_path}/data_mjj_ggFHH_score_cuts.png")
    plt.clf()
