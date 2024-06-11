import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
import argparse

from load_samples import load_parquet

def load_parquet_files(NTuples_path):

    # load the data parquet files and select the events in VBF eneriched region
    data_array = select_events_VBF_enriched(f'{NTuples_path}/DataC_2022/nominal')
    print(f'INFO: Loaded and selected VBF enriched events from {NTuples_path}/DataC_2022/nominal')

    # List of MC samples
    MC_samples = ["ggh", "vbf", "vh", "tth"]
    MC_array_dict = {}

    for MC in MC_samples:
        # load the MC parquet files and select the events in VBF eneriched region
        MC_array_dict[MC] = select_events_VBF_enriched(f'{NTuples_path}/{MC}_M-125_preEE/nominal')
        print(f'INFO: Loaded and selected VBF enriched events from {NTuples_path}/{MC}_M-125_preEE/nominal')

    return data_array, MC_array_dict

def plot_variables(config_file, NTuples_path, outpath):


    # get data and MC arrays
    data_arr, MC_arr = load_parquet_files(NTuples_path)


    # get the histograms and plot it
    for var in vars_config.keys():
        plt.style.use(hep.style.CMS)

        config = vars_config[var]
        binning = np.linspace(config["hist_range"][0], config["hist_range"][1], config["n_bins"] + 1)
        
        #data_hist, data_edges = np.histogram(data_arr[config["name"]], bins=binning)
        #data_hist_sum = np.sum(data_hist)

        MC_hists = {}
        MC_edges = {}
        for MC in MC_arr.keys():
            weight = np.asarray(MC_arr[MC].weight)
            MC_hists[MC], MC_edges[MC] = np.histogram(MC_arr[MC][config["name"]], bins=binning, weights=weight)

            hep.histplot((MC_hists[MC], MC_edges[MC]), histtype='step', label=MC)
        
        hep.histplot((data_hist, data_edges), histtype='errorbar', yerr=np.sqrt(data_hist), label="Data", color="black")

        # set other plotting configs
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.yscale("log")
        plt.legend()
        plt.xlim(config['plot_range'])
        plt.ylim(bottom=0)
        hep.cms.label("Private work", data=True, year="2022", com=13.6)

        # save the histograms
        Path(outpath).mkdir(exist_ok=True)

        plt.savefig(f"{outpath}/{var}_plot.png")
        print(f"INFO: Histogram saved for {var} in {outpath}")
        plt.clf()

var_dict['diphoton_pt']['name']
var_dict = {
    "diphoton_pt": {
    "name": "pt",
    "xlabel": "$\\mathrm{p}_{T}^{\\gamma\\gamma}$ [GeV]",
    "ylabel": "Events per bin",
    "hist_range": [0, 300],
    "n_bins": 10,
    "plot_range": [0, 300]
    },
    "second_jet_pt": {
        "name": "second_jet_pt",
        "xlabel": "$p_{T}^{j_2}$ [GeV]",
        "ylabel": "Events per bin",
        "hist_range": [0, 250],
        "n_bins": 10,
        "plot_range": [0, 250]
    },
    "dijet_delta_phi": {
        "name": "dijet_delta_phi",
        "xlabel": "$|\\Delta\\phi_{j_1,j_2}|$",
        "ylabel": "Events per bin",
        "hist_range": [0, 3.142],
        "n_bins": 10,
        "plot_range": [0, 3.142]
    },
    "dijet_mass": {
        "name": "dijet_mass",
        "xlabel": "$m_{j_1j_2} [GeV]$",
        "ylabel": "Events per bin",
        "hist_range": [200, 1000],
        "n_bins": 10,
        "plot_range": [200, 1000]
    }
}


if __name__ == "__main__":
    # Set up argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Plotting script")
    parser.add_argument('config_file', type=str, help="Path to the configuration file")
    parser.add_argument('input_dir', type=str, help="Directory containing the input samples")
    parser.add_argument('output_dir', type=str, help="Directory to save the output plots")

    # Parse the command line arguments
    args = parser.parse_args()

    plot_variables(args.config_file, args.input_dir, args.output_dir)


