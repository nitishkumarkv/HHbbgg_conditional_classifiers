import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep # this is a package that helps to plot in CMS style. mplhep = MatPlotLib for High Energy Physics
from pathlib import Path
import argparse


def load_parquet_files(NTuples_path):

    # List of MC samples
    MC_samples = ["GGJets", "GJetPt40", "GluGluHToGG", "VBFHToGG", "VHToGG", "ttHToGG", "GluGluToHH"]
    MC_array_dict = {}  # create empty dictionary in which you can add the different samples loaded
    # you can read more about the dictionaries in python

    for MC in MC_samples:
        # load the MC parquet files and add them to the MC_array_dict
        MC_array_dict[MC] = ak.from_parquet(f'{NTuples_path}/{MC}/nominal')
        print(f'INFO: Loaded and selected VBF enriched events from {NTuples_path}/{MC}/nominal')

    return MC_array_dict


def plot_variables(config, NTuples_path, outpath):

    # get data and MC arrays
    MC_arr = load_parquet_files(NTuples_path)

    # get the histograms and plot it
    for var in config.keys():
        var_config = config[var]
        plt.style.use(hep.style.CMS) # this is to set the style of the plotting to be of CMS style. You do not have to worry about it much for now

        binning = np.linspace(var_config["hist_range"][0], var_config["hist_range"][1], var_config["n_bins"] + 1)

        MC_hists = {}
        MC_edges = {}
        for MC in MC_arr.keys():
            weight = np.asarray(MC_arr[MC].weight) # get the weights for each event
            MC_hists[MC], MC_edges[MC] = np.histogram(MC_arr[MC][var_config["name"]], bins=binning, weights=weight, density=True)
            # check more about the np.histogram and its arguments

            hep.histplot((MC_hists[MC], MC_edges[MC]), histtype='step', label=MC)

        # set other plotting configs
        plt.xlabel(var_config['xlabel'])
        plt.ylabel(var_config['ylabel'])
        plt.yscale("log")
        plt.legend()
        plt.xlim(var_config['plot_range'])
        plt.ylim(bottom=0)
        hep.cms.label("Private work", data=False, year="2022", com=13.6)

        # save the histograms
        Path(outpath).mkdir(exist_ok=True) # this creates a directory in the mentioned 'outpath'. Similar to mkdir you use in terminal

        plt.savefig(f"{outpath}/{var}_plot.png") # saves the figure as png
        print(f"INFO: Histogram saved for {var} in {outpath}")
        plt.clf()

# define the variable to plot
var_dict = {
    "diphoton_pt": {
        "name": "pt", # this is the variable name in the samples
        "xlabel": "$\\mathrm{p}_{T}^{\\gamma\\gamma}$ [GeV]",
        "ylabel": "Events per bin",
        "hist_range": [0, 500],
        "n_bins": 40,
        "plot_range": [0, 500]
    },
    "diphoton_mass": {
        "name": "mass",
        "xlabel": "$m_{j_1j_2}$ [GeV]",
        "ylabel": "Events per bin",
        "hist_range": [100, 200],
        "n_bins": 40,
        "plot_range": [100, 200]
    }
}

# call the plotting function
samples_path = "/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/"
plot_variables(var_dict, samples_path, "plots")
