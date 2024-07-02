import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep # this is a package that helps to plot in CMS style. mplhep = MatPlotLib for High Energy Physics
from pathlib import Path
import argparse
import json


def load_parquet_files(NTuples_path):

    # List of MC samples
    MC_samples = ["GluGluToHH", "ttHToGG", "GGJets"]
    selected_MC_samples = ["GJetPt40", "GJetPt20To40"]
    MC_array_dict = {"GJetPt20": []}

    for MC in selected_MC_samples:
        MC_data = ak.from_parquet(f'{NTuples_path}/{MC}/nominal')
        MC_array_dict["GJetPt20"].append(MC_data)
        print(f'INFO: Loaded data from {NTuples_path}/{MC}/nominal')

    for MC in MC_samples:
        if len(MC_array_dict)<20:
        # load the MC parquet files and add them to the MC_array_dict
            MC_array_dict[MC] = ak.from_parquet(f'{NTuples_path}/{MC}/nominal')
            print(f'INFO: Loaded and selected VBF enriched events from {NTuples_path}/{MC}/nominal')
    MC_array_dict["GJetPt20"] = ak.concatenate(MC_array_dict["GJetPt20"], axis=0)
    return MC_array_dict


def plot_variables(config, NTuples_path, outpath):

    # get data and MC arrays
    MC_arr = load_parquet_files(NTuples_path)
    print(MC_arr)

    # get the histograms and plot it
    for var in config.keys():
        var_config = config[var]
        plt.style.use(hep.style.CMS) # this is to set the style of the plotting to be of CMS style. You do not have to worry about it much for now

        binning = np.linspace(var_config["hist_range"][0], var_config["hist_range"][1], var_config["n_bins"] + 1)

        MC_hists = {}
        MC_edges = {}
        for MC in MC_arr.keys():
            #if MC_arr.keys() != "GJetPt40" and if MC_arr.keys() != "GJetPt20To40":
            weight = np.asarray(MC_arr[MC].weight) # get the weights for each event
            
            # check more about the np.histogram and its arguments
            if var_config["type"] == "quotient":
                # Extract arrays and calculate the quotient of the two variables
                variable1 = np.asarray(MC_arr[MC][var_config["var1"]])
                variable2 = np.asarray(MC_arr[MC][var_config["var2"]])
                quotient = variable1 / variable2
                data_to_plot = quotient
            elif var_config["type"] == "difference":
                variable1 = np.asarray(MC_arr[MC][var_config["var1"]])
                variable2 = np.asarray(MC_arr[MC][var_config["var2"]])
                difference = variable1-variable2
                data_to_plot = difference
            elif var_config["type"] == "chisquared":
                mw=80.3
                mt=173.5
                mjj = np.asarray(MC_arr[MC][var_config["var1"]])
                mbjj = np.asarray(MC_arr[MC][var_config["var2"]])
                chisquared = ((mw-mjj)/(0.1*mw))**2+((mt-mbjj)/(0.1*mt))**2
                data_to_plot = chisquared
            else:
                # Extract the single variable
                data_to_plot = np.asarray(MC_arr[MC][var_config["name"]])
            
            MC_hists[MC], MC_edges[MC] = np.histogram(data_to_plot, bins=binning, weights=weight, density=True)
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

# call the plotting function
samples_path = "/net/scratch_cms3a/kasaraguppe/public/HHbbgg_samples/"
plot_variables(json.load(open("variables.json")), samples_path, "plots")