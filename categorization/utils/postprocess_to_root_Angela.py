#!/usr/bin/env python
import os
import argparse
import subprocess
import uproot
import awkward as ak
import numpy as np
import pandas as pd


def write_root_file(df, root_outfile, tree_name="Events"):
    """
    Write an awkward array (converted to a dictionary) into a ROOT file using uproot.
    """


    # convert the pandas dataframe to dictionary
    tree_dict = {field: df[field].to_numpy() for field in df.columns}

    with uproot.recreate(root_outfile) as f:
        f[tree_name] = tree_dict
    print(f"Wrote {root_outfile}")



def main():
    parser = argparse.ArgumentParser(
        description="Merge categorized parquet files (MC and Data) and convert them to ROOT files for final fits."
    )
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path to input samples and categorized events folder (e.g., MLP_inputs_20250218_with_mjj_mass)")
    args = parser.parse_args()

    folder_list = ["cat1", "cat2", "cat3", "singleH_enriched/th_bg_0_0p25827158051642274_th_signal_0p19800125200139082", "singleH_enriched/th_bg_0_0p7_th_signal_0p19800125200139082", "singleH_enriched/th_bg_0_0p8_th_signal_0p19800125200139082", "singleH_enriched/th_bg_0_0p9_th_signal_0p19800125200139082"]
    for folder in folder_list:
        #folder = args.base_path + f"/{f}"
        print(f"Processing folder {folder}")
        # convert each parquet file to root file
        for files in os.listdir(os.path.join(args.base_path, folder)):
            if files.endswith(".parquet"):
                parquet_file = os.path.join(args.base_path, folder, files)
                df = pd.read_parquet(parquet_file)
                write_root_file(df, parquet_file.replace(".parquet", ".root"), tree_name="DiphotonTree")
                print(f"Converted {parquet_file} to ROOT format.")


if __name__ == "__main__":
    main()
