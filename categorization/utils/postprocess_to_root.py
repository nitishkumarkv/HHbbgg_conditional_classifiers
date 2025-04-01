#!/usr/bin/env python
import os
import argparse
import subprocess
import uproot
import awkward as ak
import numpy as np

def merge_parquet_files(file_list):
    """Read a list of parquet files with awkward and concatenate them."""
    arrays = []
    for fname in file_list:
        try:
            arrays.append(ak.from_parquet(fname, columns=["mass", "dijet_mass", "weight_tot"]))
        except Exception as e:
            print(f"Warning: Could not read {fname}: {e}")
    if len(arrays) == 0:
        return None
    return ak.concatenate(arrays)

def write_root_file(ak_array, root_outfile, tree_name="Events"):
    """
    Write an awkward array (converted to a dictionary) into a ROOT file using uproot.
    """
    # Convert each field to a numpy array
    tree_dict = {field: ak.to_numpy(ak_array[field]) for field in ak_array.fields}
    with uproot.recreate(root_outfile) as f:
        f[tree_name] = tree_dict
    print(f"Wrote {root_outfile}")

def merge_mc_category_to_root(base_path, cat_num, mc_samples):
    """
    For a given category (e.g. cat1), merge the parquet files from preEE and postEE
    for each MC sample and convert them to ROOT files.
    
    The output ROOT files will be stored in:
         <base_path>/max_categorization/cat<cat_num>/root/<sample>_merged.root
    """
    cat_dir = os.path.join(base_path, "optuna_categorization", f"cat{cat_num}")
    out_root_dir = os.path.join(cat_dir, "root")
    os.makedirs(out_root_dir, exist_ok=True)
    
    eras = ["preEE", "postEE"]
    for sample in mc_samples:
        parquet_files = []
        for era in eras:
            sample_dir = os.path.join(cat_dir, era, sample)
            parquet_file = os.path.join(sample_dir, "events.parquet")
            if os.path.exists(parquet_file):
                parquet_files.append(parquet_file)
            else:
                print(f"Warning: {parquet_file} not found.")
        if len(parquet_files)==0:
            print(f"No parquet files found for sample {sample} in category cat{cat_num}.")
            continue
        merged_array = merge_parquet_files(parquet_files)
        if merged_array is None:
            print(f"Skipping sample {sample} in category cat{cat_num} due to read error.")
            continue
        out_rootfile = os.path.join(out_root_dir, f"{sample}_merged.root")
        write_root_file(merged_array, out_rootfile)

def merge_data_category_to_root(base_path, cat_num, data_samples):
    """
    For a given category (e.g. cat1), merge the parquet files from all data samples
    and convert them to a single ROOT file.
    
    The output ROOT file will be stored in:
         <base_path>/max_categorization/cat<cat_num>/root/Data_merged.root
    """
    cat_dir = os.path.join(base_path, "optuna_categorization", f"cat{cat_num}")
    out_root_dir = os.path.join(cat_dir, "root")
    os.makedirs(out_root_dir, exist_ok=True)
    
    parquet_files = []
    for data_sample in data_samples:
        sample_dir = os.path.join(cat_dir, data_sample)
        parquet_file = os.path.join(sample_dir, "events.parquet")
        if os.path.exists(parquet_file):
            parquet_files.append(parquet_file)
        else:
            print(f"Warning: {parquet_file} not found for data sample {data_sample}.")
    if len(parquet_files)==0:
        print(f"No parquet files found for data in category cat{cat_num}.")
        return
    merged_array = merge_parquet_files(parquet_files)
    if merged_array is None:
        print(f"Skipping data in category cat{cat_num} due to read error.")
        return
    out_rootfile = os.path.join(out_root_dir, "Data_merged.root")
    write_root_file(merged_array, out_rootfile)

def main():
    parser = argparse.ArgumentParser(
        description="Merge categorized parquet files (MC and Data) and convert them to ROOT files for final fits."
    )
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path to input samples and categorized events folder (e.g., MLP_inputs_20250218_with_mjj_mass)")
    parser.add_argument("--n_categories", type=int, required=True,
                        help="Number of categories to process (e.g., 6)")
    args = parser.parse_args()
    
    # Define MC and data sample names (adjust if needed)
    mc_samples = [
        "GGJets",
        "GJetPt20To40",
        "GJetPt40",
        "TTGG",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        # Optionally exclude VBFHH sample if desired:
        "VBFHHto2B2G_CV_1_C2V_1_C3_1"
    ]
    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    
    # Loop over categories
    for cat in range(1, args.n_categories+1):
        print(f"Processing category cat{cat}")
        merge_mc_category_to_root(args.base_path, cat, mc_samples)
        merge_data_category_to_root(args.base_path, cat, data_samples)
    
    print("Merging and conversion to ROOT complete.")

if __name__ == "__main__":
    main()
