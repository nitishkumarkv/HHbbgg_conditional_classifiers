import os
import sys
import argparse
import joblib
import yaml
import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mplhep
import awkward as ak
import pyarrow.parquet as pq
from typing import Any, Dict, List

################################################################################
#                             merge_samples.py                                 #
#                                                                              #
# This code was originally authored by Manos Vourliotis and later edited by    #
# Benjamin Lawrence-Sanderson. The code in this file is sourced from           #
# Manos Vourliotis's commit (82c085af) to the `postPreAppUpdates` branch of    #
# the `HHbbgg_conditional_classifiers` repository on July 31, 2025. All credit #
# for the original code goes to the original author(s). Subsequent edits to    #
# this file after the initial commit should be attributed to the respective    #
# commit authors, as one would expect.                                         #
#                                                                              #
################################################################################

# DEBUGGING:
# - Verify `save_all_columns_sim_systematics: True` if variables not found only in systematics events.py

BOOSTED_CAT = False

def sample_pq_to_ff(sample_name: str, config: dict) -> str:
    """Convert parquet sample name to finalfit sample name using config mapping.
    Example: ttHtoGG_M_125 -> ttHToGG
    """
    ff_sample_name_map: dict[str, str] = config["merge_samples"].get("finalfit_sample_name_map", {}) # pq: ff
    if len(ff_sample_name_map) == 0:
        warnings.warn("ff_sample_name_map is empty in config['merge_samples']. No sample name conversion will be applied.", UserWarning)
    if sample_name in ff_sample_name_map:
        return ff_sample_name_map[sample_name]
    return sample_name


def sample_ff_to_pq(sample_name: str, config: dict) -> str:
    """Convert finalfit sample name to parquet sample name using config mapping.
    Example: ttHToGG -> ttHtoGG_M_125
    """
    ff_sample_name_map: dict[str, str] = config["merge_samples"].get("finalfit_sample_name_map", {}) # pq: ff
    inv_map = {v: k for k, v in ff_sample_name_map.items()}
    if sample_name in inv_map:
        return inv_map[sample_name]
    return sample_name


def syst_pq_to_ff(syst_name: str, config: dict) -> str:
    """Convert parquet systematic name to finalfit systematic name using config mapping.
    Example: ScaleEB2G_IJazZ_up -> ScaleEB2GIJazZUp01sigma
    """
    ff_syst_name_map: dict[str, str] = config["merge_samples"].get("finalfit_syst_name_map", {}) # pq: ff
    if syst_name in ff_syst_name_map:
        return ff_syst_name_map[syst_name]
    return syst_name


def syst_ff_to_pq(syst_name: str, config: dict) -> str:
    """Convert finalfit systematic name to parquet systematic name using config mapping.
    Example: ScaleEB2GIJazZUp01sigma -> ScaleEB2G_IJazZ_up
    """
    ff_syst_name_map: dict[str, str] = config["merge_samples"].get("finalfit_syst_name_map", {}) # pq: ff
    inv_map = {v: k for k, v in ff_syst_name_map.items()}
    if syst_name in inv_map:
        return inv_map[syst_name]
    return syst_name


# def var_pq_to_ff(var_name: str, config: dict) -> str:
#     """Convert parquet variable name to finalfit variable name using config mapping.
#     Example: nonResReg_dijet_mass_DNNreg -> dijet_mass
#     """
#     ff_var_name_map: dict[str, str] = config["merge_samples"].get("finalfit_var_name_map", {}) # pq: ff
#     if var_name in ff_var_name_map:
#         return ff_var_name_map[var_name]
#     return var_name


# def var_ff_to_pq(var_name: str, config: dict) -> str:
#     """Convert finalfit variable name to parquet variable name using config mapping.
#     Example: dijet_mass -> nonResReg_dijet_mass_DNNreg
#     """
#     ff_var_name_map: dict[str, str] = config["merge_samples"].get("finalfit_var_name_map", {}) # pq: ff
#     inv_map = {v: k for k, v in ff_var_name_map.items()}
#     if var_name in inv_map:
#         return inv_map[var_name]
#     return var_name


class EventsWrapper():
    """Wrapper around awkward array to gracefully handle missing variable errors."""
    def __init__(self, events: ak.Array, path_for_warnings: str = ""):
        self.events = events
        self.path_for_warnings = path_for_warnings

    def __getitem__(self, name: str) -> Any:
        try:
            return self.events[name]
        except (KeyError, ak.errors.FieldNotFoundError):
            if self.path_for_warnings:
                warnings.warn(f"Attempted to access variable '{name}' that is not found in events at '{self.path_for_warnings}'. Returning None.")
            else:
                warnings.warn(f"Attempted to access variable '{name}' that is not found in events. Returning None.")            
            return None

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # Try to get the attribute from the wrapped events object
            try:
                return getattr(self.events, name)
            except AttributeError as e:
                if self.path_for_warnings:
                    raise AttributeError(f"Attempted to access attribute '{name}' that is not found in EventsWrapper or events at '{self.path_for_warnings}'.") from e
                else:
                    raise AttributeError(f"Attempted to access attribute '{name}' that is not found in EventsWrapper or events.") from e

class Samples():
    def __init__(self, config: dict, columns: dict[str, str] | None, weight_columns: dict[str, str], verbose=False):
        """Samples dictionary with built-in name handling.

        Steps:
            1) Fill sample dictionary with lists np.arrays indexed by FinalFit variable names.
                - Use parquet variable name in add() method; conversion to ff variable name done internally.
            2) Concatenate lists of arrays into single arrays, each indexed by ff variable names. 
                - Call concatenate() method once all samples have been added.
                - Special handling for score: break apa

        Args:
            config (dict): YAML "train config" dictionary.

        """
        if columns is None or len(columns) == 0:
            raise ValueError("Could not load columns to save from config['merge_samples']['columns_to_save']. Please check your configuration file.")

        self.samples            = {}
        self.config             = config
        self.columns:            dict[str, str] = columns
        self.weight_columns:     dict[str, str] = weight_columns
        self.all_save_columns:   dict[str, str] = {**self.columns, **self.weight_columns}
        # self.ff_sample_name_map: dict[str, str] = config["merge_samples"].get("ff_sample_name_map", {}) # pq: ff
        # self.ff_syst_name_map:   dict[str, str] = config["merge_samples"].get("finalfit_syst_name_map", {}) # pq: ff
        # self.ff_var_name_map:    dict[str, str] = config["merge_samples"].get("finalfit_var_name_map", {}) # pq: ff
        self.dijet_mass_key:     str            = config["merge_samples"].get("dijet_mass_key", "nonResReg_dijet_mass_DNNreg")
        self.score_key:          str            = config["merge_samples"].get("score_key", "score")
        self.score_idx_name_map: dict[int, str] = config["merge_samples"].get("score_idx_name_map", {}) # index: class name
        self.verbose:            bool           = verbose


    def add(self, var_name: str, var_data: np.ndarray):
        """var_name is from multiclass parquets (ff name)
        
        OLD:
        ff_var_name is the var name after mapping through ff name map
        """
        if var_data is None:
            # Missing variable, skip adding
            return
        if var_name not in self.all_save_columns.values(): # and self.config["merge_samples"]["finalfit_var_name_map"].get(var_name, var_name) not in self.all_save_columns:
            # Failure mode: FF name is name of different variable not in columns pq names
            return
        # Sample dict uses ff var names
        inv_map = {v: k for k, v in self.all_save_columns.items()} # ff: pq
        pq_var_name = inv_map.get(var_name, var_name) # get pq name
        # print(f"[DEBUG] Adding variable: {pq_var_name} as {var_name}")
        # print(f"[DEBUG] self.samples keys before adding: {list(self.samples.keys())}")
        if var_name not in self.samples:
            self.samples[var_name] = []
        self.samples[var_name].append(var_data)
        # print(f"[DEBUG] self.samples keys after adding: {list(self.samples.keys())}")

    def concatenate(self):
        """Concatenate samples in self.samples into single arrays."""
        samples_keys = copy.deepcopy(list(self.samples.keys())) # keep original keys to iterate over
        for ffvar in samples_keys:
            if self.verbose:
                print(f"[DEBUG] Concatenating variable: {ffvar}")
            data_list = self.samples[ffvar]
            if ffvar == self.score_key:
                # Special handling for score: list of arrays of shape (N, num_classes)
                if self.verbose:
                    print("[DEBUG] Concatenating score variable with special handling.")
                all_scores = np.concatenate(data_list, axis=0)
                self.samples[ffvar] = [row for row in all_scores]
                # Break into separate arrays per class
                num_classes = all_scores.shape[1]
                for class_idx in range(num_classes):
                    class_name = self.score_idx_name_map.get(class_idx, f"class_{class_idx}")
                    if self.verbose:
                        print(f"[DEBUG] Extracting class {class_idx} as {class_name}")
                    if class_name not in self.samples:
                        self.samples[class_name] = []
                    self.samples[class_name] = all_scores[:, class_idx] # [row[class_idx] for row in all_scores]
                continue
            try:
                self.samples[ffvar] = np.concatenate(data_list, axis=0)
            except ValueError as e:
                print(f"[ERROR] Failed to concatenate variable '{ffvar}': {e}")
                print(f"[DEBUG] data_list shapes: {[arr.shape if isinstance(arr, np.ndarray) else 'N/A' for arr in data_list]}")
                raise e


def load_weight_columns(all_columns: List[str], config: dict) -> Dict[str, str]:
    """Identify weight columns from parquet columns and apply mapping {parquet_name: finalfit_name}."""
    weight_columns = {col: col for col in all_columns if 'weight' in col}
    columns_to_save: Dict[str, str] = config["merge_samples"].get("columns_to_save", {})
    for col in list(weight_columns.keys()):
        if col in columns_to_save:
            weight_columns[col] = columns_to_save[col] # pq: ff
    return weight_columns


def load_samples(base_path, sample_list, config, data=False, syst="", verbose=False) -> pd.DataFrame:
    """Load predictions and weights, scaling weights by luminosity."""

    events_file_name = 'events_boostedCat.parquet' if BOOSTED_CAT else 'events.parquet'
    # Example MC file to get the weight columns
    parquet_file = pq.ParquetFile(base_path+"/individual_samples/preEE/ttHtoGG_M_125/"+syst+"/"+events_file_name)
    all_columns = parquet_file.schema.names
    dijet_mass_key = config["merge_samples"].get("dijet_mass_key", "nonResReg_dijet_mass_DNNreg")
    if dijet_mass_key not in all_columns:
        raise ValueError(
            f"dijet_mass_key '{dijet_mass_key}' not found in parquet columns. "
            + "You requested a mjj variable not present in the parquet files. "
            + f"Available columns are: {all_columns}"
        )

    # columns = [col for col in config["merge_samples"]["save_columns"]]
    weight_columns_dict: dict[str, str] = load_weight_columns(all_columns, config) # pq: ff
    columns_dict: dict[str, str] = config["merge_samples"].get("columns_to_save", {}) # pq: ff

    samples = Samples(config, columns=columns_dict, weight_columns=weight_columns_dict, verbose=verbose)

    # eras = ["preEE", "postEE", "preBPix", "postBPix"]
    eras: list[str] = config["samples_info"].get("eras", None)
    if eras is None:
        warnings.warn("No eras specified in config['samples_info']['eras']. Using default eras.")
        eras = ["preEE", "postEE", "preBPix", "postBPix"]
    if data:
        eras_dict: dict[str, str] | None = config["samples_info"].get("data", None)
        if eras_dict is None:
            warnings.warn("No data eras specified in config['samples_info']['data']. Using default eras.")
            eras = ["2022_EraC","2022_EraD","2022_EraE","2022_EraF","2022_EraG","2023_EraC","2023_EraD"]
        else:
            eras = list(eras_dict.keys())

    max_pq_sample_chars = max([len(sample) for sample in sample_list])

    for era in eras:
        print("\n###########")
        print(era)
        print("###########\n")
        for sample in sample_list:
            if (sample in ["GGJets", "DDQCDGJET", "TTGG", "TT", "TTG_10_100", "TTG_100_200", "TTG_200"]) and (syst != ""):
                continue
            
            if era != "postEE":
              if ("TTG_" in sample) or (sample == "TT"):
                continue
            if data:
                path = os.path.join(base_path, "individual_samples_data", era, sample)
            else:
                path = os.path.join(base_path, "individual_samples", era, sample, syst)
            y_path = os.path.join(path, 'y.npy')
            w_path = os.path.join(path, 'rel_w.npy')
            parquet_file = pq.ParquetFile(os.path.join(path, events_file_name))
            # print(parquet_file.schema.names)
            parquet_path = os.path.join(path, events_file_name)
            if verbose:
                print(f"[DEBUG] X path: {parquet_path}")
                print(f"[DEBUG] y path: {y_path}")
            events = ak.from_parquet(parquet_path, columns=list(columns_dict.keys())+list(weight_columns_dict.keys()))  # Load events
            events = EventsWrapper(events)

            # Check if files exist
            if not (os.path.exists(y_path)):
                print(f"Missing y for {path}. Skipping.")
                continue
            y = np.load(y_path)

            # Loop over eras, samples
            # samples_input = { 
            #       "sample": [ np.array(), np.array(), np.array(), ... ]
            # }
        
            samples.add("score", y) # samples_input["score"].append(y)
            samples.add("lumi", np.array(events['lumi'])) # samples_input["lumi"].append(np.array(events['lumi']))
            samples.add("event", np.array(events["event"])) # samples_input["event"].append(np.array(events['event']))
            samples.add("run", np.array(events['run'])) # samples_input["run"].append(np.array(events['run']))
            #samples_input["nonResReg_lead_bjet_hFlav"].append(np.array(events['nonResReg_lead_bjet_hFlav']))
            #samples_input["nonResReg_sublead_bjet_hFlav"].append(np.array(events['nonResReg_sublead_bjet_hFlav']))
            samples.add("mass", np.array(events['mass'])) # samples_input["mass"].append(np.array(events['mass']))
            samples.add("dijet_mass", np.array(events[dijet_mass_key])) # samples_input["dijet_mass"].append(np.array(events[dijet_mass_key]))

            if sample == "":
                sample = "Data"
            # if sample in ff_sampledict.keys():
            #     sample = ff_sampledict[sample]
            print(f"{sample:<{max_pq_sample_chars}} -> {sample_pq_to_ff(sample, config)}")
            # sample -> pq sample
            samples.add("sample", np.full(y.shape[0], sample_pq_to_ff(sample, config))) # samples_input["sample"].append(np.full(y.shape[0], sample))

            if "22" in era or "EE" in era:
                year = 2022
            elif "23" in era or "BPix" in era:
                year = 2023
            else:
                raise ValueError(f"Unknown era: {era}")
            samples.add("year", np.full(y.shape[0], year)) # samples_input["year"].append(np.full(y.shape[0], year))
            samples.add("y_proba", np.array(events["y_proba"])) # samples_input["y_proba"].append(np.array(events['y_proba'])) 
            samples.add("is_boosted", np.array(events["is_boosted"])) # samples_input["is_boosted"].append(np.array(events["is_boosted"]))

            for weight in weight_columns_dict.keys():
                if weight in events.fields:
                    samples.add(weight, np.array(events[weight])) # samples_input[weight].append(np.array(events[weight]))
                else:
                    # Default weight if not provided
                    samples.add(weight, np.array(ak.ones_like(events['mass']))) # samples_input[weight].append(np.array(ak.ones_like(events['mass'])))

    # Concatenate all data
    samples.concatenate()
    return pd.DataFrame(samples.samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge samples")
    parser.add_argument("--base_path", type=str, help="Base path to samples")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = parser.parse_args()
    base_path = args.base_path
    print(f"[INFO] Merging predictions from base path: {base_path}")

    with open(args.config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        print(f"[INFO] Loaded configuration from: {args.config_path}")

    # if config["merge_samples"]["dijet_mass_key"] not in config["merge_samples"]["finalfit_var_name_map"].keys():
    #     warnings.warn(
    #         f"dijet_mass_key '{config['merge_samples']['dijet_mass_key']}' not found in finalfit_var_name_map. "
    #         + "Make sure this is intentional. The 'dijet_mass_key' selects the correct variable to pull from every events.parquet file. "
    #         + "The 'finalfit_var_name_map' is used to rename variables for FinalFit compatibility. "
    #         + "The dijet_mass_key must be a key in finalfit_var_name_map to ensure proper renaming, if you intend to rename it (typically to 'dijet_mass')."
    #     )
    if config["merge_samples"]["dijet_mass_key"] not in config["merge_samples"]["columns_to_save"].keys():
        warnings.warn(
            f"dijet_mass_key '{config['merge_samples']['dijet_mass_key']}' not found in columns_to_save. "
            + "Make sure this is intentional. The 'dijet_mass_key' selects the correct variable to pull from every events.parquet file. "
            + "If it is not included in 'columns_to_save', it will not be saved in the merged samples output."
        )

    samples = config["merge_samples"]["samples"]
    systs = config["merge_samples"].get("systs", [""])

    merged_samples_MC = load_samples(base_path, samples, config, verbose=args.verbose)
    merged_samples_data = load_samples(base_path, [""] , config, data=True, verbose=args.verbose)

    out_paths = []
    merged_samples_path = os.path.join(base_path, "merged", "merged_samples.parquet")
    os.makedirs(os.path.dirname(merged_samples_path), exist_ok=True)
    merged_samples = pd.concat([merged_samples_MC, merged_samples_data], ignore_index=True)
    merged_samples.to_parquet(merged_samples_path, engine='pyarrow')
    out_paths.append(merged_samples_path)


    if systs is not None and len(systs) > 1:
        for syst in systs:
            print(f"\n-+-+-+-+-+-+- Systematic: {syst} -+-+-+-+-+-+-\n")
            ffsyst = config["merge_samples"].get("finalfit_syst_name_map", {}).get(syst, syst) # finalfit compatibility
            if ffsyst != syst:
                print(f"[INFO] Renaming: '{syst}' -> '{ffsyst}'")
            merged_samples_path = os.path.join(base_path, "merged", f"merged_samples_{ffsyst}.parquet")
            merged_samples_MC = load_samples(base_path, samples, config, syst=syst, verbose=args.verbose)
            merged_samples_MC.to_parquet(merged_samples_path, engine='pyarrow')
            out_paths.append(merged_samples_path)
            print()

    print("[INFO] Merged samples saved to the following paths:")
    for path in out_paths:
        print(f" - {path}")
