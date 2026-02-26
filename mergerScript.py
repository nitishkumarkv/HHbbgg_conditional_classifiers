import argparse
import numpy as np
import os
import yaml
import pandas as pd
import awkward as ak
import pyarrow.parquet as pq

ff_sampledict = {
    "GGJets": "GGJets",
    "DDQCDGJET": "DDQCDGJets",
    "TTGG": "TTGG",
    "TT": "TT",
    "TTG_10_100": "TTG_10_100",
    "TTG_100_200": "TTG_100_200",
    "TTG_200": "TTG_200",
    "ttHtoGG_M_125": "ttHToGG",
    "BBHto2G_M_125": "BBHToGG",
    "GluGluHToGG_M_125": "GluGluHToGG",
    "VBFHToGG_M_125": "VBFHToGG",
    "VHtoGG_M_125": "VHToGG",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGluToHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGluToHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGluToHH_kl-5p00_kt-1p00_c2-0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000": "VBFHH_CV-1p000_C2V-1p000_C3-1p000",
}

def load_samples(base_path, samples, classes, var_prefix, eras, no_syst_samples, data=False, syst="", save_all_columns=False, per_sample=False):
    """Load predictions and weights, scaling weights by luminosity.

    Returns a dict {(era, sample): DataFrame} if per_sample=True,
    or a single concatenated DataFrame otherwise.
    """
    # Find the first available parquet file to read the schema (weight column names)
    example_file = None
    for era in eras:
        for sample in samples:
            if sample in no_syst_samples and syst != "":
                continue
            path = (os.path.join(base_path, "individual_samples_data", era, sample)
                    if data else
                    os.path.join(base_path, "individual_samples", era, sample, syst))
            candidate = os.path.join(path, "events.parquet")
            if os.path.exists(candidate):
                example_file = candidate
                break
        if example_file:
            break
    if example_file is None:
        raise FileNotFoundError("Could not find any parquet file to read the schema from.")
    print(f"Using file {example_file} as an example to get the parquet schema")
    all_columns = pq.ParquetFile(example_file).schema.names
    weight_columns = [col for col in all_columns if 'weight' in col]
    dijet_mass_key = f"{var_prefix}_dijet_mass_DNNreg"
    score_keys = [f"{cls}_score" for cls in classes]

    #selected_columns = ["lumi", "event", "run", "mass", dijet_mass_key, "is_boosted", "y_proba"]
    selected_columns = ["lumi", "event", "run", "mass", dijet_mass_key]
    columns_to_load = selected_columns + weight_columns  # weight_columns always included

    sample_dfs = {}  # (era, sample) -> DataFrame

    for era in eras:
        print("###########")
        print(era)
        print("###########")
        print()
        for sample in samples:
            if sample in no_syst_samples and syst != "":
                continue

            if data:
                path = os.path.join(base_path, "individual_samples_data", era, sample)
            else:
                path = os.path.join(base_path, "individual_samples", era, sample, syst)
            y_path = os.path.join(path, 'y.npy')

            # Check if DNN prediction file exists
            if not os.path.exists(y_path):
                print(f"Missing y for {path}. Skipping.")
                continue

            events = ak.from_parquet(os.path.join(path, 'events.parquet'),
                                     columns=None if save_all_columns else columns_to_load)
            y = np.load(y_path)

            display_sample = sample
            if display_sample == "":
                display_sample = "Data"
            if display_sample in ff_sampledict:
                display_sample = ff_sampledict[display_sample]
            print(display_sample)
            print()

            if "2016" in era:
                year = 2016
            elif "2017" in era:
                year = 2017
            elif "2018" in era:
                year = 2018
            elif "22" in era or "EE" in era:
                year = 2022
            elif "23" in era or "BPix" in era:
                year = 2023
            elif "2024" in era:
                year = 2024
            else:
                raise ValueError(f"Unknown era: {era}")

            acc = {
                "sample": np.full(y.shape[0], display_sample),
                "year": np.full(y.shape[0], year),
            }

            if save_all_columns:
                for field in events.fields:
                    acc[field] = np.array(events[field])
            else:
                for col in selected_columns:
                    acc[col] = np.array(events[col])
                for weight in weight_columns:
                    if weight in events.fields:
                        acc[weight] = np.array(events[weight])
                    else:
                        acc[weight] = np.array(ak.ones_like(events['mass']))  # Default weight if not provided

            acc["score"] = list(y)
            for i, key in enumerate(score_keys):
                acc[key] = [row[i] for row in y]

            sample_dfs[(era, sample)] = pd.DataFrame(acc)

    if per_sample:
        return sample_dfs
    else:
        return pd.concat(list(sample_dfs.values()), ignore_index=True)

def save_per_sample(sample_dfs, base_path, syst="", data=False, merged=False):
    """Save scored samples under base_path/scored_samples.

    merged=False (default): sample_dfs is a dict {(era, sample): DataFrame};
        each is saved as .../scored_samples/[sim,data]/{era}/{sample}/[{syst}/]scored_events.parquet.
    merged=True: sample_dfs is a single DataFrame;
        saved as .../scored_samples/merged/[{syst}/]merged_scored_events.parquet.
    """
    if merged:
        subdir = "scored_samples/merged"
        parts = [base_path, subdir]
        if syst:
            parts.append(syst)
        out_dir = os.path.join(*parts)
        os.makedirs(out_dir, exist_ok=True)
        sample_dfs.to_parquet(os.path.join(out_dir, "merged_scored_events.parquet"), engine='pyarrow')
    else:
        subdir = "scored_samples/data" if data else "scored_samples/sim"
        for (era, sample), df in sample_dfs.items():
            parts = [base_path, subdir, era, sample]
            if syst:
                parts.append(syst)
            out_dir = os.path.join(*parts)
            os.makedirs(out_dir, exist_ok=True)
            df.to_parquet(os.path.join(out_dir, "scored_events.parquet"), engine='pyarrow')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path")
    parser.add_argument("config_path")
    parser.add_argument("--per-sample", action="store_true", dest="per_sample",
                        help="Save one parquet per (era, sample) under base_path/scored_samples "
                             "instead of a single merged file.")
    args = parser.parse_args()
    base_path = args.base_path
    config_path = args.config_path
    print(base_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    classes = config['classes']
    var_prefix = config['var_prefix']
    samples = list(config['sample_to_class'].keys())
    mc_eras = config['samples_info']['eras']
    data_eras = list(config['samples_info']['data'].keys())
    no_syst_samples = [s for s, c in config['sample_to_class'].items() if c == 'is_nonRes_bkg']
    systs = config.get('systematics') or []
    save_all_columns_sim_nominal = config.get('save_all_columns_sim_nominal', False)
    save_all_columns_data = config.get('save_all_columns_data', False)
    save_all_columns_sim_systematics = config.get('save_all_columns_sim_systematics', False)

    result_MC = load_samples(base_path, samples, classes, var_prefix, mc_eras, no_syst_samples,
                             save_all_columns=save_all_columns_sim_nominal, per_sample=args.per_sample)
    result_data = load_samples(base_path, [""], classes, var_prefix, data_eras, no_syst_samples,
                               data=True, save_all_columns=save_all_columns_data, per_sample=args.per_sample)

    if args.per_sample:
        save_per_sample(result_MC, base_path)
        save_per_sample(result_data, base_path, data=True)
    else:
        merged_samples = pd.concat([result_MC, result_data], ignore_index=True)
        save_per_sample(merged_samples, base_path, merged=True)

    for syst in systs:
        print(syst)
        result_MC = load_samples(base_path, samples, classes, var_prefix, mc_eras, no_syst_samples,
                                 syst=syst, save_all_columns=save_all_columns_sim_systematics, per_sample=args.per_sample)
        if args.per_sample:
            save_per_sample(result_MC, base_path, syst=syst)
        else:
            save_per_sample(result_MC, base_path, syst=syst, merged=True)
        print()
