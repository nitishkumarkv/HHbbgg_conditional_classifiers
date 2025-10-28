import os
import json
import argparse
import yaml
import pandas as pd


# i) read config and cat json
# ii) build exclusive criteria for each cat
#   1) Find number of cats and their boundaries
#   2) create inclusive cat criteria for each cat
#   3) assemble exclusive criteria by subtracting lower cats
# iii) write out new cat json with exclusive criteria

# Something like this:
#  _____________________________________
# | 1  |     |    |                     |
# |____|     |    |                     |
# |      2   |    |                     |
# |__________|    |                     |
# |               |                     |
# |       3       |                     |
# |_______________|                     |
# |                                     |
# |            boring stuff here        |
# |                                     |
# |_____________________________________|

# cat1_incl, cat2_incl, cat3_incl
# cat1_excl = cat1_incl
# cat2_excl = cat2_incl & ~cat1_incl
# cat3_excl = cat3_incl & ~cat1_incl & ~cat2_incl
# pattern continues for N cats


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _get_ff_name(col_name: str, config: dict) -> str:
    # Convert column name to finalfit style name
    name_map: dict = config["class_name_map"]["th_name_to_ff_name"]
    if col_name not in name_map:
        raise KeyError(f"Column name '{col_name}' not found in name map: {name_map}")
    return name_map[col_name]


def _load_cat_json(cat_json_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Example content of categorisation JSON file:
    [
        {
            "th_signal": 0.9542792633997237,
            "th_bg_0": 0.5848056744652141,
            "th_bg_1": 0.6244597191583707,
            "th_bg_2": 0.004685322034989716
        },
        {
            "th_signal": 0.8195304496094379,
            "th_bg_0": 0.0020518538431188353,
            "th_bg_1": 0.8701522299321695,
            "th_bg_2": 0.2234820142550845
        },
        {
            "th_signal": 0.08172076187323359,
            "th_bg_0": 0.005977478864069203,
            "th_bg_1": 0.45886373886030196,
            "th_bg_2": 0.28056295762565625
        },
        {
            "th_signal": 0.26566255445227366,
            "th_bg_0": 0.014261800623968576,
            "th_bg_1": 0.953251797470303,
            "th_bg_2": 0.656007940167993
        }
    ]

    Returns as a pandas DataFrame like:
         th_signal    th_bg_0    th_bg_1    th_bg_2
    0     0.954279   0.584806   0.624460   0.004685
    1     0.819530   0.002052   0.870152   0.223482
    2     0.081721   0.005977   0.458864   0.280563
    3     0.265663   0.014262   0.953252   0.656008
    """
    with open(cat_json_path, 'r', encoding="utf-8") as f:
        cat_data = json.load(f)
    cat_df = pd.DataFrame(cat_data)

    if verbose:
        print(f"Found {len(cat_df)} categories in categorisation JSON file: {cat_json_path}")
        print(f"Categories:\n{cat_df}")
    return cat_df


def _write_exclusive(excl_dict: dict[str, str], output_path: str, write: bool = True):
    out = json.dumps(excl_dict, ensure_ascii=True, indent=4, sort_keys=False, separators=None, allow_nan=True)
    if write:
        with open(output_path, 'w', encoding="utf-8") as f:
            f.write(out)
    # with open(output_path, 'w', encoding="utf-8") as f:
    #     json.dump(config, f, indent=4)
    return out


def _build_inclusive_criteria(cat_df: pd.DataFrame, config: dict) -> dict[str, str]:
    inclusive_criteria: dict[str, str] = {}
    n_cats = len(cat_df)
    for i in range(n_cats):
        cat_name = f"cat{i+1}"
        phrases: list[str] = []
        for col in cat_df.columns:
            threshold: float = cat_df.iloc[i][col]
            col_ff: str = _get_ff_name(col, config)
            ineq: str = config["class_criteria_inequality"][col]
            prec = config.get("json_float_precision", -1)
            if prec >= 0:
                threshold = round(threshold, prec)
            phrase = f"{col_ff} {ineq} {threshold}" # like "ggHH_score > 0.91005"
            phrases.append(phrase)
        inclusive_criteria[cat_name] = " & ".join(phrases) # like "ggHH_score > 0.91 & nonRes_score < 0.06 & ttH_score < 0.61 & singleH_score < 0.002" (vals abbreviated)
    return inclusive_criteria


def _build_exclusive_criteria(inclusive_criteria: dict[str, str], global_phrases: list[str]) -> dict[str, str]:
    NEGATION_OP: str = "~" # could also use "not"
    exclusive_criteria: dict[str, str] = {}
    n_cats: int = len(inclusive_criteria)
    for i in range(n_cats):
        cat_name = f"cat{i+1}"
        incl_crit: str = inclusive_criteria[cat_name]
        if i == 0:
            excl_crit = f"({incl_crit})"
            for phrase in global_phrases:
                excl_crit += f" & {phrase}"
        else:
            prev_incls: list[str] = [inclusive_criteria[f"cat{j+1}"] for j in range(i)]
            neg_prev_incls: list[str] = [f"{NEGATION_OP}({crit})" for crit in prev_incls] # negate previous inclusive criteria
            excl_crit: str = f"({incl_crit}) & " + " & ".join(neg_prev_incls)
            for phrase in global_phrases:
                excl_crit += f" & {phrase}"
        exclusive_criteria[cat_name] = excl_crit
    return exclusive_criteria


def convert(args: argparse.Namespace):
    # Main function
    config = _load_config(args.config)
    in_path = os.path.join(args.input_path, "optuna_categorization/best_cut_params.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Categorization JSON file not found at {in_path}")
    cat_df = _load_cat_json(in_path, verbose=args.verbose)

    # Build inclusive criteria
    inclusive_criteria: dict[str, str] = _build_inclusive_criteria(cat_df, config)
    if args.verbose:
        print("Inclusive criteria per category:")
        for cat, crit in inclusive_criteria.items():
            print(f"  {cat}: {crit}")
    
    # Build exclusive criteria
    global_phrases: list[str] = config.get("global_criteria_phrases", [])
    exclusive_criteria: dict[str, str] = _build_exclusive_criteria(inclusive_criteria, global_phrases)
    if args.verbose:
        print("Exclusive criteria per category:")
        for cat, crit in exclusive_criteria.items():
            print(f"  {cat}: {crit}")
    out_path = os.path.join(args.input_path, "optuna_categorization", args.out_file_name)
    out = _write_exclusive(exclusive_criteria, out_path)
    print("Exclusive categories JSON content:")
    print(out)
    print()
    print(f"Wrote exclusive categories JSON to: {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert categorisation JSON to exclusive categories.")
    parser.add_argument("--input_path", type=str, required=True, help="Multiclass base input path (e.g. Version_20250524_MVAID_forPreApp)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config JSON file.")
    parser.add_argument("--out_file_name", type=str, default="categories_exclusive.json", help="Output file name for exclusive categories JSON.")
    parser.add_argument("--dry_run", action="store_true", help="Don't write output file, just print info.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    convert(args)
