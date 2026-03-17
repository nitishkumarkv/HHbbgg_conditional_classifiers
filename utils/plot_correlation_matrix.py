import argparse
import math
from pathlib import Path
import sys

import awkward as ak
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.prepare_inputs import PrepareInputs


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_mhh_selection(training_config, mhh_bin):
    mhh_binning = training_config.get("mHH_binning")
    if mhh_binning is None:
        return None, None

    variable = mhh_binning.get("variable")
    edges = mhh_binning.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("mHH_binning.edges must be a list of numbers")

    boundaries = [0.0] + [float(x) for x in edges] + [math.inf]
    bins = []
    for i in range(len(boundaries) - 1):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        hi_name = "inf" if not math.isfinite(hi) else str(int(hi))
        bin_name = f"mHH_bin_{int(lo)}_to_{hi_name}"
        bins.append((i, lo, hi, bin_name))

    if mhh_bin is None:
        return variable, None

    found = None
    try:
        idx = int(mhh_bin)
        if 0 <= idx < len(bins):
            found = bins[idx]
    except ValueError:
        pass

    if found is None:
        for item in bins:
            i, lo, hi, name = item
            if name == mhh_bin or name.replace("mHH_bin_", "") == mhh_bin:
                found = item
                break

    if found is None:
        key = mhh_bin.replace("-", "_")
        for item in bins:
            _, lo, hi, _ = item
            simple = f"{int(lo)}_{'inf' if not math.isfinite(hi) else int(hi)}"
            if key == simple:
                found = item
                break

    if found is None:
        known_bins = [item[3] for item in bins]
        raise ValueError(f"Could not resolve --mhh_bin={mhh_bin}. Known bins: {known_bins}")

    _, lo, hi, _ = found
    return variable, (lo, hi)


def plot_correlation_matrices(args):
    config_path = Path(args.config_path).resolve()
    training_config = _load_yaml(config_path / "training_config.yaml")
    input_vars = _load_yaml(config_path / "input_variables.yaml")

    mhh_var, mhh_range = _resolve_mhh_selection(training_config, args.mhh_bin)

    prep = PrepareInputs(
        input_var_json=str(config_path / "input_variables.yaml"),
        training_info=training_config,
        outpath=str(Path(args.out_path).resolve()),
        mhh_var=mhh_var,
        mhh_range=mhh_range,
    )

    vars_for_training = input_vars[prep.model_type]["vars"]
    corr_mass_vars = [
        "mass",
        "nonRes_dijet_mass",
        "nonResReg_dijet_mass",
        "nonResReg_dijet_mass_DNNreg",
    ]
    vars_to_load = list(dict.fromkeys(vars_for_training + prep.extra_vars_train + corr_mass_vars))

    out_dir = Path(args.out_path).resolve() / "correlation_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    eras = [args.era] if args.era else training_config["samples_info"]["eras"]
    for era in eras:
        samples_cfg = training_config["samples_info"][era]
        sample_names = [args.sample] if args.sample else list(samples_cfg.keys())

        for sample in sample_names:
            if sample not in samples_cfg:
                raise ValueError(f"Sample {sample} not found in config for era {era}")

            parquet_path = Path(training_config["samples_info"]["samples_path"]) / samples_cfg[sample]
            events = ak.from_parquet(str(parquet_path), columns=vars_to_load)

            events = prep.preselection(events)
            events = prep.add_var(events, era)
            events = prep.get_relative_xsec_weight(events, sample, era)

            if prep.mhh_var is not None and prep.mhh_range is not None:
                events = prep._apply_mhh_filter(events)

            if len(events) == 0:
                print(f"WARNING: No events left for {sample} in {era}; skipping.")
                continue

            out_name = f"{sample}_{era}.pdf"
            if prep.mhh_var is not None and prep.mhh_range is not None:
                lo, hi = prep.mhh_range
                hi_label = "inf" if not math.isfinite(hi) else str(int(hi))
                out_name = f"{sample}_{era}_mHH_bin_{int(lo)}_to_{hi_label}.pdf"

            corr_out_path = out_dir / out_name
            print(f"INFO: Plotting correlation matrix for {sample} in {era} -> {corr_out_path}")
            prep.corr_with_mgg_mjj(events, vars_for_training, str(corr_out_path))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot the same mgg/mjj correlation matrix used in prepare_inputs.py without modifying input data."
    )
    parser.add_argument("--config_path", required=True, help="Path containing training_config.yaml and input_variables.yaml")
    parser.add_argument("--out_path", required=True, help="Directory where correlation_matrix/*.pdf will be written")
    parser.add_argument("--era", default=None, help="Optional single era to process")
    parser.add_argument("--sample", default=None, help="Optional single sample to process")
    parser.add_argument(
        "--mhh_bin",
        default=None,
        help='Optional mHH bin to filter, using the same syntax as run_multiclass_strategy.py: index, "mHH_bin_lo_to_hi", or "lo_hi".',
    )
    return parser


if __name__ == "__main__":
    plot_correlation_matrices(build_parser().parse_args())
