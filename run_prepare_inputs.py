import argparse
import math
import yaml
from pathlib import Path

from data.prepare_inputs import PrepareInputs


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset inputs for HHbbgg")
    parser.add_argument("--config_path", required=True, type=str, help="Path to training configuration directory")
    parser.add_argument("--out_path", required=True, type=str, help="Output path for prepared inputs")
    parser.add_argument("--prep_inputs_for_training", action="store_true", help="Prepare inputs for training")
    parser.add_argument("--prepare_inputs_pred_sim", action="store_true", help="Prepare inputs for prediction")
    parser.add_argument("--prepare_inputs_pred_data", action="store_true", help="Prepare inputs for prediction data")
    parser.add_argument("--prepare_inputs_pred_sys", action="store_true", help="Prepare inputs for prediction systematics")
    parser.add_argument("--max_input_files", type=int, default=None, help="Maximum number of parquet input files to process")
    parser.add_argument("--max_rows_per_file", type=int, default=None, help="Optional maximum rows to read per parquet file")
    parser.add_argument("--mhh_var", type=str, default=None, help="Optional mHH binning variable (e.g. mHH)")
    parser.add_argument("--mhh_min", type=float, default=None, help="Optional mHH minimum")
    parser.add_argument("--mhh_max", type=float, default=None, help="Optional mHH maximum")
    return parser.parse_args()


def _selected_modes(args) -> bool:
    return (
        args.prep_inputs_for_training
        or args.prepare_inputs_pred_sim
        or args.prepare_inputs_pred_data
        or args.prepare_inputs_pred_sys
    )


def _infer_mhh_range(args):
    if args.mhh_var is None:
        return None

    lo = -math.inf if args.mhh_min is None else args.mhh_min
    hi = math.inf if args.mhh_max is None else args.mhh_max
    return (lo, hi)


def main():
    args = parse_args()

    if not _selected_modes(args):
        raise SystemExit("No prepare mode selected. Use one of --prep_inputs_for_training, "
                         "--prepare_inputs_pred_sim, --prepare_inputs_pred_data, --prepare_inputs_pred_sys.")

    config_path = Path(args.config_path)
    out_path = args.out_path
    training_config_path = config_path / "training_config.yaml"
    input_vars_path = config_path / "input_variables.yaml"

    with open(training_config_path, "r", encoding="utf-8") as handle:
        training_info = yaml.safe_load(handle)

    prep_inputs = PrepareInputs(
        input_var_json=str(input_vars_path),
        training_info=training_info,
        outpath=out_path,
        mhh_var=args.mhh_var,
        mhh_range=_infer_mhh_range(args),
        max_input_files=args.max_input_files,
        max_rows_per_file=args.max_rows_per_file,
    )

    if args.prep_inputs_for_training:
        print("INFO: Preparing the inputs for training", "\n")
        prep_inputs.prep_inputs_for_training()
    if args.prepare_inputs_pred_sim:
        print("INFO: Preparing the inputs for prediction", "\n")
        prep_inputs.prep_inputs_for_prediction_sim()
    if args.prepare_inputs_pred_data:
        print("INFO: Preparing the inputs for prediction data", "\n")
        prep_inputs.prep_inputs_for_prediction_data()
    if args.prepare_inputs_pred_sys:
        print("INFO: Preparing the inputs for prediction systematics", "\n")
        prep_inputs.prep_inputs_for_prediction_sim_sys()


if __name__ == "__main__":
    main()
