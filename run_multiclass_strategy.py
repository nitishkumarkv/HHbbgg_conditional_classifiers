from data.prepare_inputs import PrepareInputs
import os
import sys
import argparse
import subprocess
import yaml
import shutil
import datetime


class TeeLogger:
  def __init__(self, terminal, log_file):
    self.terminal = terminal
    self.log_file = log_file

  def write(self, message):
    self.terminal.write(message)
    self.log_file.write(message)
    self.flush()

  def flush(self):
    self.terminal.flush()
    self.log_file.flush()


def setup_logging(out_path, log_file=None):
  os.makedirs(os.path.join(out_path, "logs"), exist_ok=True)

  if log_file is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_path, "logs", f"run_{timestamp}.log")

  log_handle = open(log_file, "a", buffering=1)

  sys.stdout = TeeLogger(sys.stdout, log_handle)
  sys.stderr = TeeLogger(sys.stderr, log_handle)

  print(f"INFO: Logging to {log_file}")

  return log_file


def get_split_dirs(out_path, training_config):
  train_dir_name = training_config.get("train_split_dir_name", "Train_split")
  final_dir_name = training_config.get("final_split_dir_name", "Final_split")

  train_out_path = os.path.join(out_path, train_dir_name)
  final_out_path = os.path.join(out_path, final_dir_name)

  return train_out_path, final_out_path


def copy_train_scaling_to_final(train_out_path, final_out_path):
  files_to_copy = [
    "mean_std_dict.pkl",
    "input_vars.txt",
  ]

  for filename in files_to_copy:
    src = os.path.join(train_out_path, filename)
    dst = os.path.join(final_out_path, filename)

    if os.path.exists(src):
      os.makedirs(final_out_path, exist_ok=True)
      shutil.copy2(src, dst)
      print(f"INFO: Copied {src} -> {dst}")
    else:
      print(f"WARNING: Cannot copy missing file: {src}")


def make_prepare_inputs(args, training_config, split_select):
  config_path = args.config_path
  input_vars_path = f"{config_path}/input_variables.yaml"

  return PrepareInputs(
    input_var_json=input_vars_path,
    training_info=training_config,
    outpath=args.out_path,
    split_select=split_select,
  )


def run_prepare_training_for_one_split(args, training_config, split_select):
  print("\n" + "=" * 80)
  print(f"INFO: Preparing training arrays for split: {split_select}")
  print(f"INFO: Base output path: {args.out_path}")
  print("=" * 80 + "\n")

  prep_inputs = make_prepare_inputs(
    args=args,
    training_config=training_config,
    split_select=split_select,
  )

  prep_inputs.prep_inputs_for_training()


def run_prepare_prediction_for_one_split(args, training_config, split_select):
  print("\n" + "=" * 80)
  print(f"INFO: Preparing prediction inputs for split: {split_select}")
  print(f"INFO: Base output path: {args.out_path}")
  print("=" * 80 + "\n")

  prep_inputs = make_prepare_inputs(
    args=args,
    training_config=training_config,
    split_select=split_select,
  )

  if args.prepare_inputs_pred_sim:
    print(f"INFO: Preparing nominal MC prediction inputs for {split_select} split\n")
    prep_inputs.prep_inputs_for_prediction_sim()

  if args.prepare_inputs_pred_data:
    split_data = training_config.get("split_data", False)
    if split_select == "train" and not split_data:
      print(
        "INFO: Skipping data prediction inputs for train split because "
        "split_data is false; all real data will be prepared in Final_split.\n"
      )
    else:
      print(f"INFO: Preparing data prediction inputs for {split_select} split\n")
      prep_inputs.prep_inputs_for_prediction_data()

  if args.prepare_inputs_pred_sys:
    print(f"INFO: Preparing systematic prediction inputs for {split_select} split\n")
    prep_inputs.prep_inputs_for_prediction_sim_sys()


def prepare_inputs(args):
  config_path = args.config_path
  base_out_path = args.out_path

  os.makedirs(base_out_path, exist_ok=True)

  with open(f"{config_path}/training_config.yaml", "r") as f:
    training_config = yaml.safe_load(f)

  train_out_path, final_out_path = get_split_dirs(base_out_path, training_config)

  print("\nINFO: Split settings")
  print("INFO: MC nominal, MC systematics, and MC training samples are always split.")
  print("INFO: split_data controls real data only.")
  print(f"INFO: split_data for real data = {training_config.get('split_data', False)}")
  print(f"INFO: train_sample_fraction = {training_config.get('train_sample_fraction', 0.5)}")
  print(f"INFO: split_seed = {training_config.get('split_seed', 12345)}")
  print(f"INFO: Train split output = {train_out_path}")
  print(f"INFO: Final split output = {final_out_path}\n")

  if args.prep_inputs_for_training:
    run_prepare_training_for_one_split(
      args=args,
      training_config=training_config,
      split_select="train",
    )

  if args.prepare_inputs_pred_sim or args.prepare_inputs_pred_data or args.prepare_inputs_pred_sys:
    run_prepare_prediction_for_one_split(
      args=args,
      training_config=training_config,
      split_select="train",
    )

  if args.prep_inputs_for_training:
    run_prepare_training_for_one_split(
      args=args,
      training_config=training_config,
      split_select="final",
    )

  if args.prepare_inputs_pred_sim or args.prepare_inputs_pred_data or args.prepare_inputs_pred_sys:
    copy_train_scaling_to_final(train_out_path, final_out_path)

    run_prepare_prediction_for_one_split(
      args=args,
      training_config=training_config,
      split_select="final",
    )

    copy_train_scaling_to_final(train_out_path, final_out_path)


def run_command(command):
  if command.strip().startswith("python3 "):
    command = command.replace("python3 ", f"{sys.executable} -u ", 1)

  print("\n" + "=" * 80)
  print("INFO: Running command:")
  print(command)
  print("=" * 80 + "\n")

  env = os.environ.copy()
  env["PYTHONUNBUFFERED"] = "1"

  process = subprocess.Popen(
    command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
    executable="/bin/bash",
  )

  for line in process.stdout:
    print(line, end="")

  process.stdout.close()
  return_code = process.wait()

  if return_code != 0:
    raise subprocess.CalledProcessError(return_code, command)


def perform_training(args):
  base_out_path = args.out_path
  config_path = args.config_path

  training_config_path = f"{config_path}/training_config.yaml"

  with open(training_config_path, "r") as f:
    training_config = yaml.safe_load(f)

  train_out_path, final_out_path = get_split_dirs(base_out_path, training_config)

  train_path = train_out_path
  final_path = final_out_path

  do_random_search = training_config["do_random_search"]

  print("\nINFO: Training will always use Train_split")
  print("INFO: Final predictions will use Final_split with the Train_split-trained model")
  print(f"INFO: Train path = {train_path}")
  print(f"INFO: Final path = {final_path}\n")

  if do_random_search:
    print("INFO: Performing random search on Train_split")
    run_command(
      f"python3 models/random_search.py "
      f"--input_path {train_path} "
      f"--training_config_path {training_config_path}"
    )

  if args.train_best_model:
    print("INFO: Training the best model on Train_split")
    run_command(
      f"python3 models/training_utils.py "
      f"--input_path {train_path} "
      f"--training_config_path {training_config_path}"
    )

  if args.plot_training_results:
    print("INFO: Plotting training results for Train_split")
    run_command(
      f"python3 models/mlp_plotter.py "
      f"--input_path {train_path} "
      f"--config_path {config_path}/training_config.yaml"
    )

  if args.get_permutation_importance:
    print("INFO: Getting permutation importance for Train_split")
    run_command(
      f"python3 models/permutation_importance.py "
      f"--input_path {train_path}"
    )

  if args.get_predictions:
    print("INFO: Getting nominal predictions for Train_split")
    run_command(
      f"python3 models/get_prediction.py "
      f"--model_folder {train_path}/after_random_search_best1/ "
      f"--samples_path {train_path} "
      f"--config_path {config_path} "
      f"--get_pred_nominal "
      f"--skip_data"
    )

  if args.get_predictions_sys:
    print("INFO: Getting systematic predictions for Train_split")
    run_command(
      f"python3 models/get_prediction.py "
      f"--model_folder {train_path}/after_random_search_best1/ "
      f"--samples_path {train_path} "
      f"--config_path {config_path} "
      f"--get_pred_sys"
    )

  if args.get_predictions:
    print("INFO: Getting nominal predictions for Final_split using Train_split model")
    run_command(
      f"python3 models/get_prediction.py "
      f"--model_folder {train_path}/after_random_search_best1/ "
      f"--samples_path {final_path} "
      f"--config_path {config_path} "
      f"--get_pred_nominal"
    )

  if args.get_predictions_sys:
    print("INFO: Getting systematic predictions for Final_split using Train_split model")
    run_command(
      f"python3 models/get_prediction.py "
      f"--model_folder {train_path}/after_random_search_best1/ "
      f"--samples_path {final_path} "
      f"--config_path {config_path} "
      f"--get_pred_sys"
    )

  if args.test_mass_sculpting:
    print("INFO: Testing mass sculpting for Train_split")
    run_command(
      f"python3 utils/test_cor_mass.py "
      f"--input_path {train_path} "
      f"--config_path {config_path}"
    )

    print("INFO: Testing mass sculpting for Final_split")
    run_command(
      f"python3 utils/test_cor_mass.py "
      f"--input_path {final_path} "
      f"--config_path {config_path}"
    )

  if args.get_data_mc_plots:
    print("INFO: Getting data-MC plots for Train_split")
    run_command(
      f"python3 utils/plotting_utils.py "
      f"--base-path {train_path} "
      f"--training_config_path {training_config_path}"
    )

    print("INFO: Getting data-MC plots for Final_split")
    run_command(
      f"python3 utils/plotting_utils.py "
      f"--base-path {final_path} "
      f"--training_config_path {training_config_path}"
    )

  if args.get_score_shape_diff_kl:
    print("INFO: Getting score shape differences for Train_split")
    run_command(
      f"python3 utils/score_shape_diff_kl.py "
      f"--folder {train_path}/individual_samples/"
    )

    print("INFO: Getting score shape differences for Final_split")
    run_command(
      f"python3 utils/score_shape_diff_kl.py "
      f"--folder {final_path}/individual_samples/"
    )


def perform_categorisation(args):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Perform MLP based classification")

  parser.add_argument("--config_path", type=str, help="Path to the configuration files")
  parser.add_argument("--out_path", type=str, help="Base path to save the inputs")

  parser.add_argument("--prep_inputs_for_training", action="store_true", help="Prepare inputs for training")
  parser.add_argument("--prepare_inputs_pred_sim", action="store_true", help="Prepare inputs for prediction")
  parser.add_argument("--prepare_inputs_pred_data", action="store_true", help="Prepare inputs for prediction data")
  parser.add_argument("--prepare_inputs_pred_sys", action="store_true", help="Prepare inputs for prediction systematics")

  parser.add_argument("--train_best_model", action="store_true", help="Train the best model")
  parser.add_argument("--plot_training_results", action="store_true", help="Plot training results")
  parser.add_argument("--get_permutation_importance", action="store_true", help="Get permutation importance")
  parser.add_argument("--get_predictions", action="store_true", help="Get predictions for nominal MC and data")
  parser.add_argument("--get_predictions_sys", action="store_true", help="Get predictions for systematics")
  parser.add_argument("--test_mass_sculpting", action="store_true", help="Test mass sculpting")
  parser.add_argument("--get_data_mc_plots", action="store_true", help="Get data-MC plots")
  parser.add_argument("--get_score_shape_diff_kl", action="store_true", help="Get score shape differences using kl samples")

  parser.add_argument("--perform_training", action="store_true", help="Perform training")
  parser.add_argument("--perform_categorisation", action="store_true", help="Perform categorisation")
  parser.add_argument("--prepare_inputs", action="store_true", help="Prepare all inputs")
  parser.add_argument("--do_all", action="store_true", help="Perform all steps")

  parser.add_argument(
    "--log_file",
    type=str,
    default=None,
    help="Optional path to save full real-time logs. Default: <out_path>/logs/run_TIMESTAMP.log",
  )

  args = parser.parse_args()

  if args.out_path is None:
    raise ValueError("--out_path must be provided")

  setup_logging(args.out_path, args.log_file)

  print("\nINFO: Command:")
  print(" ".join(sys.argv))
  print("")

  if args.do_all:
    args.prepare_inputs = True
    args.perform_training = True
    args.perform_categorisation = True

  if args.prepare_inputs:
    args.prep_inputs_for_training = True
    args.prepare_inputs_pred_sim = True
    args.prepare_inputs_pred_data = True
    args.prepare_inputs_pred_sys = True

  if args.perform_training:
    args.train_best_model = True
    args.plot_training_results = True
    args.get_permutation_importance = True
    args.get_predictions = True
    args.get_predictions_sys = True
    args.test_mass_sculpting = True
    args.get_data_mc_plots = True
    args.get_score_shape_diff_kl = True

  if (
    args.prepare_inputs
    or args.prep_inputs_for_training
    or args.prepare_inputs_pred_sim
    or args.prepare_inputs_pred_data
    or args.prepare_inputs_pred_sys
  ):
    prepare_inputs(args)

  if (
    args.perform_training
    or args.train_best_model
    or args.plot_training_results
    or args.get_permutation_importance
    or args.get_predictions
    or args.get_predictions_sys
    or args.test_mass_sculpting
    or args.get_data_mc_plots
    or args.get_score_shape_diff_kl
  ):
    perform_training(args)

  if args.perform_categorisation:
    perform_categorisation(args)
