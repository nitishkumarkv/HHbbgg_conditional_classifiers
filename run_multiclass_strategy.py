import os
import argparse
import subprocess
import yaml


def prepare_inputs(args: argparse.Namespace):
    from data.prepare_inputs import PrepareInputs # pytorch heavy, import here to avoid unnecessary imports, pylint: disable=import-outside-toplevel

    config_path = args.config_path
    out_path = args.out_path

    # create the output directory
    os.makedirs(out_path, exist_ok=True)

    input_vars_path = f"{config_path}/input_variables.yaml"

    # Load the configuration yaml files
    with open(f"{config_path}/training_config.yaml", 'r', encoding='utf-8') as f:
        training_config = yaml.safe_load(f)

    prep_inputs = PrepareInputs(input_var_path=input_vars_path,
                                training_info=training_config,
                                outpath=out_path,)

    # prepare the inputs for training
    if args.prep_inputs_for_training:
        print('INFO: Preparing the inputs for training', '\n')
        prep_inputs.prep_inputs_for_training()

    # prepare the inputs for prediction
    if args.prepare_inputs_pred_sim:
        print('INFO: Preparing the inputs for prediction', '\n')
        prep_inputs.prep_inputs_for_prediction_sim()

    # prepare the inputs for prediction data
    if args.prepare_inputs_pred_data:
        print('INFO: Preparing the inputs for prediction data', '\n')
        prep_inputs.prep_inputs_for_prediction_data()

    # prepare the inputs for systematics
    if args.prepare_inputs_pred_sys:
        print('INFO: Preparing the inputs for prediction systematics', '\n')
        prep_inputs.prep_inputs_for_prediction_sim_sys()


def perform_training(args: argparse.Namespace):
    out_path = args.out_path
    config_path = args.config_path

    # Load the configuration yaml files
    training_config_path = f"{config_path}/training_config.yaml"
    job_config_path = f"{config_path}/job_config.yaml"
    with open(f"{training_config_path}", 'r', encoding="utf-8") as f:
        training_config = yaml.safe_load(f)

    do_random_search = training_config["do_random_search"]

    # do random search
    if do_random_search:
        print('INFO: Performing random search')
        subprocess.run(
            "python3 -m models.random_search "
            + f"--input_path {out_path} "
            + f"--training_config_path {training_config_path} ",
            shell=True, check=True
        )

    # perform training
    if args.train_best_model:
        print('INFO: Training the best model')
        subprocess.run(
            "python3 -m models.training_utils "
            + f"--input_path {out_path} "
            + f"--training_config_path {training_config_path} "
            + f"--job_config_path {job_config_path} "
            + f"--model_folder {args.model_folder} ", # e.g. after_random_search_best1, where to save best model, params.json, and y predictions
            shell=True, check=True
        )

    # plot the training results
    if args.plot_training_results:
        print('INFO: Getting the results plots')
        subprocess.run(
            "python3 -m models.mlp_plotter "
            + f"--input_path {out_path} "
            + f"--model_folder {args.model_folder} ", # e.g. after_random_search_best1, where to save best model, params.json, and y predictions
            shell=True, check=True
        )

    # get permutation importance
    if args.get_permutation_importance:
        print('INFO: Getting permutation importance')
        subprocess.run(
            "python3 models/permutation_importance.py "
            + f"--input_path {out_path} "
            + f"--training_folder {args.model_folder} ",
            shell=True, check=True
        )

    # get the predictions
    if args.get_predictions:
        print('INFO: Getting the predictions nominal')
        subprocess.run(
            "python3 -m models.get_prediction "
            + f"--model_folder {out_path}/{args.model_folder}/ "
            + f"--samples_path {out_path} "
            + f"--config_path {config_path} "
            + "--get_pred_nominal ",
            shell=True, check=True
        )

    # get the predictions for systematics
    if args.get_predictions_sys:
        print('INFO: Getting the predictions systematics')
        subprocess.run(
            "python3 -m models.get_prediction "
            + f"--model_folder {out_path}/{args.model_folder}/ "
            + f"--samples_path {out_path} "
            + f"--config_path {config_path} "
            + "--get_pred_sys ",
            shell=True, check=True
        )

    # get non resonant mass for different ggFHH score cuts
    if args.test_mass_sculpting:
        print('INFO: Getting non resonant mass for different ggFHH score cuts')
        subprocess.run(
            "python3 utils/test_cor_mass.py "
            + f"--input_path {out_path} "
            + f"--config_path {config_path} ",
            shell=True, check=True
        )

    # get the predictions for data
    if args.get_data_mc_plots:
        print('INFO: Getting data-MC plots')
        subprocess.run(
            "python3 utils/plotting_utils.py "
            + f"--base-path {out_path} "
            + f"--training_config_path {training_config_path} ",
            shell=True, check=True
        )

    # get score shapes for different kl samples
    if args.get_score_shape_diff_kl:
        print('INFO: Getting score shape differences')
        subprocess.run(
            "python3 utils/score_shape_diff_kl.py "
            + f"--folder {out_path}/individual_samples/ ",
            shell=True, check=True
        )


def perform_categorization(args):
    # Load categorization config yaml file
    categorization_config_path = f"{args.config_path}/categorization_config.yaml"
    with open(f"{categorization_config_path}", 'r', encoding="utf-8") as f:
        categorization_config = yaml.safe_load(f)

    n_categories        = categorization_config["n_categories"]
    n_runs              = categorization_config["n_runs"]
    optuna_folder       = categorization_config["optuna_folder"]
    n_trials            = categorization_config["n_trials"]
    sr_strategy         = categorization_config["SR_strategy"]
    gamma_strategy      = categorization_config["gamma_strategy"]
    side_band_threshold = categorization_config["side_band_threshold"]

    if args.perform_categorisation:
        print('INFO: Performing categorisation')
        subprocess.run(
            "python3 categorisation/bayesian_categorization.py "
            + f"--base_path {args.out_path} "
            + f"--n_categories {n_categories} "
            + f"--n_runs {n_runs} "
            + f"--optuna_folder {optuna_folder} "
            + f"--n_trials {n_trials} "
            + f"--SR_strategy {sr_strategy} "
            + f"--gamma_strategy {gamma_strategy} "
            + f"--side_band_threshold {side_band_threshold} ",
            shell=True, check=True
        )


def perform_mjj_sculpting_study(args: argparse.Namespace):
    from data.prepare_inputs import PrepareInputs # pytorch heavy, import here to avoid unnecessary imports, pylint: disable=import-outside-toplevel
    out_path = args.out_path
    config_path = args.config_path

    # Path to input variables
    input_vars_path = f"{config_path}/input_variables.yaml"

    # Load the configuration yaml files
    sculpting_study_config_path = f"{config_path}/sculpting_study_config.yaml"
    try:
        with open(f"{sculpting_study_config_path}", 'r', encoding="utf-8") as f:
            sculpting_study_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"ERROR: The specified sculpting_study_config.yaml file was not found at {sculpting_study_config_path}. "
            + "Please ensure the file exists and the path is correct."
        ) from e

    training_config_path = f"{config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r', encoding="utf-8") as f:
        training_config = yaml.safe_load(f)
    

    prep_inputs = PrepareInputs(input_var_path = input_vars_path,
                                training_info = training_config,
                                sculpting_study_info = sculpting_study_config,
                                outpath=out_path,)

    # prepare the inputs for mjj sculpting study
    if args.prepare_sculpting_study_inputs:
        print('INFO: Preparing the inputs for Mjj sculpting study')
        prep_inputs.prep_inputs_for_sculpting_study()

    # train the mjj predictor
    if args.train_mjj_predictor:
        print('INFO: Training the Mjj predictor for sculpting study')
        subprocess.run(
            "python3 -m models.mjj_training_utils "
            + f"--input_path {out_path} "
            + f"--training_config_path {training_config_path} "
            + f"--sculpting_study_config_path {sculpting_study_config_path} ",
            shell=True, check=True
        )

    # get permutaion importance
    if args.mjj_predictor_permutation_importance:
        print('INFO: Getting permutation importance')
        subprocess.run(
            "python3 -m models.permutation_importance "
            + f"--input_path {out_path} "
            + "--y_path sculpting_study/y_val.npy "
            + "--training_folder sculpting_study "
            + f"--sculpting_study_config_path {sculpting_study_config_path} ",
            shell=True, check=True
        )


def merge_samples(args: argparse.Namespace):
    out_path = args.out_path
    config_path = args.config_path
    training_config_path = os.path.join(config_path, "training_config.yaml")
    verbose_flag = "--verbose " if args.verbose else ""

    # merge samples after predictions to prepare for FinalFit
    if args.merge_samples:
        print('INFO: Merging samples after predictions to prepare for FinalFit')
        subprocess.run(
            "python3 utils/merge_samples.py "
            + f"--base_path {out_path} "
            + f"--config_path {training_config_path} "
            + verbose_flag,
            shell=True, check=True
        ) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform MLP based classification')
    # Main arguments
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--out_path', type=str, help='Path to save the inputs')
    parser.add_argument('--model_folder', type=str, default='after_random_search_best1', help='Folder containing the trained model')

    # Steps
    parser.add_argument('--prep_inputs_for_training', action='store_true', help='Prepare inputs for training')
    parser.add_argument('--prepare_inputs_pred_sim', action='store_true', help='Prepare inputs for prediction')
    parser.add_argument('--prepare_inputs_pred_data', action='store_true', help='Prepare inputs for prediction data')
    parser.add_argument('--prepare_inputs_pred_sys', action='store_true', help='Prepare inputs for prediction systematics')
    parser.add_argument('--train_best_model', action='store_true', help='Train the best model')
    parser.add_argument('--plot_training_results', action='store_true', help='Plot training results')
    parser.add_argument('--get_permutation_importance', action='store_true', help='Get permutation importance')
    parser.add_argument('--get_predictions', action='store_true', help='Get predictions for nominal MC and data')
    parser.add_argument('--get_predictions_sys', action='store_true', help='Get predictions for systematics')
    parser.add_argument('--test_mass_sculpting', action='store_true', help='Test mass sculpting')
    parser.add_argument('--get_data_mc_plots', action='store_true', help='Get data-MC plots')
    parser.add_argument('--perform_categorisation', action='store_true', help='Perform categorisation')
    parser.add_argument('--get_score_shape_diff_kl', action='store_true', help='Get score shape differences using kl samples')
    parser.add_argument('--merge_samples', action='store_true', help='Merge samples after predictions to prepare for FinalFit')

    # Mjj Sculpting Study
    parser.add_argument('--prepare_sculpting_study_inputs', action='store_true', help='Prepare inputs for Mjj sculpting study')
    parser.add_argument('--train_mjj_predictor', action='store_true', help='Train Mjj predictor for sculpting study')
    parser.add_argument('--mjj_predictor_permutation_importance', action='store_true', help='Get permutation importance for Mjj predictor')

    # Groups
    parser.add_argument('--perform_training', action='store_true', help='Perform training')
    parser.add_argument('--prepare_inputs', action='store_true', help='Prepare all inputs')
    parser.add_argument('--mjj_sculpting_study', action='store_true', help='Perform all steps for Mjj sculpting study')
    parser.add_argument('--do_all', action='store_true', help='Perform all steps')

    # Other
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    args = parser.parse_args()

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

    if args.mjj_sculpting_study:
        args.prepare_sculpting_study_inputs = True
        args.train_mjj_predictor = True
        args.mjj_predictor_permutation_importance = True

    # prepare inputs
    prepare_inputs(args)

    # perform training
    perform_training(args)

    # perform categorisation
    perform_categorization(args)

    # Mjj sculpting study
    perform_mjj_sculpting_study(args)

    # merge samples
    merge_samples(args)
