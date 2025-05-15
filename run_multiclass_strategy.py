from data.prepare_inputs import PrepareInputs
import os
import argparse
import subprocess
import yaml

def prepare_inputs(args):

    config_path = args.config_path
    out_path = args.out_path

    # create the output directory
    os.makedirs(out_path, exist_ok=True)

    input_vars_path = f"{config_path}/input_variables.yaml"

    # Load the configuration yaml files
    with open(f"{config_path}/training_config.yaml", 'r') as f:
        training_config = yaml.safe_load(f)

    
    prep_inputs = PrepareInputs(input_var_json=input_vars_path,
                                sample_to_class = training_config["sample_to_class"],
                                class_num = training_config["class_num"],
                                classes = training_config["classes"],
                                sample_to_class_for_pred = training_config["sample_to_class_for_pred"],
                                outpath=out_path,
                                random_seed = training_config["random_seed"],)
    
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


def perform_training(args):

    out_path = args.out_path
    config_path = args.config_path

    # Load the configuration yaml files
    with open(f"{config_path}/training_config.yaml", 'r') as f:
        training_config = yaml.safe_load(f)

    do_random_search = training_config["do_random_search"]
    random_seed = training_config["random_seed"]

    # do random search
    if (do_random_search) and (args.do_random_search):
        num_random_search = training_config["num_random_search"]
        print('INFO: Performing random search')
        subprocess.run(f"python3 models/random_search.py --input_path {out_path} --n_trials {num_random_search}", shell=True)

    # do training for the best model
    if args.perform_training:
        # perform trainging
        print('INFO: Training the best model')
        subprocess.run(f"python3 models/training_utils.py --input_path {out_path} --random_seed {random_seed}", shell=True)

        # plot the training results
        print('INFO: Getting the results plots')
        subprocess.run(f"python3 models/mlp_plotter.py --input_path {out_path}", shell=True)

        # get permutaion importance
        print('INFO: Getting permutation importance')
        subprocess.run(f"python3 models/permutation_importance.py --input_path {out_path}", shell=True)

        # get the predictions
        print('INFO: Getting the predictions')
        subprocess.run(f"python3 models/get_prediction.py --model_folder {out_path}/after_random_search_best1/ --samples_path {out_path} --config_path {config_path}", shell=True)

        # get non resonant mass for different ggFHH score cuts
        print('INFO: Getting non resonant mass for different ggFHH score cuts')
        subprocess.run(f"python3 utils/test_cor_mass.py --input_path {out_path} --config_path {config_path}", shell=True)

    # get the predictions for data
    if args.get_data_mc_plots:
        print('INFO: Getting data-MC plots')
        subprocess.run(f"python3 utils/plotting_utils.py --base-path {out_path}", shell=True)

def perform_categorisation(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--samples_path', type=str, help='Path to the samples, Eg: /eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v2/Run3_2022')
    parser.add_argument('--out_path', type=str, help='Path to save the inputs')
    args = parser.parse_args()

    # prepare inputs
    prepare_inputs(args)

    # perform training
    perform_training(args)

    # perform categorisation
    perform_categorisation(args)