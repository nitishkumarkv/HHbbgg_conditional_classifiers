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
                                training_info = training_config,
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


def perform_training(args):

    out_path = args.out_path
    config_path = args.config_path

    # Load the configuration yaml files
    training_config_path = f"{config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    do_random_search = training_config["do_random_search"]

    # do random search
    if do_random_search:
        print('INFO: Performing random search')
        subprocess.run(f"python3 models/random_search.py --input_path {out_path} --training_config_path {training_config_path}", shell=True)        

    # perform trainging
    if args.train_best_model:
        print('INFO: Training the best model')
        subprocess.run(f"python3 models/training_utils.py --input_path {out_path} --training_config_path {training_config_path}", shell=True)

    # plot the training results
    if args.plot_training_results:
        print('INFO: Getting the results plots')
        subprocess.run(f"python3 models/mlp_plotter.py --input_path {out_path}", shell=True)
        
    # get permutaion importance
    if args.get_permutation_importance:
        print('INFO: Getting permutation importance')
        subprocess.run(f"python3 models/permutation_importance.py --input_path {out_path}", shell=True)
    
    # get the predictions
    if args.get_predictions:
        print('INFO: Getting the predictions')
        subprocess.run(f"python3 models/get_prediction.py --model_folder {out_path}/after_random_search_best1/ --samples_path {out_path} --config_path {config_path}", shell=True)

    # get non resonant mass for different ggFHH score cuts
    if args.test_mass_sculpting:
        print('INFO: Getting non resonant mass for different ggFHH score cuts')
        subprocess.run(f"python3 utils/test_cor_mass.py --input_path {out_path} --config_path {config_path}", shell=True)

    # get the predictions for data
    if args.get_data_mc_plots:
        print('INFO: Getting data-MC plots')
        subprocess.run(f"python3 utils/plotting_utils.py --base-path {out_path} --training_config_path {training_config_path}", shell=True)

def perform_categorisation(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform MLP based classification')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--out_path', type=str, help='Path to save the inputs')
    parser.add_argument('--prep_inputs_for_training', action='store_true', help='Prepare inputs for training')
    parser.add_argument('--prepare_inputs_pred_sim', action='store_true', help='Prepare inputs for prediction')
    parser.add_argument('--prepare_inputs_pred_data', action='store_true', help='Prepare inputs for prediction data')
    parser.add_argument('--train_best_model', action='store_true', help='Train the best model')
    parser.add_argument('--plot_training_results', action='store_true', help='Plot training results')
    parser.add_argument('--get_permutation_importance', action='store_true', help='Get permutation importance')
    parser.add_argument('--get_predictions', action='store_true', help='Get feature importance')
    parser.add_argument('--test_mass_sculpting', action='store_true', help='Test mass sculpting')
    parser.add_argument('--perform_training', action='store_true', help='Perform training')
    parser.add_argument('--get_data_mc_plots', action='store_true', help='Get data-MC plots')
    parser.add_argument('--perform_categorisation', action='store_true', help='Perform categorisation')
    parser.add_argument('--prepare_inputs', action='store_true', help='Prepare all inputs')
    parser.add_argument('--do_all', action='store_true', help='Perform all steps')
    args = parser.parse_args()

    if args.do_all:
        args.prepare_inputs = True
        args.perform_training = True
        args.perform_categorisation = True

    if args.prepare_inputs:
        args.prep_inputs_for_training = True
        args.prepare_inputs_pred_sim = True
        args.prepare_inputs_pred_data = True

    if args.perform_training:
        args.train_best_model = True
        args.plot_training_results = True
        args.get_permutation_importance = True
        args.get_predictions = True
        args.test_mass_sculpting = True
        args.get_data_mc_plots = True

    # prepare inputs
    prepare_inputs(args)

    # perform training
    perform_training(args)

    # perform categorisation
    perform_categorisation(args)