from data.prepare_inputs import PrepareInputs
from data.prepare_inputs_for_preds import PrepareInputs as PrepareInputsPred
#from models.random_search import perform_random_search
import json
import os
import argparse
import subprocess

def prepare_inputs(config_path, samples_path, out_path):

    # create the output directory
    os.makedirs(out_path, exist_ok=True)

    # Load the configuration json files
    #with open(f"{config_path}/input_variables.json", 'r') as f:
    #    input_vars = json.load(f)
    
    input_vars_path = f"{config_path}/input_variables.json"

    with open(f"{config_path}/samples_and_classes.json", 'r') as f:
        samples_and_classes = json.load(f)

    ## Prepare the inputs
    prep_inputs = PrepareInputs(input_var_json=input_vars_path,
                                sample_to_class = samples_and_classes["sample_to_class"],
                                class_num = samples_and_classes["class_num"],
                                classes = samples_and_classes["classes"],)
    prep_inputs.prep_input_for_mlp(samples_path, out_path)

    # Prepare the inputs for prediction sim
    print('INFO: Preparing the inputs for prediction sim')
    prep_inputs_pred = PrepareInputsPred(input_var_json=input_vars_path,
                                         sample_to_class = samples_and_classes["sample_to_class_pred"],
                                         class_num = samples_and_classes["class_num_pred"],
                                         classes = samples_and_classes["classes"],)
    prep_inputs_pred.prep_input_for_mlp(samples_path, out_path)

    #print('INFO: Preparing the inputs for prediction sim')
    #prep_inputs_pred = PrepareInputsPred(input_var_json=input_vars_path,
    #                                     sample_to_class = samples_and_classes["sample_to_class_pred"],
    #                                     class_num = samples_and_classes["class_num"],
    #                                     classes = samples_and_classes["classes"],)
    #prep_inputs_pred.prep_input_for_mlp(samples_path, out_path)

    # prepare the inputs for prediction data
    print('INFO: Preparing the inputs for prediction data')
    prep_inputs_pred_data = PrepareInputsPred(input_var_json=input_vars_path,
                                         sample_to_class = samples_and_classes["samples_data"],
                                         class_num = samples_and_classes["class_num"],
                                         classes = samples_and_classes["classes"],)
    prep_inputs_pred_data.prep_input_for_mlp_data(samples_path, out_path)

    # Perform random search
    #perform_random_search(out_path, ntrial = 1)


def end_to_end_commands(out_path, config_path):

    # do random search
    print('INFO: Performing random search')
    subprocess.run(f"python3 models/random_search.py --input_path {out_path}", shell=True)

    # do training for the best model
    print('INFO: Training the best model')
    subprocess.run(f"python3 models/training_utils.py --input_path {out_path}", shell=True)

    # get results plots
    print('INFO: Getting the results plots')
    subprocess.run(f"python3 models/mlp_plotter.py --input_path {out_path}", shell=True)

    # get the predictions
    print('INFO: Getting the predictions')
    subprocess.run(f"python3 models/get_prediction.py --model_folder {out_path}/after_random_search_best1/ --samples_path {out_path} --config_path {config_path}", shell=True)

    # get non resonant mass for different ggFHH score cuts
    print('INFO: Getting non resonant mass for different ggFHH score cuts')
    subprocess.run(f"python3 test_cor_mass.py --input_path {out_path} --config_path {config_path}", shell=True)

    # categorisation

    # 

    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--samples_path', type=str, help='Path to the samples, Eg: /eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v2/Run3_2022')
    parser.add_argument('--out_path', type=str, help='Path to save the inputs')
    args = parser.parse_args()
    prepare_inputs(args.config_path, args.samples_path, args.out_path)
    end_to_end_commands(args.out_path, args.config_path)