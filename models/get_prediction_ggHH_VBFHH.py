import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from mlp import MLP

def get_prediction_ggHH(model_dict_path, model_path, X):

    if isinstance(model_dict_path, str):
        with open(model_dict_path, 'r') as f:
            best_params = json.load(f)
            #print(best_params)
    else:
        best_params = model_dict_path

    best_num_layers = best_params['num_layers']
    best_num_nodes = best_params['num_nodes']
    best_act_fn_name = best_params['act_fn_name']
    best_act_fn = getattr(nn, best_act_fn_name)
    best_dropout_prob = best_params['dropout_prob']
    input_size = X.shape[1]
    output_size = 4
    # output_size = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    model = MLP(input_size, best_num_layers, best_num_nodes, output_size, best_act_fn, best_dropout_prob).to(device)
    model.to(device)
    model_state = torch.load(model_path, weights_only=False, map_location=torch.device(device))
    # if model_path.endswith("mlp.pth"):
    #     print("Loading model from mlp.pth")
    #     model.load_state_dict(model_state['model_state_dict'])
    # else:
    print(f"Loading model from {model_path}")
    model.load_state_dict(model_state['best_weights'])

    model.eval()
    batch_size = 1024
    y_preds = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].to(device)
        with torch.no_grad():
            y_batch = model(X_batch)
            y_batch = F.softmax(y_batch, dim=1)
            y_preds.append(y_batch.cpu().numpy())
    
    y = np.concatenate(y_preds, axis=0)
    print(y.shape)

    return y

def prep_inputs_for_prediction_sim(samples_path, training_config, input_vars_txt, out_dir):

    os.makedirs(out_dir, exist_ok=True)
     
    with open(input_vars_txt, "r") as f:
        vars_for_training = json.load(f)

    extra_vars = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_HHbbggCandidate_mass", "weight", "pt", "nonResReg_dijet_pt", "nonResReg_lead_bjet_pt", "nonResReg_sublead_bjet_pt", "nonResReg_lead_bjet_eta", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi", "lead_genPartFlav", "sublead_genPartFlav"]

    vars_to_load = vars_for_training + extra_vars

    for era in training_config["samples_info"]["eras"]:
        for samples in training_config["samples_info"][era].keys():
            parquet_path = f"individual_samples/{era}/{samples}/events.parquet"
            events = ak.from_parquet(f"{samples_path}/{parquet_path}")

            print(f"INFO: Number of events in {samples} for {era}: {len(events)}")

            write_chunk = training_config["write_chunk"]
            comb_inputs = pd.DataFrame()
            i = 0
            while len(events) > 0:
                events_intermediate = events[:write_chunk]
                events = events[write_chunk:]
                comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                i+=1

            for col in ["nonResReg_vbfpair_CosThetaStar_CS", "nonResReg_vbfpair_CosThetaStar_gg", "nonResReg_vbfpair_CosThetaStar_jj"]:
                if col in comb_inputs.columns:
                    comb_inputs[col] = comb_inputs[col].abs()
            X = comb_inputs[vars_for_training]
            # relative_weights = comb_inputs["rel_xsec_weight"]

            X = X.values
            # relative_weights = relative_weights.values

            # save all the numpy arrays
            print("INFO: saving inputs for VBFHH MVA")
            full_path_to_save = f"{out_dir}/individual_samples/{era}/{samples}"
            os.makedirs(full_path_to_save, exist_ok=True)
            print(f"Saving X for {samples} in {era} era to {full_path_to_save}/X.npy")
            np.save(f"{full_path_to_save}/X", X)
            # np.save(f"{full_path_to_save}/rel_w", relative_weights)
            
    return 0

def prep_inputs_for_prediction_sim_sys(samples_path, training_config, input_vars_txt, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    with open(input_vars_txt, "r") as f:
        vars_for_training = json.load(f)

    extra_vars = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_HHbbggCandidate_mass", "weight", "pt", "nonResReg_dijet_pt", "nonResReg_lead_bjet_pt", "nonResReg_sublead_bjet_pt", "nonResReg_lead_bjet_eta", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi", "lead_genPartFlav", "sublead_genPartFlav"]

    vars_to_load = vars_for_training + extra_vars

    for era in training_config["samples_info"]["eras"]:
        for samples in training_config["samples_info"][era].keys():
            for sys in training_config["systematics"]:
                
                if samples in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
                        continue

                parquet_path = f"individual_samples/{era}/{samples}/{sys}/events.parquet"
                
                if not os.path.exists(f"{samples_path}/{parquet_path}"):
                    print(f"WARNING: {samples} for {era} for {sys} does not exist. Skipping.: {samples_path}/{parquet_path}")
                    continue

                events = ak.from_parquet(f"{samples_path}/{parquet_path}")

                print(f"INFO: Number of events in {samples} for {era} for {sys}: {len(events)}")

                write_chunk = training_config["write_chunk"]
                comb_inputs = pd.DataFrame()
                i = 0
                while len(events) > 0:
                    events_intermediate = events[:write_chunk]
                    events = events[write_chunk:]
                    comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
                    i+=1

                for col in ["nonResReg_vbfpair_CosThetaStar_CS", "nonResReg_vbfpair_CosThetaStar_gg", "nonResReg_vbfpair_CosThetaStar_jj"]:
                    if col in comb_inputs.columns:
                        comb_inputs[col] = comb_inputs[col].abs()
                X = comb_inputs[vars_for_training]
                # relative_weights = comb_inputs["rel_xsec_weight"]

                X = X.values
                # relative_weights = relative_weights.values

                # save all the numpy arrays
                print("INFO: saving inputs for VBFHH MVA")
                full_path_to_save = f"{out_dir}/individual_samples/{era}/{samples}/{sys}"
                os.makedirs(full_path_to_save, exist_ok=True)
                np.save(f"{full_path_to_save}/X", X)
                # np.save(f"{full_path_to_save}/rel_w", relative_weights)
                
    return 0


def prep_inputs_for_prediction_data(samples_path, training_config, input_vars_txt, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    with open(input_vars_txt, "r") as f:
        vars_for_training = json.load(f)

    extra_vars = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_HHbbggCandidate_mass", "weight", "pt", "nonResReg_dijet_pt", "nonResReg_lead_bjet_pt", "nonResReg_sublead_bjet_pt", "nonResReg_lead_bjet_eta", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi", "lead_genPartFlav", "sublead_genPartFlav"]

    vars_to_load = vars_for_training + extra_vars

    for data in training_config["samples_info"]["data"]:

        sample_to_era = {
            "2016preVFP_EraBv1": "2016preVFP",
            "2016preVFP_EraBv2": "2016preVFP",
            "2016preVFP_EraC": "2016preVFP",
            "2016preVFP_EraD": "2016preVFP",
            "2016preVFP_EraE": "2016preVFP",
            "2016preVFP_EraF": "2016preVFP",
            "2016postVFP_EraF": "2016postVFP",
            "2016postVFP_EraG": "2016postVFP",
            "2016postVFP_EraH": "2016postVFP",
            "2017_EraB": "2017",
            "2017_EraC": "2017",
            "2017_EraD": "2017",
            "2017_EraE": "2017",
            "2017_EraF": "2017",
            "2018_EraA": "2018",
            "2018_EraB": "2018",
            "2018_EraC": "2018",
            "2018_EraD": "2018",
            "2017": "2017",
            "2018": "2018",
            "2022_EraE": "postEE", 
            "2022_EraF": "postEE", 
            "2022_EraG": "postEE", 
            "2022_EraC": "preEE", 
            "2022_EraD": "preEE",
            "2023_EraC": "preBPix",
            "2023_EraD": "postBPix",
            "2024_EraC_EG0": "2024",
            "2024_EraC_EG1": "2024",
            "2024_EraD_EG0": "2024",
            "2024_EraD_EG1": "2024",
            "2024_EraE_EG0": "2024",
            "2024_EraE_EG1": "2024",
            "2024_EraF_EG0": "2024",
            "2024_EraF_EG1": "2024",
            "2024_EraG_EG0": "2024",
            "2024_EraG_EG1": "2024",
            "2024_EraH_EG0": "2024",
            "2024_EraH_EG1": "2024",
            "2024_EraIv1_EG0": "2024",
            "2024_EraIv1_EG1": "2024",
            "2024_EraIv2_EG0": "2024",
            "2024_EraIv2_EG1": "2024",
            "2025_EraCv1_EG0": "2025",
            "2025_EraCv1_EG1": "2025",
            "2025_EraCv1_EG2": "2025",
            "2025_EraCv1_EG3": "2025",
            "2025_EraCv2_EG0": "2025",
            "2025_EraCv2_EG1": "2025",
            "2025_EraCv2_EG2": "2025",
            "2025_EraCv2_EG3": "2025",
            "2025_EraDv1_EG0": "2025",
            "2025_EraDv1_EG1": "2025",
            "2025_EraDv1_EG2": "2025",
            "2025_EraDv1_EG3": "2025",
            "2025_EraEv1_EG0": "2025",
            "2025_EraEv1_EG1": "2025",
            "2025_EraEv1_EG2": "2025",
            "2025_EraEv1_EG3": "2025",
            "2025_EraFv1_EG0": "2025",
            "2025_EraFv1_EG1": "2025",
            "2025_EraFv1_EG2": "2025",
            "2025_EraFv1_EG3": "2025",
            "2025_EraFv2_EG0": "2025",
            "2025_EraFv2_EG1": "2025",
            "2025_EraFv2_EG2": "2025",
            "2025_EraFv2_EG3": "2025",
            "2025_EraGv1_EG0": "2025",
            "2025_EraGv1_EG1": "2025",
            "2025_EraGv1_EG2": "2025",
            "2025_EraGv1_EG3": "2025",
        }
        
        parquet_path = f"individual_samples_data/{data}/events.parquet"
        
        if not os.path.exists(f"{samples_path}/{parquet_path}"):
            print(f"WARNING: {data} does not exist. Skipping.: {samples_path}/{parquet_path}")
            continue

        events = ak.from_parquet(f"{samples_path}/{parquet_path}")

        print(f"INFO: Number of events in {data}: {len(events)}")

        write_chunk = training_config["write_chunk"]
        comb_inputs = pd.DataFrame()
        i = 0
        while len(events) > 0:
            events_intermediate = events[:write_chunk]
            events = events[write_chunk:]
            comb_inputs = pd.concat([comb_inputs, pd.DataFrame(ak.to_list(events_intermediate))])
            i+=1

        for col in ["nonResReg_vbfpair_CosThetaStar_CS", "nonResReg_vbfpair_CosThetaStar_gg", "nonResReg_vbfpair_CosThetaStar_jj"]:
            if col in comb_inputs.columns:
                comb_inputs[col] = comb_inputs[col].abs()
        X = comb_inputs[vars_for_training]
        # relative_weights = comb_inputs["rel_xsec_weight"]

        X = X.values
        # relative_weights = relative_weights.values

        # save all the numpy arrays
        print(f"INFO: saving inputs for {data}")
        full_path_to_save = f"{out_dir}/individual_samples_data/{data}"
        os.makedirs(full_path_to_save, exist_ok=True)
        np.save(f"{full_path_to_save}/X", X)
        # np.save(f"{full_path_to_save}/rel_w", relative_weights)
        
    return 0

def get_prediction_VBFHH(model, X):

    if model.input_shape[-1] != X.shape[1]:
        raise RuntimeError("Input mismatch: model expects {model.input_shape[-1]} features, but X has {X.shape[1]}")

    batch_size = 1024
    y = model.predict(X, batch_size=batch_size)
    print(y.shape)

    return y

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Preform MLP based classification')
    parser.add_argument('--MVA_choice', type=str, choices=['ggHH_only', 'VBFHH_only', 'all'], default='all', help='Choice of MVA: ggHH or VBFHH')
    # ggHH model
    parser.add_argument('--ggHH_model_folder', type=str, help='Path to the ggHH model folder containing the params.json and mlp.pth files')
    parser.add_argument('--samples_path', type=str, help='Path to the samples')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    # VBFHH model
    parser.add_argument('--VBFHH_model', type=str, help='Path to the VBFHH model')
    parser.add_argument('--VBFHH_vars_path', type=str)
    parser.add_argument('--VBFHH_out_dir', type=str)
    # nominal only or with syst
    parser.add_argument('--get_pred_nominal', action='store_true')
    parser.add_argument('--get_pred_sys', action='store_true')
    parser.add_argument('--get_pred_data', action='store_true')

    args = parser.parse_args()

    model_folder = args.ggHH_model_folder
    model_dict_path = f"{model_folder}/params.json"
    model_path = f"{model_folder}/mlp.pth"

    # load the configuration yaml files
    training_config_path = f"{args.config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    samples_path = args.samples_path
    eras = training_config["samples_info"]["eras"]

    if args.MVA_choice != "ggHH_only":
        input_vars_txt = args.VBFHH_vars_path
        if args.VBFHH_model:    
            model_VBFHH_Run2 = tf.keras.models.load_model(args.VBFHH_model)
            model_VBFHH_Run3 = tf.keras.models.load_model(args.VBFHH_model)
        else:
            print("No VBFHH model provided. Will use the default model for VBFHH MVA.")
            model_VBFHH_Run2 = tf.keras.models.load_model("/eos/user/c/chuxue/HHbbgg/inputs_VBFMVA/VBFMVA_v6parquet.keras")            
            model_VBFHH_Run3 = tf.keras.models.load_model("/eos/user/c/chuxue/HHbbgg/inputs_VBFMVA/VBFMVA_v6parquet.keras")

        if args.get_pred_data:
            prep_inputs_for_prediction_data(samples_path, training_config, input_vars_txt, args.VBFHH_out_dir)
        if args.get_pred_nominal:
            prep_inputs_for_prediction_sim(samples_path, training_config, input_vars_txt, args.VBFHH_out_dir)
        if args.get_pred_sys:
            prep_inputs_for_prediction_sim_sys(samples_path, training_config, input_vars_txt, args.VBFHH_out_dir)

    if args.get_pred_nominal:

        for era in eras:
            samples = training_config["samples_info"][era].keys()

            if args.MVA_choice != "ggHH_only":
                if era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
                    model_VBFHH = model_VBFHH_Run2
                    print(f"Using Run2 VBFHH model for {era}")
                else:
                    model_VBFHH = model_VBFHH_Run3
                    print(f"Using Run3 VBFHH model for {era}")

            for sample in samples:
                if args.MVA_choice != "VBFHH_only":
                    ## ggHH prediction
                    inputs_path = f"{samples_path}/individual_samples/{era}/{sample}"

                    device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
                    # device = 'cpu'
                    print("Device: ", device)
                    X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                    print(f"Getting prediction for {sample} in {era} era")
                    #pred = get_prediction(model_dict_path, model_path, X)
                    pred = get_prediction_ggHH(model_dict_path, model_path, X)
                    print(np.sum(pred, axis=1))
                    # save the prediction
                    print(f"Saving prediction for {sample} in {era} era \n")
                    np.save(f"{samples_path}/individual_samples/{era}/{sample}/y.npy", pred)

                if args.MVA_choice != "ggHH_only":
                    ## VBFHH prediction
                    inputs_path_VBFHH = f"{args.VBFHH_out_dir}/individual_samples/{era}/{sample}"

                    X_VBFHH = np.load(f'{inputs_path_VBFHH}/X.npy').astype(np.float32)

                    print(f"Getting prediction for {sample} in {era} era for VBFHH MVA")
                    #pred = get_prediction(model_dict_path, model_path, X)
                    pred_VBFHH = get_prediction_VBFHH(model_VBFHH, X_VBFHH)
                    print(np.sum(pred_VBFHH, axis=1))
                    # save the prediction
                    print(f"Saving prediction for {sample} in {era} era for VBFHH MVA \n")
                    np.save(f"{args.VBFHH_out_dir}/individual_samples/{era}/{sample}/y.npy", pred_VBFHH)

    elif args.get_pred_sys:
        for era in eras:   
            samples = training_config["samples_info"][era].keys()

            if args.MVA_choice != "ggHH_only":
                if era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
                    model_VBFHH = model_VBFHH_Run2
                    print(f"Using Run2 VBFHH model for {era}")
                else:
                    model_VBFHH = model_VBFHH_Run3
                    print(f"Using Run3 VBFHH model for {era}")

            for sample in samples:
                if sample in ["GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT", "TTGG"]:
                    continue
                for sys in training_config["systematics"]:
                    if args.MVA_choice != "VBFHH_only":
                        ## ggHH prediction
                        inputs_path = f"{samples_path}/individual_samples/{era}/{sample}/{sys}/"

                        device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
                        # device = 'cpu'
                        print("Device: ", device)
                        X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                        print(f"Getting prediction for {sample} in {era} era")
                        #pred = get_prediction(model_dict_path, model_path, X)
                        pred = get_prediction_ggHH(model_dict_path, model_path, X)
                        print(np.sum(pred, axis=1))
                        # save the prediction
                        print(f"Saving prediction for {sample} in {era} era for {sys} \n")
                        np.save(f"{samples_path}/individual_samples/{era}/{sample}/{sys}/y.npy", pred)

                    if args.MVA_choice != "ggHH_only":
                        ## VBFHH prediction
                        inputs_path_VBFHH = f"{args.VBFHH_out_dir}/individual_samples/{era}/{sample}/{sys}/"

                        X_VBFHH = np.load(f'{inputs_path_VBFHH}/X.npy').astype(np.float32)

                        print(f"Getting prediction for {sample} in {era} era for VBFHH MVA ")
                        #pred = get_prediction(model_dict_path, model_path, X)
                        pred_VBFHH = get_prediction_VBFHH(model_VBFHH, X_VBFHH)
                        print(np.sum(pred_VBFHH, axis=1))
                        # save the prediction
                        print(f"Saving prediction for {sample} in {era} era for {sys} for VBFHH MVA \n")
                        print(f"Saving prediction path: {args.VBFHH_out_dir}/individual_samples/{era}/{sample}/{sys}/y.npy \n")
                        np.save(f"{args.VBFHH_out_dir}/individual_samples/{era}/{sample}/{sys}/y.npy", pred_VBFHH)
    
    if args.get_pred_data:
        data_samples = training_config["samples_info"]["data"].keys()

        for data_sample in data_samples:

            if args.MVA_choice != "ggHH_only":
                if data_sample in ["2016preVFP", "2016postVFP", "2017", "2018"]:
                    model_VBFHH = model_VBFHH_Run2
                    print(f"Using Run2 VBFHH model for {data_sample}")
                else:
                    model_VBFHH = model_VBFHH_Run3
                    print(f"Using Run3 VBFHH model for {data_sample}")

            if args.MVA_choice != "VBFHH_only":
                ## ggHH prediction
                inputs_path = f"{samples_path}/individual_samples_data/{data_sample}"
                device = torch.device('cuda:'+training_config["cuda_device"] if torch.cuda.is_available() else 'cpu')
                # device = 'cpu'
                if not os.path.exists(f'{inputs_path}/X.npy'):
                    print(f"WARNING: {data_sample} does not exist. Skipping.: {inputs_path}/X.npy")
                    continue
                X = torch.tensor(np.load(f'{inputs_path}/X.npy'), dtype=torch.float32).to(device)

                print(f"Getting prediction for {data_sample}")
                #pred = get_prediction(model_dict_path, model_path, X)
                pred = get_prediction_ggHH(model_dict_path, model_path, X)
                print(np.sum(pred, axis=1))
                # save the prediction
                print(f"Saving prediction for {data_sample} \n")
                np.save(f"{samples_path}/individual_samples_data/{data_sample}/y.npy", pred)

            if args.MVA_choice != "ggHH_only":
                ## VBFHH prediction
                inputs_path_VBFHH = f"{args.VBFHH_out_dir}/individual_samples_data/{data_sample}"
                X_VBFHH = np.load(f'{inputs_path_VBFHH}/X.npy').astype(np.float32)

                print(f"Getting prediction for {data_sample} for VBFHH MVA")
                #pred = get_prediction(model_dict_path, model_path, X)
                pred_VBFHH = get_prediction_VBFHH(model_VBFHH, X_VBFHH)
                print(np.sum(pred_VBFHH, axis=1))
                # save the prediction
                print(f"Saving prediction for {data_sample} for VBFHH MVA \n")
                np.save(f"{args.VBFHH_out_dir}/individual_samples_data/{data_sample}/y.npy", pred_VBFHH)
