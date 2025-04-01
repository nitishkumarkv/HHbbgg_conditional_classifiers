from utils.MLP_based_categorization import prep_inputs, train_MLP_based_categorization
import torch
import argparse

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preform MLP based categorization')

    parser.add_argument('--samples_path', type=str, help='Path to the samples')
    parser.add_argument('--path_to_save_inputs', type=str, help='Path to save the inputs')
    parser.add_argument('--model_config_path', type=str, help='Path to the model configuration')

    args = parser.parse_args()

    # Define the samples
    samples = ["GGJets",
                   "GJetPt20To40",
                   "GJetPt40",
                   "TTGG",
                   "ttHtoGG_M_125",
                   "GluGluHToGG_M_125",
                   "VBFHToGG_M_125",
                   "VHtoGG_M_125",
                   "GluGlutoHHto2B2G_kl-1p00_kt-1p00_c2-0p00",
                   "VBFHHto2B2G_CV_1_C2V_1_C3_1"]

    # Prepare the inputs
    prep_inputs(args.samples_path, samples, args.path_to_save_inputs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_MLP_based_categorization(args.path_to_save_inputs, args.model_config_path, device)


if __name__ == "__main__":
    main()
