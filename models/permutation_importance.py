import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import pandas as pd
from scipy.spatial.distance import correlation
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
from mlp import MLP
from mjj_predictor_mlp import MJJPredictorMLP
import pickle
import json
import os
import yaml
hep.style.use("CMS")
from utils.device import get_torch_device


# Wrapper class for your model
class ModelEstimatorWrapper:
    def __init__(self, param_dict_path, model_path, args, additional_config=None):
        self.param_dict_path = param_dict_path
        self.model_path = model_path
        self.args = args
        self.model = None
        self.sculpting_study_mode = "sculpting_study" in args.training_folder
        # Task flag: sculpting study => regression; otherwise => classification
        self.is_regression = self.sculpting_study_mode
        additional_config = additional_config or {}
        if self.sculpting_study_mode:
            self.sculpting_study_config = additional_config
        else:
            self.sculpting_study_config = None
    
    def load_model(self, input_size):
        # Load the model parameters
        device = get_torch_device()
        self.device = device

        # Load best parameters from JSON file
        if isinstance(self.param_dict_path, str):
            with open(self.param_dict_path, 'r') as f:
                best_params = json.load(f)
                print("Loaded best parameters:", best_params)
        else:
            best_params = self.param_dict_path

        best_num_layers = best_params['num_layers']
        best_num_nodes = best_params['num_nodes']
        best_act_fn_name = best_params['act_fn_name']
        best_act_fn = getattr(nn, best_act_fn_name)
        best_dropout_prob = best_params['dropout_prob']
        if self.sculpting_study_mode and self.sculpting_study_config is not None:
            output_size = len(self.sculpting_study_config["target_variables"])
        else:
            output_size = best_params.get('output_size', 4)  # Adjust as per your problem

        # Define the model
        if "sculpting_study" in self.args.training_folder:
            self.model = MJJPredictorMLP(
                input_size, best_num_layers, best_num_nodes, output_size,
                best_act_fn, best_dropout_prob
            ).to(device)
        else:
            self.model = MLP(
                input_size, best_num_layers, best_num_nodes, output_size,
                best_act_fn, best_dropout_prob
            ).to(device)

        # Load the model state
        model_state = torch.load(self.model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.eval()
    
    def fit(self, X, y):
        # Dummy fit method to satisfy scikit-learn's requirement
        pass

    def predict(self, X):
        if self.model is None:
            self.load_model(X.shape[1])
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # For regression tasks, return raw outputs
            if self.is_regression:
                return outputs.cpu().numpy()
            # For classification tasks, return class indices
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        if self.model is None:
            self.load_model(X.shape[1])
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Only meaningful for classification
            if self.is_regression:
                raise ValueError("predict_proba called for regression task")
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

# Define your scoring functions
def weighted_accuracy(y_true, y_pred, sample_weight):
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

def weighted_log_loss(y_true, y_pred_proba, sample_weight):
    # Return negative log loss to align with scikit-learn's maximization
    return -log_loss(y_true, y_pred_proba, sample_weight=sample_weight)

def weighted_mse(y_true, y_pred, sample_weight):
    """Negative weighted MSE (so higher is better). Supports multi-output by averaging per sample."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = (y_true - y_pred) ** 2
    if err.ndim == 2:
        per_sample = err.mean(axis=1)
    else:
        per_sample = err
    return -np.average(per_sample, weights=sample_weight)

def plot_permutation_importance_log_loss(importances, stds, feature_names):
    # Sort importances and features
    indices = np.argsort(importances)
    sorted_features = np.array(feature_names)[indices]
    sorted_importances = importances[indices]
    sorted_stds = stds[indices]

    # Create a horizontal bar plot
    plt.figure(figsize=(10, len(sorted_features) * 0.4))
    plt.barh(range(len(sorted_features)), sorted_importances, xerr=sorted_stds, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Increase in log loss score')
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)  # Add a vertical line at x=0 for reference
    plt.tight_layout()
    plt.show()

def class_specific_log_loss(y_true, y_pred_proba, sample_weight, class_index):
    """Compute log loss only for a specific class (as binary classification)."""
    y_true = np.argmax(y_true, axis=1)  # Convert to class indices
    y_true_binary = (y_true == class_index).astype(int)
    y_pred_class = y_pred_proba[:, class_index]

    # Build binary 2D probability array
    y_pred_binary = np.stack([1 - y_pred_class, y_pred_class], axis=1)

    # 💥 Fix the dtype of y_true_binary to prevent multilabel confusion
    return -log_loss(y_true_binary.tolist(), y_pred_binary, sample_weight=sample_weight, labels=[0, 1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Permutation Importance for MLP')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--X_path', type=str, default="X_val.npy", help='Path to the input features numpy file relative to input_path')
    parser.add_argument("--y_path", type=str, default="y_val.npy", help='Path to the target labels numpy file relative to input_path')
    parser.add_argument('--rel_w_path', type=str, default="rel_w_val.npy", help='Path to the relative weights numpy file relative to input_path')
    parser.add_argument("--weights_path", type=str, default="class_weights_for_val.npy", help='Path to the class weights numpy file relative to input_path')
    parser.add_argument("--input_vars_path", type=str, default="input_vars.txt", help='Path to the input variable names text file relative to input_path')
    parser.add_argument("--training_folder", type=str, default="after_random_search_best1", help='Path to the training folder containing model and params.json relative to input_path')
    parser.add_argument("--sculpting_study_config_path", type=str, default=None, help='Path to the sculpting study config file relative to input_path')
    args = parser.parse_args()
    input_path= args.input_path

    # # load data
    # print("INFO: Loading inputs")
    # X_train = np.load(f'{input_path}/X_train.npy')
    # y_train = np.load(f'{input_path}/y_train.npy')
    # rel_w_train = np.load(f'{input_path}/rel_w_train.npy')

    # class_weights_for_train_no_aboslute = np.load(f'{input_path}/true_class_weights.npy')

    X_val = np.load(os.path.join(args.input_path, args.X_path))
    y_val = np.load(os.path.join(args.input_path, args.y_path))
    rel_w_val = np.load(os.path.join(args.input_path, args.rel_w_path))
    class_weights_for_val = np.load(os.path.join(args.input_path, args.weights_path))
    print(y_val)

    # load list of input features
    with open(os.path.join(args.input_path, args.input_vars_path), 'r', encoding="utf-8") as f:
        input_vars = json.load(f)

    if args.sculpting_study_config_path is not None:
        with open(args.sculpting_study_config_path, 'r', encoding="utf-8") as f:
            additional_config = yaml.safe_load(f)
    else:
        additional_config = None

    print("INFO: Inputs loaded")


    ####### Permutation Importance #######
    # Paths to your model dictionary and model state
    training_folder = os.path.join(args.input_path, args.training_folder)
    param_dict_path = os.path.join(training_folder, 'params.json')
    model_path = os.path.join(training_folder, 'mlp.pth')
    path_to_importance_plots = os.path.join(training_folder, 'permutation_importances_plots')
    os.makedirs(path_to_importance_plots, exist_ok=True)

    # Instantiate your model wrapper
    model_wrapper = ModelEstimatorWrapper(param_dict_path, model_path, args, additional_config)

    if not os.path.exists(os.path.join(path_to_importance_plots, 'permutation_importances.pkl')):

        # Choose scoring based on task
        if model_wrapper.is_regression:
            print("Computing permutation importance using weighted MSE (regression)...")
            scoring_fn = lambda estimator, X, y: weighted_mse(y, estimator.predict(X), class_weights_for_val)
        else:
            print("Computing permutation importance using weighted log loss (classification)...")
            scoring_fn = lambda estimator, X, y: weighted_log_loss(y, estimator.predict_proba(X), class_weights_for_val)

        result_log_loss = permutation_importance(
            model_wrapper, X_val, y_val, n_repeats=5,
            scoring=scoring_fn,
            random_state=42
        )

        importances_log_loss = result_log_loss.importances_mean
        std_log_loss = result_log_loss.importances_std

        df_importances = pd.DataFrame({
            'Feature': input_vars,
            #'Accuracy_importance': importances_accuracy,
            #'Accuracy_std': std_accuracy,
            'log_loss_importance': importances_log_loss,
            'log_loss_std': std_log_loss
        })

        # Sort by accuracy importance
        df_importances.sort_values(by='log_loss_importance', ascending=True, inplace=True)

        # save the importances to a pickle file
        df_importances.to_pickle(f'{path_to_importance_plots}/permutation_importances.pkl')

    else:
        print("INFO: Loading permutation importances from pickle file")
        df_importances = pd.read_pickle(f'{path_to_importance_plots}/permutation_importances.pkl')

    print("\nPermutation Importances:")
    print(df_importances)

    # Plot the feature importances for weighted log loss
    plt.figure(figsize=(10, 20))
    plt.barh(df_importances['Feature'], df_importances['log_loss_importance'], xerr=df_importances['log_loss_std'])
    #plt.gca().invert_yaxis()
    plt.yticks(fontsize=10)
    plt.xlabel('Permutation importance')
    plt.tight_layout()
    plt.savefig(f'{path_to_importance_plots}/permutation_importance_log_loss.png', dpi=300)
    plt.clf()


    df_sorted = df_importances.sort_values('log_loss_importance', ascending=False)
    x = np.arange(len(df_sorted))
    plt.figure(figsize=(max(20, len(df_sorted) * 0.35), 8))  # Dynamically scale width
    plt.bar(x, df_sorted['log_loss_importance'], yerr=df_sorted['log_loss_std'], align='center')
    plt.xticks(x, df_sorted['Feature'], rotation=45, ha='right', fontsize=10)
    plt.ylabel('Permutation importance')
    plt.tight_layout()
    plt.savefig(f'{path_to_importance_plots}/permutation_importance_log_loss_vertical_cleaned.png', dpi=300)


    # for specific classes
    class_names = ["nonRes class", "ttH class", "other single H class", "ggFHH class", "VBFHH class"]
    class_labels = [f'{class_names[i]}' for i in range(4)]  # or use your own names

#    for class_idx, class_name in enumerate(class_labels):
#        print(f"\n>> Computing permutation importance for class {class_name}...")
#
#        result_class = permutation_importance(
#            model_wrapper, X_val, y_val, n_repeats=5,
#            scoring=lambda estimator, X, y: class_specific_log_loss(
#                y, estimator.predict_proba(X), class_weights_for_val, class_idx),
#            random_state=42
#        )
#
#        importances_class = result_class.importances_mean
#        std_class = result_class.importances_std
#
#        df_class = pd.DataFrame({
#            'Feature': input_vars,
#            f'{class_name}_importance': importances_class,
#            f'{class_name}_std': std_class
#        }).sort_values(by=f'{class_name}_importance', ascending=True)
#
#        # Save per-class pickle
#        df_class.to_pickle(f'{path_to_importance_plots}/permutation_importances_{class_name}.pkl')
#
#        # Sort and plot class-specific importance
#        df_class_sorted = df_class.sort_values(by=f'{class_name}_importance', ascending=False)
#        x = np.arange(len(df_class_sorted))
#        plt.figure(figsize=(max(20, len(df_class_sorted) * 0.35), 8))  # Dynamically scale width
#        plt.bar(x, df_class_sorted[f'{class_name}_importance'], 
#                yerr=df_class_sorted[f'{class_name}_std'], align='center')
#
#        plt.xticks(x, df_class_sorted['Feature'], rotation=45, ha='right', fontsize=10)
#        plt.ylabel('Permutation importance')
#        plt.title(f'Permutation importance for {class_name}')
#        plt.tight_layout()
#        plt.savefig(f'{path_to_importance_plots}/permutation_importance_{class_name}_vertical_cleaned.png', dpi=300)
#        plt.clf()
