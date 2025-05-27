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
import pickle
import json
import os
hep.style.use("CMS")


# Wrapper class for your model
class ModelEstimatorWrapper:
    def __init__(self, param_dict_path, model_path):
        self.param_dict_path = param_dict_path
        self.model_path = model_path
        self.model = None
    
    def load_model(self, input_size):
        # Load the model parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        output_size = best_params.get('output_size', 4)  # Adjust as per your problem

        # Define the model
        self.model = MLP(
            input_size, best_num_layers, best_num_nodes, output_size,
            best_act_fn, best_dropout_prob
        ).to(device)

        # Load the model state
        model_state = torch.load(self.model_path, weights_only=False, map_location=device)
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
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        if self.model is None:
            self.load_model(X.shape[1])
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

# Define your scoring functions
def weighted_accuracy(y_true, y_pred, sample_weight):
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

def weighted_log_loss(y_true, y_pred_proba, sample_weight):
    # Return negative log loss to align with scikit-learn's maximization
    return -log_loss(y_true, y_pred_proba, sample_weight=sample_weight)

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
    args = parser.parse_args()
    input_path= args.input_path

    # # load data
    # print("INFO: Loading inputs")
    # X_train = np.load(f'{input_path}/X_train.npy')
    # y_train = np.load(f'{input_path}/y_train.npy')
    # rel_w_train = np.load(f'{input_path}/rel_w_train.npy')

    # class_weights_for_train_no_aboslute = np.load(f'{input_path}/true_class_weights.npy')

    X_val = np.load(f'{input_path}/X_val.npy')
    y_val = np.load(f'{input_path}/y_val.npy')
    rel_w_val = np.load(f'{input_path}/rel_w_val.npy')
    class_weights_for_val = np.load(f'{input_path}/class_weights_for_val.npy')
    print(y_val)

    # load list of input features
    with open(f'{input_path}/input_vars.txt', 'r') as f:
        input_vars = json.load(f)

    print("INFO: Inputs loaded")


    ####### Permutation Importance #######
    # Paths to your model dictionary and model state
    training_folder = f"{input_path}/after_random_search_best1/"
    param_dict_path = f'{training_folder}/params.json'
    model_path = f'{training_folder}/mlp.pth'
    path_to_importance_plots = f'{training_folder}/permutation_importances_plots/'
    os.makedirs(path_to_importance_plots, exist_ok=True)

    # Instantiate your model wrapper
    model_wrapper = ModelEstimatorWrapper(param_dict_path, model_path)

    if not os.path.exists(f'{path_to_importance_plots}/permutation_importances.pkl'):

        # Compute permutation importance for weighted log loss
        print("Computing permutation importance using weighted log loss...")
        result_log_loss = permutation_importance(
            model_wrapper, X_val, y_val, n_repeats=5,
            scoring=lambda estimator, X, y: weighted_log_loss(y, estimator.predict_proba(X), class_weights_for_val),
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
