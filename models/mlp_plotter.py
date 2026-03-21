import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import copy
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import awkward as ak
import pyarrow as pa
import mplhep as hep
from mlp import MLP
import pickle

if not hasattr(pa.lib, "PyExtensionType") and hasattr(pa.lib, "ExtensionType"):
    pa.lib.PyExtensionType = pa.lib.ExtensionType

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path, weights_only=False)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss_hist = checkpoint['train_loss_hist']
    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    train_loss_hist_no_aboslute = checkpoint['train_loss_hist_no_absolute_weights']
    val_acc_hist = checkpoint['val_acc_hist']
    lr_hist = checkpoint['lr_hist']
    best_weights = checkpoint['best_weights']
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {file_path}, resuming from epoch {start_epoch + 1}')
    return start_epoch, train_loss_hist, train_loss_hist_no_aboslute, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Plot the results of the MLP')
    parser.add_argument('--input_path', type=str, help='Path to the inputs')
    args = parser.parse_args()

    #inputs_for_MLP = "../data/inputs_for_MLP_202411226/"
    #input_path="train_inputs_for_MLP_202411226/after_random_search_best1/"
    inputs_for_MLP = args.input_path
    input_path = f"{inputs_for_MLP}/after_random_search_best1/"
    path_for_plots = f"{input_path}/plots/"
    os.makedirs(path_for_plots, exist_ok=True)
    path_to_checkpoint = f"{input_path}/mlp.pth"

    # load the checkpoint
    start_epoch, train_loss_hist, train_loss_hist_no_absolute_weights, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss = load_checkpoint(path_to_checkpoint)


    colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']

    #plot loss function
    plt.plot(train_loss_hist, label="train")
    plt.plot(val_loss_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/loss_plot.png')
    plt.clf()

    plt.plot(train_loss_hist_no_absolute_weights, label="train")
    plt.plot(val_loss_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/loss_plot_no_abs.png')
    plt.clf()

    # plot learning rate
    plt.plot(lr_hist)
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.savefig(f'{path_for_plots}/lr_plot.png')
    plt.clf()

    #plot accuracy
    plt.plot(train_acc_hist, label="train")
    plt.plot(val_acc_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/acc_plot.png')
    plt.clf()


    # load predictions
    y_pred_val_ = np.load(f"{input_path}/y_pred_val.npy")
    y_val_ = np.load(f'{inputs_for_MLP}/y_val.npy')
    #rel_w_val = np.load(f'{inputs_for_MLP}/rel_w_val.npy')
    rel_w_val_ = np.load(f'{inputs_for_MLP}/class_weights_for_val.npy')
    y_pred_val = y_pred_val_
    y_val = y_val_
    rel_w_val = rel_w_val_


    y_pred_train = np.load(f"{input_path}/y_pred_train.npy")
    y_train = np.load(f'{inputs_for_MLP}/y_train.npy')
    rel_w_train = np.load(f'{inputs_for_MLP}/true_class_weights.npy')

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from itertools import combinations

    # Load process numbers and mappings
    try:
        proc_num_train = np.load(f'{inputs_for_MLP}/proc_num_train.npy')
        proc_num_val = np.load(f'{inputs_for_MLP}/proc_num_val.npy')
        with open(f'{inputs_for_MLP}/process_numbers_mapping.json', 'r') as f:
            process_numbers_mapping = json.load(f)
        with open(f'{inputs_for_MLP}/sample_to_class_mapping.json', 'r') as f:
            sample_to_class_mapping = json.load(f)
        # Create reverse mapping: process number -> sample name for each class
        process_num_to_sample = {sample: proc_num for sample, proc_num in process_numbers_mapping.items()}
        has_process_number = True
        print("INFO: Process number data loaded successfully")
    except FileNotFoundError:
        print("WARNING: Process number files not found. Per-sample ROC curves will not be generated.")
        has_process_number = False

    # Class names
    class_names = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH", "VBFToHH_sig"]
    #n_classes = len(class_names)
    n_classes = y_val.shape[1]

    # One-vs-All ROC Curves
    one_vs_all_auc_dict = {}
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        class_name = class_names[i]
        y_true_binary = y_val[:, i]
        y_score = y_pred_val[:, i]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=rel_w_val)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:0.4f})')

        # Store the AUC for this class
        one_vs_all_auc_dict[f"{class_name}_fpr"] = fpr
        one_vs_all_auc_dict[f"{class_name}_tpr"] = tpr

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path_for_plots}/roc_curve_one_vs_all.png')

    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/roc_curve_one_vs_all_logx.png')
    plt.clf()

    # save auc scores
    with open(f'{path_for_plots}/roc_curve_one_vs_all.json', 'w') as f:
        json.dump(one_vs_all_auc_dict, f)


    # One-vs-One ROC Curves
    # For each pair of classes
    glu_idx = class_names.index("GluGluToHH")

    # List of other class indices
    other_classes = [i for i in range(n_classes) if i != glu_idx]

    # Iterate over GluGluToHH vs each other class individually
    plt.figure(figsize=(8, 6))

    GluGluToHH_one_vs_one_roc = {}
    # Iterate over GluGluToHH vs each other class individually
    for j in other_classes:
        i = glu_idx  # Index of GluGluToHH
        class_name_i = class_names[i]
        class_name_j = class_names[j]
        # Select samples belonging to class i or class j
        idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
        y_true_binary = y_val[idx, i]
        y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
        weights = rel_w_val[idx]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve on the same figure
        plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

        # store the AUC
        GluGluToHH_one_vs_one_roc[f"{class_name}_fpr"] = fpr
        GluGluToHH_one_vs_one_roc[f"{class_name}_tpr"] = tpr

    # Plot the diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')

    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual_logx.png')
    plt.clf()

    #save the auc scores
    with open(f'{path_for_plots}/GluGluToHH_vs_all.json', 'w') as f:
        json.dump(GluGluToHH_one_vs_one_roc, f)

    if n_classes>4:
        # One-vs-One ROC Curves
        # For each pair of classes
        glu_idx = class_names.index("VBFToHH_sig")

        VBFToHH_one_vs_one_roc = {}
        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]

        # Iterate over GluGluToHH vs each other class individually
        plt.figure(figsize=(8, 6))

        # Iterate over GluGluToHH vs each other class individually
        for j in other_classes:
            i = glu_idx  # Index of GluGluToHH
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

            # store the AUC
            VBFToHH_one_vs_one_roc[f"{class_name}_fpr"] = fpr
            VBFToHH_one_vs_one_roc[f"{class_name}_tpr"] = tpr

        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')

        plt.xlim([0.0001, 1.0])
        plt.xscale('log')
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual_logx.png')

        plt.clf()

        # save AUC scores
        with open(f'{path_for_plots}/VBFToHH_vs_all.json', 'w') as f:
            json.dump(VBFToHH_one_vs_one_roc, f)

    y_val = y_train
    y_pred_val = y_pred_train
    rel_w_val = rel_w_train

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        class_name = class_names[i]
        y_true_binary = y_val[:, i]
        y_score = y_pred_val[:, i]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=rel_w_val)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title('One-vs-All ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path_for_plots}/train_roc_curve_one_vs_all.png')
    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/train_roc_curve_one_vs_all_logx.png')
    plt.clf()

    if n_classes>4:
            
        # One-vs-One ROC Curves
        # For each pair of classes
        glu_idx = class_names.index("VBFToHH_sig")
    
        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]
    
        # Iterate over GluGluToHH vs each other class individually
        plt.figure(figsize=(8, 6))
    
        # Iterate over GluGluToHH vs each other class individually
        for j in other_classes:
            i = glu_idx  # Index of GluGluToHH
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')
    
        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')
    
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
    
        # Save the combined plot
        plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
        plt.close()

    # One-vs-One ROC Curves
    # For each pair of classes
    glu_idx = class_names.index("GluGluToHH")

    # List of other class indices
    other_classes = [i for i in range(n_classes) if i != glu_idx]

    # Iterate over GluGluToHH vs each other class individually
    plt.figure(figsize=(8, 6))

    # Iterate over GluGluToHH vs each other class individually
    for j in other_classes:
        i = glu_idx  # Index of GluGluToHH
        class_name_i = class_names[i]
        class_name_j = class_names[j]
        # Select samples belonging to class i or class j
        idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
        y_true_binary = y_val[idx, i]
        y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
        weights = rel_w_val[idx]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve on the same figure
        plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

    # Plot the diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt

colours = ['blue', 'red', 'green', 'orange', 'purple']

import numpy as np
import matplotlib.pyplot as plt
import mplhep

plt.style.use(mplhep.style.CMS)  # Use CMS-like style

colours = ['blue', 'red', 'green', 'orange', 'purple']

import numpy as np
import matplotlib.pyplot as plt
import mplhep

plt.style.use(mplhep.style.CMS)

colours = ['blue', 'red', 'green', 'orange', 'purple']

for i in range(n_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    class_name = class_names[i]

    max_y = 0  # Track max y for ylim

    # --- TRAIN: step plot with shaded uncertainty ---
    for j in range(n_classes):
        mask = y_train[:, j] == 1
        y_vals = y_pred_train[mask, i]
        weights = rel_w_train[mask]
        weights_sq = weights**2

        hist_raw, bin_edges = np.histogram(y_vals, bins=25, weights=weights, range=(0, 1))
        hist_sq_raw, _ = np.histogram(y_vals, bins=bin_edges, weights=weights_sq, range=(0, 1))
        bin_widths = np.diff(bin_edges)

        total_weight = np.sum(hist_raw)
        if total_weight == 0:
            continue  # Avoid division by zero for empty bins/classes

        # Normalize to density
        hist_density = hist_raw / (total_weight * bin_widths)
        uncertainty_density = np.sqrt(hist_sq_raw) / (total_weight * bin_widths)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        max_y = max(max_y, np.max(hist_density + uncertainty_density))

        # Step line
        ax.step(
            bin_centers,
            hist_density,
            where='mid',
            label=f'Train {class_names[j]}',
            color=colours[j],
            linewidth=2,
        )

        # Shaded uncertainty band
        ax.fill_between(
            bin_centers,
            hist_density - uncertainty_density,
            hist_density + uncertainty_density,
            step='mid',
            color=colours[j],
            alpha=0.3,
        )

    # --- VALIDATION: dots with error bars ---
    for j in range(n_classes):
        mask = y_val_[:, j] == 1
        y_vals = y_pred_val_[mask, i]
        weights = rel_w_val_[mask]
        weights_sq = weights**2

        hist_raw, bin_edges = np.histogram(y_vals, bins=25, weights=weights, range=(0, 1))
        hist_sq_raw, _ = np.histogram(y_vals, bins=bin_edges, weights=weights_sq, range=(0, 1))
        bin_widths = np.diff(bin_edges)

        total_weight = np.sum(hist_raw)
        if total_weight == 0:
            continue

        hist_density = hist_raw / (total_weight * bin_widths)
        uncertainty_density = np.sqrt(hist_sq_raw) / (total_weight * bin_widths)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        max_y = max(max_y, np.max(hist_density + uncertainty_density))

        ax.errorbar(
            bin_centers,
            hist_density,
            yerr=uncertainty_density,
            fmt='o',
            label=f'Valid {class_names[j]}',
            color=colours[j],
            markersize=5,
            capsize=2,
            elinewidth=1,
        )

    # Labels and style
    ax.set_xlabel(f'{class_name} score')
    ax.set_ylabel('a.u.')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3, top=max_y * 100)
    ax.set_xlim(left=0, right=1)

    ax.legend(ncol=2, fontsize=10)
    # Uncomment this if you want the CMS label
    # mplhep.cms.label(loc=0, data=True, label='Preliminary')

    fig.tight_layout()
    fig.savefig(f'{path_for_plots}/{class_name}_score.png')
    plt.close(fig)


# ============================================================================
# Per-sample ROC curves for GluGluToHH vs other classes
# ============================================================================
if has_process_number:
    print("\nINFO: Generating per-sample ROC curves for GluGluToHH...")
    
    # Only process GluGluToHH class
    target_class_name = "GluGluToHH"
    try:
        class_idx = class_names.index(target_class_name)
    except ValueError:
        print(f"WARNING: Class '{target_class_name}' not found in class_names")
        class_idx = None
    
    if class_idx is not None:
        # Get all samples belonging to this class
        samples_in_class = [sample for sample, cls in sample_to_class_mapping.items() 
                           if cls == target_class_name]
        
        if len(samples_in_class) > 1:
            print(f"Processing class '{target_class_name}' with samples: {samples_in_class}")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sample_roc_dict = {}
            
            for sample_name in samples_in_class:
                proc_num = process_numbers_mapping[sample_name]
                
                # Signal: events from GluGluToHH AND this specific sample
                class_mask = (y_val[:, class_idx] == 1)
                sample_mask = (proc_num_val == proc_num)
                signal_mask = class_mask & sample_mask
                
                # Background: all events NOT from GluGluToHH (other classes)
                background_mask = (y_val[:, class_idx] == 0)
                
                # Create binary labels: this sample vs all other classes
                y_true_binary = np.zeros(len(y_val))
                y_true_binary[signal_mask] = 1
                y_score = y_pred_val[:, class_idx]
                weights = rel_w_val.copy()
                
                # Compute ROC curve
                if np.sum(y_true_binary) > 0 and np.sum(background_mask) > 0:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
                    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, label=f'{sample_name} (AUC = {roc_auc:.4f})', linewidth=2.5)
                    sample_roc_dict[f"{sample_name}_fpr"] = list(fpr)
                    sample_roc_dict[f"{sample_name}_tpr"] = list(tpr)
                    sample_roc_dict[f"{sample_name}_auc"] = float(roc_auc)
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (Other Classes)', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'Per-Sample ROC: {target_class_name} vs Other Classes', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            # Save normal scale version
            fig.savefig(f'{path_for_plots}/roc_per_sample_{target_class_name}_vs_others.png', dpi=150)
            
            # Save log scale version
            ax.set_xlim([0.0001, 1.0])
            ax.set_xscale('log')
            fig.savefig(f'{path_for_plots}/roc_per_sample_{target_class_name}_vs_others_logx.png', dpi=150)
            plt.close(fig)
            
            # Save AUC scores to JSON
            with open(f'{path_for_plots}/roc_per_sample_{target_class_name}_vs_others.json', 'w') as f:
                json.dump(sample_roc_dict, f, indent=2)
            
            print(f"INFO: Per-sample ROC curves for {target_class_name} completed!")
        else:
            print(f"WARNING: Class '{target_class_name}' has only {len(samples_in_class)} sample(s). Skipping per-sample ROC.")


# ============================================================================
# Per-sample validation plots for GluGluToHH
# ============================================================================
if has_process_number:
    print("\nINFO: Generating per-sample validation plots for GluGluToHH...")
    
    target_class_name = "GluGluToHH"
    try:
        class_idx = class_names.index(target_class_name)
    except ValueError:
        print(f"WARNING: Class '{target_class_name}' not found in class_names")
        class_idx = None
    
    if class_idx is not None:
        samples_in_class = [sample for sample, cls in sample_to_class_mapping.items() 
                           if cls == target_class_name]
        
        if len(samples_in_class) > 1:
            print(f"Processing per-sample validation plots for '{target_class_name}' with samples: {samples_in_class}")
            
            # Create one figure for all samples
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors_per_sample = plt.cm.tab10(np.linspace(0, 1, len(samples_in_class)))
            
            max_y = 0
            
            for sample_idx, sample_name in enumerate(samples_in_class):
                proc_num = process_numbers_mapping[sample_name]
                color = colors_per_sample[sample_idx]
                
                # --- TRAIN data for this sample ---
                class_mask_train = (y_train[:, class_idx] == 1)
                sample_mask_train = (proc_num_train == proc_num)
                mask_train = class_mask_train & sample_mask_train
                
                y_vals_train = y_pred_train[mask_train, class_idx]
                weights_train = rel_w_train[mask_train]
                weights_sq_train = weights_train ** 2
                
                if len(y_vals_train) > 0:
                    hist_raw_train, bin_edges = np.histogram(y_vals_train, bins=25, weights=weights_train, range=(0, 1))
                    hist_sq_raw_train, _ = np.histogram(y_vals_train, bins=bin_edges, weights=weights_sq_train, range=(0, 1))
                    bin_widths = np.diff(bin_edges)
                    
                    total_weight_train = np.sum(hist_raw_train)
                    if total_weight_train > 0:
                        hist_density_train = hist_raw_train / (total_weight_train * bin_widths)
                        uncertainty_density_train = np.sqrt(hist_sq_raw_train) / (total_weight_train * bin_widths)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        
                        max_y = max(max_y, np.max(hist_density_train + uncertainty_density_train))
                        
                        # Step line for train
                        ax.step(bin_centers, hist_density_train, where='mid', 
                               label=f'Train {sample_name}', color=color, linewidth=2.5, linestyle='-')
                        
                        # Shaded uncertainty band for train
                        ax.fill_between(bin_centers, 
                                       hist_density_train - uncertainty_density_train,
                                       hist_density_train + uncertainty_density_train,
                                       step='mid', color=color, alpha=0.2)
                
                # --- VALIDATION data for this sample ---
                class_mask_val = (y_val_[:, class_idx] == 1)
                sample_mask_val = (proc_num_val == proc_num)
                mask_val = class_mask_val & sample_mask_val
                
                y_vals_val = y_pred_val_[mask_val, class_idx]
                weights_val = rel_w_val_[mask_val]
                weights_sq_val = weights_val ** 2
                
                if len(y_vals_val) > 0:
                    hist_raw_val, bin_edges_val = np.histogram(y_vals_val, bins=25, weights=weights_val, range=(0, 1))
                    hist_sq_raw_val, _ = np.histogram(y_vals_val, bins=bin_edges_val, weights=weights_sq_val, range=(0, 1))
                    bin_widths_val = np.diff(bin_edges_val)
                    
                    total_weight_val = np.sum(hist_raw_val)
                    if total_weight_val > 0:
                        hist_density_val = hist_raw_val / (total_weight_val * bin_widths_val)
                        uncertainty_density_val = np.sqrt(hist_sq_raw_val) / (total_weight_val * bin_widths_val)
                        bin_centers_val = 0.5 * (bin_edges_val[:-1] + bin_edges_val[1:])
                        
                        max_y = max(max_y, np.max(hist_density_val + uncertainty_density_val))
                        
                        # Error bars for validation
                        ax.errorbar(bin_centers_val, hist_density_val, yerr=uncertainty_density_val, 
                                   fmt='o', label=f'Valid {sample_name}', color=color, 
                                   markersize=6, capsize=2, elinewidth=1.5, alpha=0.8)
            
            # Labels and style
            ax.set_xlabel(f'{target_class_name} score', fontsize=12)
            ax.set_ylabel('a.u.', fontsize=12)
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-3, top=max_y * 100)
            ax.set_xlim(left=0, right=1)
            ax.set_title(f'Per-Sample Score Distributions: {target_class_name}', fontsize=14, fontweight='bold')
            
            ax.legend(ncol=2, fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(f'{path_for_plots}/{target_class_name}_per_sample_score.png', dpi=150)
            plt.close(fig)
            
            print(f"INFO: Per-sample validation plots for {target_class_name} completed!")
        else:
            print(f"WARNING: Class '{target_class_name}' has only {len(samples_in_class)} sample(s). Skipping per-sample plots.")


# ============================================================================
# Confusion Matrices
# ============================================================================
print("\nINFO: Generating confusion matrices...")

# Fixes problem with VBFToHH_sig in class_names when it is not present y predictions
# TODO: very hacky and will break if DNN out shape changes
if "VBFToHH_sig" in class_names and y_pred_val_.shape[1] <= 4:
    print("WARNING: 'VBFToHH_sig' found in class_names but not present in predictions. Removing from class_names for confusion matrix plotting.")
    class_names_pruned = [name for name in class_names if name != "VBFToHH_sig"]
else:
    class_names_pruned = class_names

# Validation confusion matrix
y_pred_val_labels = np.argmax(y_pred_val_, axis=1)
y_val_labels = np.argmax(y_val_, axis=1)
cm_val = confusion_matrix(y_val_labels, y_pred_val_labels)

# Plot validation confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
try:
    im = ax.imshow(cm_val, interpolation='auto', cmap=plt.cm.Blues)
except ValueError:
    # Fallback to 'nearest' interpolation if 'auto' fails (matplotlib 3.9 version compatibility)
    im = ax.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names_pruned, rotation=45, ha='right')
ax.set_yticklabels(class_names_pruned)

# Add text annotations
for i in range(n_classes):
    for j in range(n_classes):
        text = ax.text(j, i, f'{cm_val[i, j]:.0f}', ha="center", va="center", 
                      color="white" if cm_val[i, j] > cm_val.max() / 2 else "black", fontsize=11)

plt.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(f'{path_for_plots}/confusion_matrix_validation.png', dpi=150)
plt.close(fig)

# Training confusion matrix
y_pred_train_labels = np.argmax(y_pred_train, axis=1)
y_train_labels = np.argmax(y_train, axis=1)
cm_train = confusion_matrix(y_train_labels, y_pred_train_labels)

fig, ax = plt.subplots(figsize=(10, 8))
try:
    im = ax.imshow(cm_train, interpolation='auto', cmap=plt.cm.Blues)
except ValueError:
    # Fallback to 'nearest' interpolation if 'auto' fails (matplotlib 3.9 version compatibility)
    im = ax.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Training Set', fontsize=14, fontweight='bold')
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names_pruned, rotation=45, ha='right')
ax.set_yticklabels(class_names_pruned)

for i in range(n_classes):
    for j in range(n_classes):
        text = ax.text(j, i, f'{cm_train[i, j]:.0f}', ha="center", va="center", 
                      color="white" if cm_train[i, j] > cm_train.max() / 2 else "black", fontsize=11)

plt.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(f'{path_for_plots}/confusion_matrix_training.png', dpi=150)
plt.close(fig)

print("INFO: Confusion matrices completed!")

