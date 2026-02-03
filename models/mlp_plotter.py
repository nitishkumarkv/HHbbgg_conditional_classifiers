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
import mplhep as hep
from mlp import MLP
import pickle
import yaml


def load_class_names_from_config(config_path):
    """Load class names from training config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['classes']


def identify_signal_classes(class_names):
    """
    Identify signal classes for special ROC curve treatment.
    Returns dict with 'ggF' and 'VBF' keys, values are class names or None.
    """
    signal_classes = {'ggF': None, 'VBF': None}

    for class_name in class_names:
        # Check for ggF signal class
        if any(x in class_name.lower() for x in ['gluglutohh', 'ggfhh', 'gghh']):
            signal_classes['ggF'] = class_name
        # Check for VBF signal class
        elif any(x in class_name.lower() for x in ['vbfhh', 'vbftohh']):
            signal_classes['VBF'] = class_name

    return signal_classes


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
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to training config YAML file (to load class names)')
    args = parser.parse_args()

    #inputs_for_MLP = "../data/inputs_for_MLP_202411226/"
    #input_path="train_inputs_for_MLP_202411226/after_random_search_best1/"
    inputs_for_MLP = args.input_path
    input_path = f"{inputs_for_MLP}/after_random_search_best1/"
    path_for_plots = f"{input_path}/plots/"
    os.makedirs(path_for_plots, exist_ok=True)
    path_to_checkpoint = f"{input_path}/mlp.pth"

    # Load class names from config if provided, otherwise use default
    if args.config_path:
        class_names = load_class_names_from_config(args.config_path)
        print(f"Loaded class names from config: {class_names}")
    else:
        # Fallback to default (new scheme)
        class_names = ["non_b_bkg", "b_bkg", "VBFHH", "ggFHH"]
        print(f"WARNING: No config_path provided. Using default class names: {class_names}")
        print(f"         Consider using --config_path to ensure correct class names.")

    # Identify signal classes for ROC curves
    signal_classes = identify_signal_classes(class_names)
    print(f"Identified signal classes: {signal_classes}")

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

    # Verify class names match the data
    n_classes = y_val.shape[1]
    if len(class_names) != n_classes:
        raise ValueError(f"Number of class names ({len(class_names)}) doesn't match data ({n_classes})")
    print(f"Number of classes: {n_classes}")

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


    # One-vs-One ROC Curves for ggF signal class (if exists)
    if signal_classes['ggF'] is not None:
        glu_idx = class_names.index(signal_classes['ggF'])

        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]

        # Iterate over ggF vs each other class individually
        plt.figure(figsize=(8, 6))

        GluGluToHH_one_vs_one_roc = {}
        for j in other_classes:
            i = glu_idx
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

            # store the AUC
            GluGluToHH_one_vs_one_roc[f"{class_name_j}_fpr"] = fpr
            GluGluToHH_one_vs_one_roc[f"{class_name_j}_tpr"] = tpr

        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')

        plt.xlim([0.0001, 1.0])
        plt.xscale('log')
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual_logx.png')
        plt.clf()

        # save the auc scores
        with open(f'{path_for_plots}/ggF_vs_all.json', 'w') as f:
            json.dump(GluGluToHH_one_vs_one_roc, f)

    # One-vs-One ROC Curves for VBF signal class (if exists)
    if signal_classes['VBF'] is not None:
        vbf_idx = class_names.index(signal_classes['VBF'])

        VBFToHH_one_vs_one_roc = {}
        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != vbf_idx]

        # Iterate over VBF vs each other class individually
        plt.figure(figsize=(8, 6))

        for j in other_classes:
            i = vbf_idx
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

            # store the AUC
            VBFToHH_one_vs_one_roc[f"{class_name_j}_fpr"] = fpr
            VBFToHH_one_vs_one_roc[f"{class_name_j}_tpr"] = tpr

        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
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
        with open(f'{path_for_plots}/VBF_vs_all.json', 'w') as f:
            json.dump(VBFToHH_one_vs_one_roc, f)

    # Training set ROC curves
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

    # Training set: VBF one-vs-one ROC curves (if exists)
    if signal_classes['VBF'] is not None:
        vbf_idx = class_names.index(signal_classes['VBF'])

        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != vbf_idx]

        # Iterate over VBF vs each other class individually
        plt.figure(figsize=(8, 6))

        for j in other_classes:
            i = vbf_idx
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]
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
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
        plt.close()

    # Training set: ggF one-vs-one ROC curves (if exists)
    if signal_classes['ggF'] is not None:
        glu_idx = class_names.index(signal_classes['ggF'])

        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]

        # Iterate over ggF vs each other class individually
        plt.figure(figsize=(8, 6))

        for j in other_classes:
            i = glu_idx
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]
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

    # Ensure we have enough colors for all classes
    if n_classes > len(colours):
        # Generate additional colors if needed
        import matplotlib.cm as cm
        additional_colors = cm.get_cmap('tab10')(range(len(colours), n_classes))
        colours.extend([tuple(c) for c in additional_colors])

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
