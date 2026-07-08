"""
mlp_plotter_2fold.py
====================

Companion to ``mlp_plotter.py`` that reads in **two (or more) output folders at
once** and produces combined plots:

  * ROC curves    -> one figure per ROC type, with one curve per input folder.
                     Same physics class is drawn in the SAME color; different
                     folders are distinguished by different LINE STYLES.

  * Score distributions -> the data of all input folders is MERGED together and
                     a single histogram is produced for each score (output node).

It expects exactly the same per-folder layout produced by the training, i.e.
for each ``--input_paths`` entry ``F``:

    F/after_random_search_best1/y_pred_train.npy
    F/after_random_search_best1/y_pred_val.npy
    F/y_train.npy
    F/y_val.npy
    F/true_class_weights.npy        (train weights)
    F/class_weights_for_val.npy     (val weights)

Example
-------
    python mlp_plotter_2fold.py \
        --input_paths out_..._trainEven out_..._trainOdd \
        --labels even odd \
        --output_path out_..._2fold_plots
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import roc_curve, auc

hep.style.use("CMS")

# COM_ENERGY = 13 / 13.6

# ----------------------------------------------------------------------------
# Class configuration -- kept identical to mlp_plotter.py so inputs/outputs match
# ----------------------------------------------------------------------------
CLASS_NAMES = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH"]  # , "VBFToHH_sig"]

# One stable color per physics class (index-aligned with CLASS_NAMES).
# Taken from the active (CMS) style's default color cycle so that ROC colors
# match mlp_plotter.py exactly -- there each class is drawn in cycle order with
# no explicit color, i.e. class i -> i-th color of the prop_cycle.
CLASS_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Line styles cycled per input folder (so each folder is visually distinct).
FOLDER_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_folder(folder):
    """Load predictions, truth and weights for one output folder.

    Returns a dict with train/val arrays. ``input_path`` mirrors the layout used
    in mlp_plotter.py (predictions live under ``after_random_search_best1``).
    """
    input_path = os.path.join(folder, "after_random_search_best1")

    data = {
        "y_pred_val": np.load(os.path.join(input_path, "y_pred_val.npy")),
        "y_val": np.load(os.path.join(folder, "y_val.npy")),
        "rel_w_val": np.load(os.path.join(folder, "class_weights_for_val.npy")),
        "y_pred_train": np.load(os.path.join(input_path, "y_pred_train.npy")),
        "y_train": np.load(os.path.join(folder, "y_train.npy")),
        "rel_w_train": np.load(os.path.join(folder, "true_class_weights.npy")),
    }
    # "all" split = train + val concatenated (full dataset for this folder).
    data["y_pred_all"] = np.vstack([data["y_pred_train"], data["y_pred_val"]])
    data["y_all"] = np.vstack([data["y_train"], data["y_val"]])
    data["rel_w_all"] = np.concatenate([data["rel_w_train"], data["rel_w_val"]])
    print(f"INFO: loaded '{folder}' "
          f"(val: {data['y_val'].shape[0]} ev, train: {data['y_train'].shape[0]} ev)")
    return data


# ----------------------------------------------------------------------------
# ROC: one figure per ROC type, one curve per folder, color == class
# ----------------------------------------------------------------------------
def plot_roc_one_vs_all(folders_data, labels, n_classes, path_for_plots,
                        split="val"):
    """One-vs-all ROC. For every class i, draw one curve per folder.

    Same class -> same color; different folder -> different linestyle.
    """
    y_pred_key = f"y_pred_{split}"
    y_true_key = f"y_{split}"
    w_key = f"rel_w_{split}"

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_dict = {}

    for i in range(n_classes):
        class_name = CLASS_NAMES[i]
        color = CLASS_COLORS[i % len(CLASS_COLORS)]

        for f_idx, (data, label) in enumerate(zip(folders_data, labels)):
            linestyle = FOLDER_LINESTYLES[f_idx % len(FOLDER_LINESTYLES)]

            y_true_binary = data[y_true_key][:, i]
            y_score = data[y_pred_key][:, i]
            weights = data[w_key]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2,
                    label=f"{class_name} [{label}] (AUC={roc_auc:0.4f})")

            auc_dict[f"{class_name}_{label}_fpr"] = list(fpr)
            auc_dict[f"{class_name}_{label}_tpr"] = list(tpr)
            auc_dict[f"{class_name}_{label}_auc"] = float(roc_auc)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR", fontsize=14)
    ax.set_ylabel("TPR", fontsize=14)
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, length=8)
    ax.minorticks_on()
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    # hep.cms.label(data=False, ax=ax, loc=0,
    #               fontsize=15,)
    plt.tight_layout()

    outpath = os.path.join(path_for_plots, f"roc_one_vs_all_{split}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"INFO: >>> {outpath}")

    ax.set_xlim([1e-4, 1.0])
    ax.set_xscale("log")
    outpath = os.path.join(path_for_plots, f"roc_one_vs_all_{split}_logx.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"INFO: >>> {outpath}")
    plt.close(fig)

    outpath = os.path.join(path_for_plots, f"roc_one_vs_all_{split}.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(auc_dict, fh, indent=2)
    print(f"INFO: >>> {outpath}")


def plot_roc_signal_vs_each(folders_data, labels, n_classes, path_for_plots,
                            signal_class="GluGluToHH", split="val"):
    """One-vs-one ROC: signal vs each background, one curve per (bkg, folder).

    Color == background class; linestyle == folder.
    """
    if signal_class not in CLASS_NAMES:
        return
    sig_idx = CLASS_NAMES.index(signal_class)
    other_classes = [i for i in range(n_classes) if i != sig_idx]

    y_pred_key = f"y_pred_{split}"
    y_true_key = f"y_{split}"
    w_key = f"rel_w_{split}"

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_dict = {}

    for j in other_classes:
        color = CLASS_COLORS[j % len(CLASS_COLORS)]
        for f_idx, (data, label) in enumerate(zip(folders_data, labels)):
            linestyle = FOLDER_LINESTYLES[f_idx % len(FOLDER_LINESTYLES)]

            y_true = data[y_true_key]
            idx = (y_true[:, sig_idx] == 1) | (y_true[:, j] == 1)
            y_true_binary = y_true[idx, sig_idx]
            y_score = data[y_pred_key][idx, sig_idx]
            weights = data[w_key][idx]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2,
                    label=f"vs {CLASS_NAMES[j]} [{label}] (AUC={roc_auc:0.4f})")

            auc_dict[f"{signal_class}_vs_{CLASS_NAMES[j]}_{label}_auc"] = float(roc_auc)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR", fontsize=14)
    ax.set_ylabel("TPR", fontsize=14)
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, length=8)
    ax.minorticks_on()
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    # hep.cms.label(data=False, ax=ax, loc=0,
    #               fontsize=15,)
    plt.tight_layout()

    outpath = os.path.join(path_for_plots, f"roc_{signal_class}_vs_each_{split}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"INFO: >>> {outpath}")

    ax.set_xlim([1e-4, 1.0])
    ax.set_xscale("log")
    outpath = os.path.join(path_for_plots, f"roc_{signal_class}_vs_each_{split}_logx.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"INFO: >>> {outpath}")
    plt.close(fig)

    outpath = os.path.join(path_for_plots, f"roc_{signal_class}_vs_each_{split}.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(auc_dict, fh, indent=2)
    print(f"INFO: >>> {outpath}")


# ----------------------------------------------------------------------------
# Score distributions: MERGE all folders, one figure per score (output node)
# Drawn EXACTLY like mlp_plotter.py: train = step + shaded band, val = dots with
# error bars. The only difference is that the two (2-fold) trainings are merged.
# ----------------------------------------------------------------------------
# Same palette as mlp_plotter.py so colors match one-to-one.
SCORE_COLOURS = ['blue', 'red', 'green', 'orange', 'purple']


def plot_merged_score_distributions(folders_data, n_classes, path_for_plots,
                                    bins=25):
    """Merge the folds and draw one figure per score, identical to mlp_plotter.py.

    Train and validation are merged *separately* across folders (so a 2-fold run
    yields the full train sample and the full val sample). For each score node
    ``i`` we overlay, per true class ``j``:
      * Train -> step line + shaded uncertainty band
      * Valid -> dots with error bars
    Histograms are area-normalized (density).
    """
    # Merge train across folders, and val across folders -- kept separate.
    y_pred_train = np.vstack([d["y_pred_train"] for d in folders_data])
    y_train = np.vstack([d["y_train"] for d in folders_data])
    rel_w_train = np.concatenate([d["rel_w_train"] for d in folders_data])

    y_pred_val = np.vstack([d["y_pred_val"] for d in folders_data])
    y_val = np.vstack([d["y_val"] for d in folders_data])
    rel_w_val = np.concatenate([d["rel_w_val"] for d in folders_data])

    colours = SCORE_COLOURS

    for i in range(n_classes):
        fig, ax = plt.subplots(figsize=(8, 6))
        class_name = CLASS_NAMES[i]

        max_y = 0  # Track max y for ylim

        # --- TRAIN: step plot with shaded uncertainty ---
        for j in range(n_classes):
            mask = y_train[:, j] == 1
            y_vals = y_pred_train[mask, i]
            weights = rel_w_train[mask]
            weights_sq = weights**2

            hist_raw, bin_edges = np.histogram(y_vals, bins=bins, weights=weights, range=(0, 1))
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
                label=f'Train {CLASS_NAMES[j]}',
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
            mask = y_val[:, j] == 1
            y_vals = y_pred_val[mask, i]
            weights = rel_w_val[mask]
            weights_sq = weights**2

            hist_raw, bin_edges = np.histogram(y_vals, bins=bins, weights=weights, range=(0, 1))
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
                label=f'Valid {CLASS_NAMES[j]}',
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

        fig.tight_layout()
        outpath = os.path.join(path_for_plots, f"{class_name}_score.png")
        fig.savefig(outpath)
        print(f"INFO: >>> {outpath}")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined MLP plots over 2+ output folders.")
    parser.add_argument("--input_paths", type=str, nargs="+", required=True,
                        help="Two or more output folders (each with the standard layout).")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Legend label per folder (defaults to basename).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to write plots (default: <first folder>/plots_2fold).")
    args = parser.parse_args()

    if len(args.input_paths) < 2:
        parser.error("Provide at least two folders to --input_paths.")

    if args.labels is None:
        labels = [os.path.basename(os.path.normpath(p)) for p in args.input_paths]
    else:
        if len(args.labels) != len(args.input_paths):
            parser.error("--labels must have the same length as --input_paths.")
        labels = args.labels

    path_for_plots = args.output_path or os.path.join(
        os.path.normpath(args.input_paths[0]), "plots_2fold")
    os.makedirs(path_for_plots, exist_ok=True)

    folders_data = [load_folder(p) for p in args.input_paths]

    # n_classes from the prediction shape (robust to 3- vs 4-class configs).
    n_classes = folders_data[0]["y_val"].shape[1]
    if n_classes > len(CLASS_NAMES):
        raise ValueError(f"n_classes={n_classes} exceeds configured CLASS_NAMES "
                         f"({CLASS_NAMES}); update CLASS_NAMES.")

    print(f"\nINFO: {len(folders_data)} folders, n_classes={n_classes}")
    print(f"INFO: labels = {labels}")
    print(f"INFO: output -> {path_for_plots}\n")

    # ROC overlays (validation, training, and the full train+val dataset)
    for split in ("val", "train", "all"):
        plot_roc_one_vs_all(folders_data, labels, n_classes, path_for_plots, split)
        plot_roc_signal_vs_each(folders_data, labels, n_classes, path_for_plots,
                                signal_class="GluGluToHH", split=split)

    # Merged score distributions (one figure per output node)
    plot_merged_score_distributions(folders_data, n_classes, path_for_plots)

    print("\nINFO: done.")
