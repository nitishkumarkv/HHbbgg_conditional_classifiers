"""
Plot ROC curves binned by mhh variable from prepared inputs.

This version accepts MULTIPLE base folders. The two (or more) folders are
expected to share the same sub-folder structure
(individual_samples/{era}/{sample}/events.parquet + y.npy). All events from
all folders are MERGED before plotting: if the same era/sample exists in more
than one folder, their events are simply concatenated.

This script reads, from each base folder:
- mhh variable, sample names, and weights from parquet files
  (individual_samples/{era}/{sample}/events.parquet)
- Predictions from y.npy in the same directory
- Class labels from sample_to_class_mapping.json (merged across folders)

Then generates ROC curves for each mhh bin, with automatic Run2/Run3 separation.

Usage:
    python plot_roc_mhh_binned_merged.py \
        --base_paths <path_to_inputs_A> <path_to_inputs_B> \
        --mhh_var_name nonResReg_M_X \
        --mhh_bins "250,350,450,600" \
        --output_path <optional_output_dir>
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pyarrow.parquet as pq

# Define which eras belong to which run
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["preEE", "postEE", "preBPix", "postBPix", "2024", "2025"]


def load_mhh_y_from_parquet(base_paths, mhh_var_name):
    """
    Load mhh variable, predictions, labels, weights, and era labels from
    parquet and y.npy files, MERGING all events across multiple base folders.

    Args:
        base_paths: List of base paths, each containing an individual_samples
                    folder and a sample_to_class_mapping.json. Events from all
                    folders are concatenated together.
        mhh_var_name: Name of the mhh variable column (e.g., "nonResReg_M_X")

    Returns:
        y_pred_val: Model predictions, shape (n_samples, n_classes)
        y_val: True labels (one-hot), shape (n_samples, n_classes)
        rel_w_val: Relative weights, shape (n_samples,)
        mhh_var_val: mhh values, shape (n_samples,)
        era_labels: Era labels, shape (n_samples,)
        sample_names: Sample names, shape (n_samples,)
        class_names: List of class names
    """
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    print(f"\nLoading data from {len(base_paths)} base folder(s):")
    for bp in base_paths:
        print(f"  - {bp}")

    # ------------------------------------------------------------------
    # Load and merge sample_to_class_mapping.json from every base folder
    # ------------------------------------------------------------------
    sample_to_class_mapping = {}
    for bp in base_paths:
        mapping_file = f"{bp}/sample_to_class_mapping.json"
        with open(mapping_file, 'r', encoding="utf-8") as f:
            this_mapping = json.load(f)
        # Warn on any conflicting class assignment for the same sample name
        for sample, cls in this_mapping.items():
            if sample in sample_to_class_mapping and sample_to_class_mapping[sample] != cls:
                print(
                    f"  WARNING: sample '{sample}' has conflicting class mapping "
                    f"('{sample_to_class_mapping[sample]}' vs '{cls}'); "
                    f"keeping '{cls}'"
                )
        sample_to_class_mapping.update(this_mapping)
        print(f"Loaded sample_to_class_mapping from {mapping_file}")

    # Class names - derived from the mapping
    class_names = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH", "VBFToHH_sig"]
    json_to_class = {
        "is_non_resonant_bkg": "non_resonant_bkg",
        "is_ttH_bkg": "ttH",
        "is_single_H_bkg": "other_single_H",
        "is_GluGluToHH_sig": "GluGluToHH",
        "is_VBFToHH_sig": "VBFToHH_sig"
    }

    # ------------------------------------------------------------------
    # Find all parquet files across every base folder
    # ------------------------------------------------------------------
    parquet_files = []
    for bp in base_paths:
        found = sorted(glob.glob(f"{bp}/individual_samples/*/*/events.parquet"))
        print(f"Found {len(found)} parquet files in {bp}")
        parquet_files.extend(found)

    if len(parquet_files) == 0:
        raise ValueError(
            "No parquet files found in any of "
            f"{[bp + '/individual_samples/*/*/events.parquet' for bp in base_paths]}"
        )

    print(f"Total parquet files across all folders: {len(parquet_files)}")

    # Load all data
    mhh_values_all = []
    y_pred_all = []
    y_val_all = []
    rel_w_all = []
    era_labels_all = []
    sample_names_all = []

    for pf in parquet_files:
        try:
            # Extract era and sample from path: individual_samples/{era}/{sample}/events.parquet
            path_parts = pf.split('/individual_samples/')[1].split('/')
            era = path_parts[0]
            sample_name = path_parts[1]
            sample_dir = os.path.dirname(pf)

            # Load parquet file
            table = pq.read_table(pf, columns=[mhh_var_name, 'weight_tot'])
            mhh_vals = table[mhh_var_name].to_numpy()
            # Create sample_names array with the extracted sample name
            sample_names = np.array([sample_name] * len(mhh_vals))

            try:
                weights = table['weight_tot'].to_numpy()
            except:
                # If 'weight_tot' column doesn't exist, use uniform weights
                weights = np.ones(len(mhh_vals))

            # Load predictions from y.npy
            y_pred_path = f"{sample_dir}/y.npy"
            y_pred = np.load(y_pred_path)

            # Verify dimensions match
            if len(mhh_vals) != len(y_pred):
                raise ValueError(f"Mismatch: {len(mhh_vals)} mhh values vs {len(y_pred)} predictions")

            # Create one-hot labels from sample_to_class_mapping
            n_classes = len(class_names)
            y_true = np.zeros((len(sample_names), n_classes), dtype=np.float32)

            for idx, sample_name in enumerate(sample_names):
                if sample_name not in sample_to_class_mapping:
                    raise ValueError(f"Sample '{sample_name}' not found in sample_to_class_mapping.json")

                json_class = sample_to_class_mapping[sample_name]
                model_class = json_to_class.get(json_class)

                if model_class is None:
                    raise ValueError(f"Unknown class mapping: {json_class}")

                class_idx = class_names.index(model_class)
                y_true[idx, class_idx] = 1

            mhh_values_all.extend(mhh_vals)
            y_pred_all.extend(y_pred)
            y_val_all.extend(y_true)
            rel_w_all.extend(weights)
            era_labels_all.extend([era] * len(mhh_vals))
            sample_names_all.extend(sample_names)
            print(f"  ✓ {era}/{os.path.basename(sample_dir)}: {len(mhh_vals)} events")
        except Exception as e:
            print(f"  ✗ {pf}: {e}")
            import traceback
            traceback.print_exc()

    if len(mhh_values_all) == 0:
        raise ValueError(f"Could not load any data from parquet/y.npy files")

    # Convert to numpy arrays
    mhh_var_val = np.array(mhh_values_all)
    y_pred_val = np.array(y_pred_all)
    y_val = np.array(y_val_all)
    rel_w_val = np.array(rel_w_all)
    era_labels_val = np.array(era_labels_all)
    sample_names_val = np.array(sample_names_all)

    print(f"\nTotal samples loaded (all folders merged): {len(mhh_var_val)}")
    print(f"  y_pred_val shape: {y_pred_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    print(f"  rel_w_val shape: {rel_w_val.shape}")
    print(f"  era_labels_val shape: {era_labels_val.shape}")
    print(f"  sample_names_val shape: {sample_names_val.shape}")

    # Print era distribution
    unique_eras, era_counts = np.unique(era_labels_val, return_counts=True)
    print(f"Era distribution:")
    for era, count in zip(unique_eras, era_counts):
        print(f"  {era}: {count} events ({100*count/len(era_labels_val):.1f}%)")

    return y_pred_val, y_val, rel_w_val, mhh_var_val, era_labels_val, sample_names_val, class_names


def plot_roc_curves_mhh_binned(
    y_pred_val, y_val, rel_w_val, mhh_var_val, mhh_bins,
    class_names, output_path, mhh_var_name, era_labels=None, sample_names=None
):
    """
    Generate ROC curves binned by mhh variable.

    Args:
        y_pred_val: Model predictions, shape (n_samples, n_classes)
        y_val: True labels, shape (n_samples, n_classes)
        rel_w_val: Relative weights, shape (n_samples,)
        mhh_var_val: mhh variable values, shape (n_samples,)
        mhh_bins: Bin edges as numpy array
        class_names: List of class names
        output_path: Where to save plots
        mhh_var_name: Name of the mhh variable for plot titles
    """
    os.makedirs(output_path, exist_ok=True)

    n_classes = y_val.shape[1]
    n_bins = len(mhh_bins) - 1

    print(f"\n{'='*70}")
    print(f"Generating ROC curves binned by {mhh_var_name}")
    print(f"Number of bins: {n_bins}")
    print(f"Bin edges: {mhh_bins}")
    print(f"{'='*70}\n")

    # Digitize to assign samples to bins
    bin_labels_val = np.digitize(mhh_var_val, mhh_bins) - 1

    # # ========================================================================
    # # 1. Individual class ROC curves (each class in separate subplots)
    # # ========================================================================
    # print("\nSection 1: Individual class ROC curves (all data)")

    # for class_idx in range(n_classes):
    #     class_name = class_names[class_idx]

    #     fig, axes = plt.subplots(n_bins, 1, figsize=(8, 5 * n_bins))
    #     if n_bins == 1:
    #         axes = [axes]

    #     roc_data = {}

    #     for bin_idx in range(n_bins):
    #         ax = axes[bin_idx]
    #         bin_mask = (bin_labels_val == bin_idx)
    #         n_in_bin = np.sum(bin_mask)
            
    #         if n_in_bin == 0:
    #             print(f"WARNING: No samples in {mhh_var_name} bin {bin_idx} [{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}]")
    #             ax.text(0.5, 0.5, 'No data in bin', ha='center', va='center', fontsize=12)
    #             continue
            
    #         y_true_binary = y_val[bin_mask, class_idx]
    #         y_score = y_pred_val[bin_mask, class_idx]
    #         weights = rel_w_val[bin_mask]
            
    #         # Compute ROC curve
    #         try:
    #             fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
    #             fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    #             roc_auc = auc(fpr, tpr)
    #         except Exception as e:
    #             print(f"ERROR in bin {bin_idx}: {e}")
    #             continue
            
    #         # Plot
    #         ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC = {roc_auc:.4f}')
    #         ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    #         ax.set_xlim([0.0, 1.0])
    #         ax.set_ylim([0.0, 1.05])
    #         ax.set_xlabel('False Positive Rate', fontsize=11)
    #         ax.set_ylabel('True Positive Rate', fontsize=11)
    #         ax.set_title(
    #             f'{mhh_var_name} bin: [{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}]',
    #             fontsize=12
    #         )
    #         ax.legend(loc="lower right", fontsize=11)
    #         ax.grid(True, alpha=0.3)
            
    #         # Store data
    #         bin_key = f"bin{bin_idx}_{mhh_bins[bin_idx]:.1f}_{mhh_bins[bin_idx+1]:.1f}"
    #         roc_data[bin_key] = {
    #             "fpr": list(fpr),
    #             "tpr": list(tpr),
    #             "auc": float(roc_auc),
    #             "n_samples": int(n_in_bin)
    #         }
            
    #         fig.tight_layout()
            
    #         # Save PNG
    #         outpath = f"{output_path}/roc_curve_{class_name}_vs_all_{mhh_var_name}_binned.png"
    #         fig.savefig(outpath, dpi=150)
    #         print(f"INFO: >>> {outpath}")
            
    #         # Save JSON data
    #         outpath_json = f"{output_path}/roc_curve_{class_name}_vs_all_{mhh_var_name}_binned.json"
    #         with open(outpath_json, 'w', encoding="utf-8") as f:
    #             json.dump(roc_data, f, indent=2)
    #         print(f"INFO: >>> {outpath_json}")
            
    #         plt.close(fig)
        
    # # ========================================================================
    # # 2. All classes in one bin (for comparing across classes within a bin)
    # # ========================================================================
    # fig, axes = plt.subplots(n_bins, 1, figsize=(10, 5 * n_bins))
    # if n_bins == 1:
    #     axes = [axes]
    
    # colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen', 'crimson']
    
    # for bin_idx in range(n_bins):
    #     ax = axes[bin_idx]
    #     bin_mask = (bin_labels_val == bin_idx)
    #     n_in_bin = np.sum(bin_mask)
        
    #     if n_in_bin == 0:
    #         ax.text(0.5, 0.5, 'No data in bin', ha='center', va='center', fontsize=12)
    #         continue
        
    #     for class_idx in range(n_classes):
    #         class_name = class_names[class_idx]
    #         y_true_binary = y_val[bin_mask, class_idx]
    #         y_score = y_pred_val[bin_mask, class_idx]
    #         weights = rel_w_val[bin_mask]
            
    #         try:
    #             fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
    #             fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    #             roc_auc = auc(fpr, tpr)
                
    #             ax.plot(fpr, tpr, color=colors[class_idx % len(colors)], linewidth=2,
    #                    label=f'{class_name} (AUC = {roc_auc:.4f})')
    #         except Exception as e:
    #             print(f"WARNING: Could not plot {class_name} in bin {bin_idx}: {e}")
        
    #     ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    #     ax.set_xlim([0.0, 1.0])
    #     ax.set_ylim([0.0, 1.05])
    #     ax.set_xlabel('False Positive Rate', fontsize=11)
    #     ax.set_ylabel('True Positive Rate', fontsize=11)
    #     ax.set_title(
    #         f'{mhh_var_name} bin {bin_idx}: [{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}] '
    #         f'({n_in_bin} events)',
    #         fontsize=12
    #     )
    #     ax.legend(loc="lower right", fontsize=10, ncol=2)
    #     ax.grid(True, alpha=0.3)
    
    # fig.tight_layout()
    # outpath = f"{output_path}/roc_curve_all_classes_{mhh_var_name}_binned.png"
    # fig.savefig(outpath, dpi=150)
    # print(f"INFO: >>> {outpath}")
    # plt.close(fig)
    
    # # ========================================================================
    # # 3. Generate separate plots for Run2 and Run3 (if era labels provided)
    # # ========================================================================
    # if era_labels is not None:
    #     print(f"\nSection 3: Run2/Run3 separated ROC curves")
        
    #     # Identify Run2 and Run3 samples
    #     run2_mask = np.isin(era_labels, RUN2_ERAS)
    #     run3_mask = np.isin(era_labels, RUN3_ERAS)
        
    #     for run_name, run_mask in [("Run2", run2_mask), ("Run3", run3_mask)]:
    #         n_run = np.sum(run_mask)
    #         if n_run == 0:
    #             print(f"WARNING: No {run_name} samples found")
    #             continue
            
    #         print(f"\n{run_name}: {n_run} samples ({100*n_run/len(era_labels):.1f}%)")
            
    #         # Create mhh bins for this run
    #         run_mhh = mhh_var_val[run_mask]
    #         run_bin_labels = np.digitize(run_mhh, mhh_bins) - 1
            
    #         # Plot each class separately for this run
    #         for class_idx in range(n_classes):
    #             class_name = class_names[class_idx]
                
    #             fig, axes = plt.subplots(n_bins, 1, figsize=(8, 5 * n_bins))
    #             if n_bins == 1:
    #                 axes = [axes]
                
    #             roc_data = {}
                
    #             for bin_idx in range(n_bins):
    #                 ax = axes[bin_idx]
    #                 bin_mask_run = (run_bin_labels == bin_idx)
    #                 n_in_bin = np.sum(bin_mask_run)
                    
    #                 if n_in_bin == 0:
    #                     ax.text(0.5, 0.5, 'No data in bin', ha='center', va='center', fontsize=12)
    #                     continue
                    
    #                 y_true_binary = y_val[run_mask, :][bin_mask_run, class_idx]
    #                 y_score = y_pred_val[run_mask, :][bin_mask_run, class_idx]
    #                 weights = rel_w_val[run_mask][bin_mask_run]
                    
    #                 try:
    #                     fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
    #                     fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    #                     roc_auc = auc(fpr, tpr)
    #                 except Exception as e:
    #                     print(f"ERROR in {run_name} bin {bin_idx}: {e}")
    #                     continue
                    
    #                 ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC = {roc_auc:.4f}')
    #                 ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    #                 ax.set_xlim([0.0, 1.0])
    #                 ax.set_ylim([0.0, 1.05])
    #                 ax.set_xlabel('False Positive Rate', fontsize=11)
    #                 ax.set_ylabel('True Positive Rate', fontsize=11)
    #                 ax.set_title(
    #                     f'{run_name} {mhh_var_name} bin {bin_idx}: [{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}] '
    #                     f'({n_in_bin} events)',
    #                     fontsize=12
    #                 )
    #                 ax.legend(loc="lower right", fontsize=11)
    #                 ax.grid(True, alpha=0.3)
                    
    #                 bin_key = f"bin{bin_idx}_{mhh_bins[bin_idx]:.1f}_{mhh_bins[bin_idx+1]:.1f}"
    #                 roc_data[bin_key] = {
    #                     "fpr": list(fpr),
    #                     "tpr": list(tpr),
    #                     "auc": float(roc_auc),
    #                     "n_samples": int(n_in_bin)
    #                 }
                
    #             fig.tight_layout()
                
    #             # Save PNG
    #             outpath = f"{output_path}/roc_curve_{class_name}_vs_all_{run_name}_{mhh_var_name}_binned.png"
    #             fig.savefig(outpath, dpi=150)
    #             print(f"INFO: >>> {outpath}")
                
    #             # Save JSON data
    #             outpath_json = f"{output_path}/roc_curve_{class_name}_vs_all_{run_name}_{mhh_var_name}_binned.json"
    #             with open(outpath_json, 'w', encoding="utf-8") as f:
    #                 json.dump(roc_data, f, indent=2)
    #             print(f"INFO: >>> {outpath_json}")
                
    #             plt.close(fig)
            
    #         # Plot all classes in one figure for this run
    #         fig, axes = plt.subplots(n_bins, 1, figsize=(10, 5 * n_bins))
    #         if n_bins == 1:
    #             axes = [axes]
            
    #         for bin_idx in range(n_bins):
    #             ax = axes[bin_idx]
    #             bin_mask_run = (run_bin_labels == bin_idx)
    #             n_in_bin = np.sum(bin_mask_run)
                
    #             if n_in_bin == 0:
    #                 ax.text(0.5, 0.5, 'No data in bin', ha='center', va='center', fontsize=12)
    #                 continue
                
    #             for class_idx in range(n_classes):
    #                 class_name = class_names[class_idx]
    #                 y_true_binary = y_val[run_mask, :][bin_mask_run, class_idx]
    #                 y_score = y_pred_val[run_mask, :][bin_mask_run, class_idx]
    #                 weights = rel_w_val[run_mask][bin_mask_run]
                    
    #                 try:
    #                     fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
    #                     fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    #                     roc_auc = auc(fpr, tpr)
                        
    #                     ax.plot(fpr, tpr, color=colors[class_idx % len(colors)], linewidth=2,
    #                            label=f'{class_name} (AUC = {roc_auc:.4f})')
    #                 except Exception as e:
    #                     print(f"WARNING: Could not plot {class_name} in {run_name} bin {bin_idx}: {e}")
                
    #             ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    #             ax.set_xlim([0.0, 1.0])
    #             ax.set_ylim([0.0, 1.05])
    #             ax.set_xlabel('False Positive Rate', fontsize=11)
    #             ax.set_ylabel('True Positive Rate', fontsize=11)
    #             ax.set_title(
    #                 f'{run_name} {mhh_var_name} bin {bin_idx}: [{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}] '
    #                 f'({n_in_bin} events)',
    #                 fontsize=12
    #             )
    #             ax.legend(loc="lower right", fontsize=10, ncol=2)
    #             ax.grid(True, alpha=0.3)
            
    #         fig.tight_layout()
    #         outpath = f"{output_path}/roc_curve_all_classes_{run_name}_{mhh_var_name}_binned.png"
    #         fig.savefig(outpath, dpi=150)
    #         print(f"INFO: >>> {outpath}")
    #         plt.close(fig)
    
    ## ========================================================================
    # 4. All ggHH benchmarks vs non_resonant_bkg, one figure per mhh bin
    #    Optionally separated by Run2/Run3 (if era labels provided)
    # ========================================================================
    if sample_names is not None:
        print("\nSection 4: All ggHH benchmarks vs non_resonant_bkg (by mhh bin)")

        idx_nonres = class_names.index("non_resonant_bkg")
        idx_gghh = class_names.index("GluGluToHH")

        # Find all unique ggHH benchmark/sample names
        gghh_sample_names = sorted(np.unique(sample_names[y_val[:, idx_gghh] == 1]))

        print(f"Found {len(gghh_sample_names)} ggHH benchmark/sample(s)")
        for s in gghh_sample_names:
            print(f"  - {s}")

        # Define colors for different samples
        colors = plt.cm.tab20(np.linspace(0, 1, len(gghh_sample_names)))

        # Determine which runs to plot
        run_configs = [("all", None)]
        if era_labels is not None:
            run2_mask = np.isin(era_labels, RUN2_ERAS)
            run3_mask = np.isin(era_labels, RUN3_ERAS)
            if np.sum(run2_mask) > 0:
                run_configs.append(("Run2", run2_mask))
            if np.sum(run3_mask) > 0:
                run_configs.append(("Run3", run3_mask))

        # Loop over runs
        for run_label, run_mask in run_configs:
            print(f"\n  --- {run_label} ---")

            # Select data for this run (or all if run_mask is None)
            if run_mask is None:
                run_bin_labels = bin_labels_val
                run_sample_names = sample_names
                run_y_val = y_val
                run_y_pred_val = y_pred_val
                run_rel_w_val = rel_w_val
            else:
                run_bin_labels = np.digitize(mhh_var_val[run_mask], mhh_bins) - 1
                run_sample_names = sample_names[run_mask]
                run_y_val = y_val[run_mask]
                run_y_pred_val = y_pred_val[run_mask]
                run_rel_w_val = rel_w_val[run_mask]

            # Create one figure per mhh bin for this run
            for bin_idx in range(n_bins):
                fig, ax = plt.subplots(figsize=(10, 8))

                roc_data = {}

                for sample_idx, gghh_sample in enumerate(gghh_sample_names):
                    # in this mhh bin
                    bin_mask = (run_bin_labels == bin_idx)

                    # keep only this ggHH benchmark OR nonres background
                    sample_mask = (
                        ((run_sample_names == gghh_sample) & (run_y_val[:, idx_gghh] == 1)) |
                        (run_y_val[:, idx_nonres] == 1)
                    )

                    subset_mask = bin_mask & sample_mask
                    n_in_bin = np.sum(subset_mask)

                    if n_in_bin == 0:
                        print(f"  WARNING: No data for {gghh_sample} in bin {bin_idx}")
                        continue

                    # truth: this ggHH sample = 1, nonres = 0
                    y_true_binary = ((run_sample_names[subset_mask] == gghh_sample) & (run_y_val[subset_mask, idx_gghh] == 1)).astype(int)

                    # safety check: need both positive and negative classes
                    if len(np.unique(y_true_binary)) < 2:
                        print(f"  WARNING: Only one class for {gghh_sample} in bin {bin_idx}")
                        continue

                    weights = run_rel_w_val[subset_mask]

                    # score: pairwise ggHH vs nonres score
                    eps = 1e-12
                    y_score = run_y_pred_val[subset_mask, idx_gghh] / (
                        run_y_pred_val[subset_mask, idx_gghh] + run_y_pred_val[subset_mask, idx_nonres] + eps
                    )

                    try:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_score, sample_weight=weights)
                        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
                        roc_auc = auc(fpr, tpr)
                    except Exception as e:
                        print(f"  WARNING: Could not compute ROC for {gghh_sample} in bin {bin_idx}: {e}")
                        continue

                    ax.plot(fpr, tpr, color=colors[sample_idx], linewidth=2.5,
                           label=f'{gghh_sample} (AUC = {roc_auc:.4f})')

                    bin_key = f"{gghh_sample}"
                    roc_data[bin_key] = {
                        "fpr": list(fpr),
                        "tpr": list(tpr),
                        "auc": float(roc_auc),
                        "n_samples": int(n_in_bin)
                    }

                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                # ax.set_title(
                #     f'{run_label} | ggHH vs non_resonant_bkg | {mhh_var_name} bin {bin_idx}: '
                #     f'[{mhh_bins[bin_idx]:.1f}, {mhh_bins[bin_idx+1]:.1f}]',
                #     fontsize=13
                # )
                ax.legend(loc="lower right", fontsize=14, ncol=1)
                ax.grid(True, alpha=0.3)

                fig.tight_layout()

                outpath = f"{output_path}/roc_curve_all_gghh_vs_non_resonant_bkg_{run_label}_{mhh_var_name}_bin{bin_idx}.png"
                fig.savefig(outpath, dpi=150)
                print(f"  INFO: >>> {outpath}")

                outpath_json = f"{output_path}/roc_curve_all_gghh_vs_non_resonant_bkg_{run_label}_{mhh_var_name}_bin{bin_idx}.json"
                with open(outpath_json, 'w', encoding="utf-8") as f:
                    json.dump(roc_data, f, indent=2)
                print(f"  INFO: >>> {outpath_json}")

                plt.close(fig)

    print(f"\n{'='*70}")
    print("ROC curve generation completed!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot ROC curves binned by mhh variable, merging multiple base folders'
    )
    parser.add_argument('--base_paths', type=str, nargs='+', required=True,
                       help='One or more base paths (each containing individual_samples '
                            'and sample_to_class_mapping.json). Events from all folders '
                            'are merged together. Example: --base_paths /pathA /pathB')
    parser.add_argument('--mhh_var_name', type=str, default="nonResReg_vbfpair_M_X",
                       help='Name of the mhh variable (e.g., nonResReg_M_X)')
    parser.add_argument('--mhh_bins', type=str, default="0,350,inf",
                       help='Comma-separated bin edges (e.g., "250,350,450,600")')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output directory. Default: '
                            '<first_base_path>/roc_plots_mhh_binned_merged/')

    args = parser.parse_args()

    base_paths = args.base_paths

    # Set output path
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = f"{base_paths[0]}/roc_plots_mhh_binned_merged/"

    # Parse bin edges
    try:
        mhh_bins = np.array([float(x) for x in args.mhh_bins.split(',')])
    except ValueError as e:
        print(f"ERROR: Invalid bin edges: {e}")
        return 1

    if len(mhh_bins) < 2:
        print("ERROR: Need at least 2 bin edges")
        return 1

    print("\n" + "="*70)
    print("ROC Curves with mhh Binning (merged folders)")
    print("="*70)
    print(f"Base paths:    {base_paths}")
    print(f"mhh variable:  {args.mhh_var_name}")
    print(f"Bin edges:     {mhh_bins}")
    print(f"Output path:   {output_path}")
    print("="*70 + "\n")

    try:
        # Load all data directly from parquet and y.npy files (merged across folders)
        y_pred_val, y_val, rel_w_val, mhh_var_val, era_labels_val, sample_names_val, class_names = load_mhh_y_from_parquet(
            base_paths, args.mhh_var_name
        )

        # Generate plots
        plot_roc_curves_mhh_binned(
            y_pred_val, y_val, rel_w_val, mhh_var_val, mhh_bins,
            class_names, output_path, args.mhh_var_name,
            era_labels=era_labels_val,
            sample_names=sample_names_val
        )

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
