"""
Plot ROC curves binned by an mHH variable from prepared inputs.

The script reads per-sample parquet files and ``y.npy`` predictions, then
compares every configured ggHH benchmark with the non-resonant background.
For each mHH bin it writes both a linear ROC plot and a log-x version.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import auc, roc_curve


hep.style.use("CMS")

# Define which eras belong to which run.
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["preEE", "postEE", "preBPix", "postBPix", "2024", "2025"]

# Explicit display labels for the ggHH samples used in this analysis.  The
# tuple order follows the legend title: (kappa_lambda, kappa_t, C_2).
GGHH_BENCHMARK_LABELS = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "(1, 1, 0)",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "(0, 1, 0)",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "(2.45, 1, 0)",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "(5, 1, 0)",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": "(1, 1, 0.1)",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": "(1, 1, 0.35)",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": "(1, 1, 3)",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": "(1, 1, -2)",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": "(0, 1, 1)",
    "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": "(-20, 1, 2.24)",
}


def load_mhh_y_from_parquet(base_path, mhh_var_name):
    """Load mHH, predictions, labels, weights, eras, and sample names."""
    print(f"\nLoading data from parquet and y.npy files in {base_path}")

    mapping_file = f"{base_path}/sample_to_class_mapping.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        sample_to_class_mapping = json.load(f)
    print(f"Loaded sample_to_class_mapping from {mapping_file}")

    class_names = [
        "non_resonant_bkg",
        "ttH",
        "other_single_H",
        "GluGluToHH",
        "VBFToHH_sig",
    ]
    json_to_class = {
        "is_non_resonant_bkg": "non_resonant_bkg",
        "is_ttH_bkg": "ttH",
        "is_single_H_bkg": "other_single_H",
        "is_GluGluToHH_sig": "GluGluToHH",
        "is_VBFToHH_sig": "VBFToHH_sig",
    }

    parquet_files = sorted(
        glob.glob(f"{base_path}/individual_samples/*/*/events.parquet")
    )
    if not parquet_files:
        raise ValueError(
            f"No parquet files found in "
            f"{base_path}/individual_samples/*/*/events.parquet"
        )
    print(f"Found {len(parquet_files)} parquet files")

    mhh_values_all = []
    y_pred_all = []
    y_val_all = []
    rel_w_all = []
    era_labels_all = []
    sample_names_all = []

    for parquet_file in parquet_files:
        try:
            path_parts = parquet_file.split("/individual_samples/")[1].split("/")
            era = path_parts[0]
            sample_name = path_parts[1]
            sample_dir = os.path.dirname(parquet_file)

            table = pq.read_table(
                parquet_file, columns=[mhh_var_name, "weight_tot"]
            )
            mhh_values = table[mhh_var_name].to_numpy()
            sample_names = np.full(len(mhh_values), sample_name, dtype=object)

            try:
                weights = table["weight_tot"].to_numpy()
            except Exception:
                weights = np.ones(len(mhh_values))

            y_pred = np.load(f"{sample_dir}/y.npy")
            if len(mhh_values) != len(y_pred):
                raise ValueError(
                    f"Mismatch: {len(mhh_values)} mhh values vs "
                    f"{len(y_pred)} predictions"
                )

            y_true = np.zeros(
                (len(sample_names), len(class_names)), dtype=np.float32
            )
            if sample_name not in sample_to_class_mapping:
                raise ValueError(
                    f"Sample '{sample_name}' not found in "
                    "sample_to_class_mapping.json"
                )

            json_class = sample_to_class_mapping[sample_name]
            model_class = json_to_class.get(json_class)
            if model_class is None:
                raise ValueError(f"Unknown class mapping: {json_class}")
            y_true[:, class_names.index(model_class)] = 1

            mhh_values_all.extend(mhh_values)
            y_pred_all.extend(y_pred)
            y_val_all.extend(y_true)
            rel_w_all.extend(weights)
            era_labels_all.extend([era] * len(mhh_values))
            sample_names_all.extend(sample_names)
            print(f"  ✓ {era}/{os.path.basename(sample_dir)}: {len(mhh_values)} events")
        except Exception as error:
            print(f"  ✗ {parquet_file}: {error}")
            import traceback

            traceback.print_exc()

    if not mhh_values_all:
        raise ValueError("Could not load any data from parquet/y.npy files")

    mhh_var_val = np.asarray(mhh_values_all)
    y_pred_val = np.asarray(y_pred_all)
    y_val = np.asarray(y_val_all)
    rel_w_val = np.asarray(rel_w_all)
    era_labels_val = np.asarray(era_labels_all)
    sample_names_val = np.asarray(sample_names_all)

    print(f"\nTotal samples loaded: {len(mhh_var_val)}")
    print(f"  y_pred_val shape: {y_pred_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    print(f"  rel_w_val shape: {rel_w_val.shape}")
    print(f"  era_labels_val shape: {era_labels_val.shape}")
    print(f"  sample_names_val shape: {sample_names_val.shape}")

    unique_eras, era_counts = np.unique(era_labels_val, return_counts=True)
    print("Era distribution:")
    for era, count in zip(unique_eras, era_counts):
        fraction = 100 * count / len(era_labels_val)
        print(f"  {era}: {count} events ({fraction:.1f}%)")

    return (
        y_pred_val,
        y_val,
        rel_w_val,
        mhh_var_val,
        era_labels_val,
        sample_names_val,
        class_names,
    )


def plot_roc_curves_mhh_binned(
    y_pred_val,
    y_val,
    rel_w_val,
    mhh_var_val,
    mhh_bins,
    class_names,
    output_path,
    mhh_var_name,
    era_labels=None,
    sample_names=None,
):
    """Plot all ggHH benchmarks against non-resonant background by mHH bin."""
    os.makedirs(output_path, exist_ok=True)

    n_bins = len(mhh_bins) - 1
    bin_labels_val = np.digitize(mhh_var_val, mhh_bins) - 1

    print(f"\n{'=' * 70}")
    print(f"Generating ROC curves binned by {mhh_var_name}")
    print(f"Number of bins: {n_bins}")
    print(f"Bin edges: {mhh_bins}")
    print(f"{'=' * 70}\n")

    if sample_names is None:
        print("WARNING: sample_names were not provided; no plots were generated")
        return

    print("\nSection 4: All ggHH benchmarks vs non_resonant_bkg (by mhh bin)")

    idx_nonres = class_names.index("non_resonant_bkg")
    idx_gghh = class_names.index("GluGluToHH")
    gghh_sample_names = sorted(np.unique(sample_names[y_val[:, idx_gghh] == 1]))

    print(f"Found {len(gghh_sample_names)} ggHH benchmark/sample(s)")
    for sample_name in gghh_sample_names:
        print(f"  - {sample_name}")
        if sample_name not in GGHH_BENCHMARK_LABELS:
            print(
                "    WARNING: No display label configured; "
                "the full sample name will be used"
            )

    colors = plt.cm.tab20(np.linspace(0, 1, len(gghh_sample_names)))

    run_configs = [("all", None)]
    if era_labels is not None:
        run2_mask = np.isin(era_labels, RUN2_ERAS)
        run3_mask = np.isin(era_labels, RUN3_ERAS)
        if np.sum(run2_mask) > 0:
            run_configs.append(("Run2", run2_mask))
        if np.sum(run3_mask) > 0:
            run_configs.append(("Run3", run3_mask))

    for run_label, run_mask in run_configs:
        print(f"\n  --- {run_label} ---")

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

        for bin_idx in range(n_bins):
            fig, ax = plt.subplots(figsize=(8, 6))
            roc_data = {}

            for sample_idx, gghh_sample in enumerate(gghh_sample_names):
                bin_mask = run_bin_labels == bin_idx
                sample_mask = (
                    (
                        (run_sample_names == gghh_sample)
                        & (run_y_val[:, idx_gghh] == 1)
                    )
                    | (run_y_val[:, idx_nonres] == 1)
                )
                subset_mask = bin_mask & sample_mask
                n_in_bin = np.sum(subset_mask)

                if n_in_bin == 0:
                    print(
                        f"  WARNING: No data for {gghh_sample} "
                        f"in bin {bin_idx}"
                    )
                    continue

                y_true_binary = (
                    (run_sample_names[subset_mask] == gghh_sample)
                    & (run_y_val[subset_mask, idx_gghh] == 1)
                ).astype(int)
                if len(np.unique(y_true_binary)) < 2:
                    print(
                        f"  WARNING: Only one class for {gghh_sample} "
                        f"in bin {bin_idx}"
                    )
                    continue

                weights = run_rel_w_val[subset_mask]
                eps = 1e-12
                y_score = run_y_pred_val[subset_mask, idx_gghh] / (
                    run_y_pred_val[subset_mask, idx_gghh]
                    + run_y_pred_val[subset_mask, idx_nonres]
                    + eps
                )

                try:
                    fpr, tpr, _ = roc_curve(
                        y_true_binary, y_score, sample_weight=weights
                    )
                    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
                    roc_auc = auc(fpr, tpr)
                except Exception as error:
                    print(
                        f"  WARNING: Could not compute ROC for {gghh_sample} "
                        f"in bin {bin_idx}: {error}"
                    )
                    continue

                benchmark_label = GGHH_BENCHMARK_LABELS.get(
                    gghh_sample, gghh_sample
                )
                ax.plot(
                    fpr,
                    tpr,
                    color=colors[sample_idx],
                    linewidth=2,
                    label=f"{benchmark_label} (AUC={roc_auc:0.4f})",
                )

                roc_data[gghh_sample] = {
                    "fpr": list(fpr),
                    "tpr": list(tpr),
                    "auc": float(roc_auc),
                    "n_samples": int(n_in_bin),
                }

            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("FPR", fontsize=14)
            ax.set_ylabel("TPR", fontsize=14)
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=True,
                length=8,
            )
            ax.minorticks_on()
            legend = ax.legend(
                loc="lower right",
                fontsize=11,
                ncol=1,
                title=r"ggHH ($\kappa_\lambda$, $\kappa_t$, $C_2$)",
            )
            legend.get_title().set_fontsize(13)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            output_stem = (
                f"{output_path}/roc_curve_all_gghh_vs_non_resonant_bkg_"
                f"{run_label}_{mhh_var_name}_bin{bin_idx}"
            )
            linear_outpath = f"{output_stem}.png"
            fig.savefig(
                linear_outpath, dpi=150, bbox_inches="tight"
            )
            print(f"  INFO: >>> {linear_outpath}")

            ax.set_xlim([1e-4, 1.0])
            ax.set_xscale("log")
            logx_outpath = f"{output_stem}_logx.png"
            fig.savefig(logx_outpath, dpi=150, bbox_inches="tight")
            print(f"  INFO: >>> {logx_outpath}")

            json_outpath = f"{output_stem}.json"
            with open(json_outpath, "w", encoding="utf-8") as f:
                json.dump(roc_data, f, indent=2)
            print(f"  INFO: >>> {json_outpath}")

            plt.close(fig)

    print(f"\n{'=' * 70}")
    print("ROC curve generation completed!")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ROC curves binned by mhh variable"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help=(
            "Path to inputs containing individual_samples and "
            "sample_to_class_mapping.json"
        ),
    )
    parser.add_argument(
        "--mhh_var_name",
        type=str,
        default="nonResReg_vbfpair_M_X",
        help="Name of the mhh variable (e.g. nonResReg_M_X)",
    )
    parser.add_argument(
        "--mhh_bins",
        type=str,
        default="0,350,inf",
        help='Comma-separated bin edges (e.g. "250,350,450,600")',
    )
    args = parser.parse_args()

    output_path = f"{args.base_path}/roc_plots_mhh_binned/"
    try:
        mhh_bins = np.array([float(value) for value in args.mhh_bins.split(",")])
    except ValueError as error:
        print(f"ERROR: Invalid bin edges: {error}")
        return 1

    if len(mhh_bins) < 2:
        print("ERROR: Need at least 2 bin edges")
        return 1

    print("\n" + "=" * 70)
    print("ROC Curves with mhh Binning")
    print("=" * 70)
    print(f"Path:          {args.base_path}")
    print(f"mhh variable:  {args.mhh_var_name}")
    print(f"Bin edges:     {mhh_bins}")
    print(f"Output path:   {output_path}")
    print("=" * 70 + "\n")

    try:
        loaded_data = load_mhh_y_from_parquet(args.base_path, args.mhh_var_name)
        (
            y_pred_val,
            y_val,
            rel_w_val,
            mhh_var_val,
            era_labels_val,
            sample_names_val,
            class_names,
        ) = loaded_data

        plot_roc_curves_mhh_binned(
            y_pred_val,
            y_val,
            rel_w_val,
            mhh_var_val,
            mhh_bins,
            class_names,
            output_path,
            args.mhh_var_name,
            era_labels=era_labels_val,
            sample_names=sample_names_val,
        )
        return 0
    except Exception as error:
        print(f"\nERROR: {error}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
