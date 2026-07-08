import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os
import mplhep as hep

plt.style.use(hep.style.CMS)

# ============================================================
#  Column names  ← adjust here if your parquet uses other names
# ============================================================
COL_SAMPLE = "sample"
COL_MASS   = "mass"
COL_DIJET  = "dijet_mass"
COL_WEIGHT = "weight_tot"

# ============================================================
#  Sample groups  ← adjust as needed
# ============================================================
SAMPLE_GROUPS = {
    "GGJets":      ["GGJets"],
    # "GGJets+TTGG": ["TTGG", "GGJets"],
    "ttHToGG":     ["ttHToGG"],
    "ggHToGG":     ["GluGluHToGG"],
}

PLOT_VARS = [
    # (column,    save_name,    x-label,                  plot range)
    (COL_MASS,  "mass",       "di-photon mass [GeV]",     (100, 180)),
    (COL_DIJET, "dijet_mass", "di-jet mass DNNreg [GeV]", (80,  190)),
]

BINS = 18

COLORS = {
    "Inclusive": "black",
    "ggHH-lowMhh-1":  "blue",
    "ggHH-lowMhh-2":  "red",
    "ggHH-highMhh-1": "blue",
    "ggHH-highMhh-2": "red",
    "ggHH-highMhh-3": "green",
    "VBFHH-lowMhh":   "blue",
    "VBFHH-highMhh":  "red",
    "ttH-lep-1":      "blue",
    "ttH-lep-2":      "red",
    "ttH-lep-3":      "green",
    "ttH-had-1":      "blue",
    "ttH-had-2":      "red",
    "ttH-had-3":      "green",
}
# ============================================================
#  Plotting helper — unchanged from original
# ============================================================
def plot_with_errorbars(data, weights, bins, range_, label, ax, color, inclusive=False):
    hist, bin_edges = np.histogram(data, bins=bins, range=range_, weights=weights)
    sumw2, _        = np.histogram(data, bins=bins, range=range_, weights=weights**2)

    bin_widths  = np.diff(bin_edges)
    norm_factor = np.sum(hist * bin_widths)
    if norm_factor > 0:
        hist  /= norm_factor
        sumw2 /= norm_factor**2

    hep.histplot(
        hist, bin_edges,
        yerr=np.sqrt(sumw2),
        label=label,
        histtype="step",
        ax=ax,
        linewidth=4 if inclusive else 2,
        color=color
    )


# ============================================================
#  Apply SR cuts from JSON using pandas .query()
#  Returns dict: {sr_name -> boolean np.ndarray of length len(df)}
# ============================================================
def apply_cuts(df, sr_cuts, sequential):
    """
    sequential=True  → each event can enter at most one SR
                        (first matching SR wins; same as original script logic)
    sequential=False → SRs are independent (overlap possible)
    """
    masks     = {}
    remaining = pd.Series(True, index=df.index)   # all events eligible at start

    for sr_name, cut_str in sr_cuts.items():
        try:
            if sequential:
                # only query events still available
                matched_idx = df[remaining].query(cut_str).index
            else:
                matched_idx = df.query(cut_str).index
        except Exception as e:
            print(f"[WARN] Could not evaluate cut for '{sr_name}': {e}")
            matched_idx = pd.Index([])

        mask = pd.Series(False, index=df.index)
        mask[matched_idx] = True
        masks[sr_name] = mask.values.copy()

        if sequential:
            remaining &= ~mask

    return masks


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sculpting check — reads a flat parquet + JSON SR definitions"
    )
    parser.add_argument(
        "--input_parquet", type=str, nargs="+", required=True,
        help="One or more parquet files (same schema, concatenated automatically)",
    )
    parser.add_argument(
        "--cats_json", type=str, required=None,
        help="JSON file mapping SR name -> pandas-query cut string",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Root output directory; per-group sub-dirs are created automatically",
    )
    parser.add_argument(
        "--sequentialCats", action=argparse.BooleanOptionalAction, default=True,
        help="If True (default), events are removed from the pool once assigned to an SR. "
             "Use --no-sequentialCats to allow overlapping SRs.",
    )
    args = parser.parse_args()

    # ── Load parquet(s) ────────────────────────────────────────────────
    print(f"Loading {len(args.input_parquet)} parquet file(s)...")
    df_all = pd.concat(
        [pd.read_parquet(p) for p in args.input_parquet],
        ignore_index=True,
    )
    print(f"  Total events : {len(df_all)}")
    print(f"  Columns      : {list(df_all.columns)}")

    # ── Load SR cuts ───────────────────────────────────────────────────
    if args.cats_json is None:
        print("\n[WARN] No SR categories JSON provided, skipping SR-based plots.")
        sr_cuts = {
            "ggHH score > 0.6": "ggHH_score > 0.6",
            "ggHH score > 0.9": "ggHH_score > 0.9",
            "ggHH score > 0.95": "ggHH_score > 0.95",
        }
        sequential = False
    else:
        with open(args.cats_json) as f:
            sr_cuts = json.load(f)
        sequential = args.sequentialCats
    print(f"\nSR categories loaded ({len(sr_cuts)}): {list(sr_cuts.keys())}")
    print(f"Sequential categories: {sequential}")

    # ── Loop over sample groups ────────────────────────────────────────
    for group_name, target_samples in SAMPLE_GROUPS.items():
        df = (
            df_all[df_all[COL_SAMPLE].isin(target_samples)]
            .reset_index(drop=True)
        )
        if len(df) == 0:
            print(f"\n[WARN] No events for group '{group_name}', skipping.")
            continue
        print(f"\nGroup '{group_name}': {len(df)} events")

        out_path = os.path.join(args.output_path, "cor_plots", group_name)
        os.makedirs(out_path, exist_ok=True)

        rel_w = df[COL_WEIGHT].values

        # ── Apply SR cuts ──────────────────────────────────────────────
        sr_masks = apply_cuts(df, sr_cuts, sequential=sequential)

        # ── Plot A: all SRs on one canvas, per plot variable ──────────
        for col, save_name, xlabel, range_ in PLOT_VARS:            
            if col not in df.columns:
                print(f"  [WARN] Column '{col}' not found, skipping.")
                continue
            data_all = df[col].values

            fig, ax = plt.subplots(figsize=(12, 8))
            plot_with_errorbars(data_all, rel_w, BINS, range_, "Inclusive", ax, color=COLORS.get("Inclusive", "gray"), inclusive=True)

            for sr_name, mask in sr_masks.items():
                # if sr_name not in ["VBFHH-lowMhh", "VBFHH-highMhh"]:
                #     print(f"  [WARN] Skipping '{sr_name}' for group '{group_name}' (not in this sample).")
                #     continue
                # if sr_name not in ["ggHH-lowMhh-1", "ggHH-lowMhh-2"]:
                #     print(f"  [WARN] Skipping '{sr_name}' for group '{group_name}' (not in this sample).")
                #     continue
                # if sr_name not in ["ggHH-highMhh-1", "ggHH-highMhh-2", "ggHH-highMhh-3"]:
                #     print(f"  [WARN] Skipping '{sr_name}' for group '{group_name}' (not in this sample).")
                #     continue
                # if sr_name not in ["ttH-lep-1", "ttH-lep-2", "ttH-lep-3"]:
                #     print(f"  [WARN] Skipping '{sr_name}' for group '{group_name}' (not in this sample).")
                #     continue
                if sr_name not in ["ttH-had-1", "ttH-had-2", "ttH-had-3"]:
                    print(f"  [WARN] Skipping '{sr_name}' for group '{group_name}' (not in this sample).")
                    continue
                plot_with_errorbars(data_all[mask], rel_w[mask], BINS, range_, sr_name, ax, color=COLORS.get(sr_name, "gray"))

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Normalized events")
            ax.legend()
            plt.title(f"{group_name}")
            plt.tight_layout()
            fig.savefig(f"{out_path}/nonResSamples_{save_name}_SR_sculpting_ttH-had.png")
            plt.close()
            print(f"  Saved: nonResSamples_{save_name}_SR_sculpting.png")

        # ── Plot B: one canvas per SR, per plot variable ───────────────
        for sr_name, mask in sr_masks.items():
            n_sr = mask.sum()
            for col, save_name, xlabel, range_ in PLOT_VARS:
                if col not in df.columns:
                    continue
                data_all = df[col].values

                fig, ax = plt.subplots(figsize=(10, 8))
                plot_with_errorbars(data_all,       rel_w,       BINS, range_, "Inclusive",                ax, color="blue", inclusive=True)
                plot_with_errorbars(data_all[mask], rel_w[mask], BINS, range_, f"{sr_name}", ax, color="red")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Normalized events")
                ax.legend()
                plt.title(f"{group_name} — {sr_name}")
                plt.tight_layout()
                fig.savefig(f"{out_path}/{sr_name}_{save_name}_sculpting.png")
                plt.close()

    print("\nDone. Plots saved to:", args.output_path)