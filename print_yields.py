#!/usr/bin/env python3
"""
Print weighted yields per MC process and data for a set of score cuts.

Reads from scored_samples/merged/merged_scored_events.parquet inside the
specified version directory (which contains sample, year, weight_tot and the
classifier score columns). Excludes the ggHH SM signal from MC yields.

Usage:
    python print_yields.py <version_dir> [options]

Score cut options (all optional, combined with AND):
    --ggHH-sig-min FLOAT     minimum is_ggHH_sig_score
    --ggHH-sig-max FLOAT     maximum is_ggHH_sig_score
    --Res-bkg-min  FLOAT     minimum is_Res_bkg_score
    --Res-bkg-max  FLOAT     maximum is_Res_bkg_score
    --nonRes-bkg-min FLOAT   minimum is_nonRes_bkg_score
    --nonRes-bkg-max FLOAT   maximum is_nonRes_bkg_score

Manual scaling (optional):
    --DDQCDGJets-sf      FLOAT   multiply DDQCDGJets yield by this factor (default: 1)
    --GGJets-sf          FLOAT   multiply GGJets yield by this factor (default: 1)
    --sf-nonRes-bkg-min  FLOAT   lower bound of is_nonRes_bkg_score range where SF is applied
    --sf-nonRes-bkg-max  FLOAT   upper bound of is_nonRes_bkg_score range where SF is applied

    If the SF range bounds are given, the SF is applied only to events whose
    is_nonRes_bkg_score falls within [min, max]; events outside keep weight 1.
    If no range is specified, the SF is applied to all events of that process.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


SIGNAL_SAMPLE = "GluGluToHH_kl-1p00_kt-1p00_c2-0p00"
SCORE_COLS = ["is_ggHH_sig_score", "is_Res_bkg_score", "is_nonRes_bkg_score"]
NEEDED_COLS = ["sample", "weight_tot"] + SCORE_COLS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print MC and data yields after score cuts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "version_dir",
        nargs="?",
        default=(
            "Version_20260222_bTagWPs_revisedNewBaseline_ResNonResggHHVBFHH"
            "_xsecWeightInSignal_1D_16-25_stopAt140Epochs_forPreApp"
        ),
        help="Version directory (default: the forPreApp version)",
    )
    parser.add_argument("--ggHH-sig-min", type=float, default=None, metavar="F",
                        help="Minimum is_ggHH_sig_score")
    parser.add_argument("--ggHH-sig-max", type=float, default=None, metavar="F",
                        help="Maximum is_ggHH_sig_score")
    parser.add_argument("--Res-bkg-min", type=float, default=None, metavar="F",
                        help="Minimum is_Res_bkg_score")
    parser.add_argument("--Res-bkg-max", type=float, default=None, metavar="F",
                        help="Maximum is_Res_bkg_score")
    parser.add_argument("--nonRes-bkg-min", type=float, default=None, metavar="F",
                        help="Minimum is_nonRes_bkg_score")
    parser.add_argument("--nonRes-bkg-max", type=float, default=None, metavar="F",
                        help="Maximum is_nonRes_bkg_score")
    parser.add_argument("--DDQCDGJets-sf", type=float, default=1.0, metavar="F",
                        help="Scale factor applied to DDQCDGJets yield (default: 1)")
    parser.add_argument("--GGJets-sf", type=float, default=1.0, metavar="F",
                        help="Scale factor applied to GGJets yield (default: 1)")
    parser.add_argument("--sf-nonRes-bkg-min", type=float, default=None, metavar="F",
                        help="Lower bound of is_nonRes_bkg_score range where SF is applied")
    parser.add_argument("--sf-nonRes-bkg-max", type=float, default=None, metavar="F",
                        help="Upper bound of is_nonRes_bkg_score range where SF is applied")
    return parser.parse_args()


def build_mask(df, args):
    mask = pd.Series(True, index=df.index)
    cuts = []
    if args.ggHH_sig_min is not None:
        mask &= df["is_ggHH_sig_score"] >= args.ggHH_sig_min
        cuts.append(f"is_ggHH_sig_score >= {args.ggHH_sig_min}")
    if args.ggHH_sig_max is not None:
        mask &= df["is_ggHH_sig_score"] <= args.ggHH_sig_max
        cuts.append(f"is_ggHH_sig_score <= {args.ggHH_sig_max}")
    if args.Res_bkg_min is not None:
        mask &= df["is_Res_bkg_score"] >= args.Res_bkg_min
        cuts.append(f"is_Res_bkg_score >= {args.Res_bkg_min}")
    if args.Res_bkg_max is not None:
        mask &= df["is_Res_bkg_score"] <= args.Res_bkg_max
        cuts.append(f"is_Res_bkg_score <= {args.Res_bkg_max}")
    if args.nonRes_bkg_min is not None:
        mask &= df["is_nonRes_bkg_score"] >= args.nonRes_bkg_min
        cuts.append(f"is_nonRes_bkg_score >= {args.nonRes_bkg_min}")
    if args.nonRes_bkg_max is not None:
        mask &= df["is_nonRes_bkg_score"] <= args.nonRes_bkg_max
        cuts.append(f"is_nonRes_bkg_score <= {args.nonRes_bkg_max}")
    return mask, cuts


def print_yields(yields_mc, total_mc, data_yield):
    col_w = max(len(s) for s in yields_mc) + 2
    val_w = 14

    header = f"{'Process':<{col_w}}  {'Yield':>{val_w}}"
    print(header)
    print("-" * len(header))

    for sample, y in sorted(yields_mc.items()):
        print(f"{sample:<{col_w}}  {y:>{val_w}.4f}")

    print("-" * len(header))
    print(f"{'Total MC':<{col_w}}  {total_mc:>{val_w}.4f}")
    print(f"{'Data':<{col_w}}  {data_yield:>{val_w}.0f}")
    print(f"{'Data / Total MC':<{col_w}}  {data_yield / total_mc:>{val_w}.4f}" if total_mc else "")


def main():
    args = parse_args()

    # argparse replaces hyphens with underscores
    version_dir = Path(args.version_dir)
    parquet_path = version_dir / "scored_samples" / "merged" / "merged_scored_events.parquet"

    if not parquet_path.exists():
        sys.exit(f"Error: parquet file not found at {parquet_path}")

    print(f"Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path, columns=NEEDED_COLS)
    print(f"  Total events: {len(df):,}")

    mask, cuts = build_mask(df, args)

    if cuts:
        print("\nApplied cuts:")
        for c in cuts:
            print(f"  {c}")
    else:
        print("\nNo score cuts applied.")

    df = df[mask]
    print(f"  Events after cuts: {len(df):,}\n")

    data_df = df[df["sample"] == "Data"]
    mc_df = df[(df["sample"] != "Data") & (df["sample"] != SIGNAL_SAMPLE)]

    yields_mc = mc_df.groupby("sample")["weight_tot"].sum().to_dict()

    sf_range_min = args.sf_nonRes_bkg_min
    sf_range_max = args.sf_nonRes_bkg_max
    for sample, sf in [("DDQCDGJets", args.DDQCDGJets_sf), ("GGJets", args.GGJets_sf)]:
        if sf == 1.0 or sample not in yields_mc:
            continue
        proc = mc_df[mc_df["sample"] == sample]
        if sf_range_min is not None or sf_range_max is not None:
            in_range = pd.Series(True, index=proc.index)
            if sf_range_min is not None:
                in_range &= proc["is_nonRes_bkg_score"] >= sf_range_min
            if sf_range_max is not None:
                in_range &= proc["is_nonRes_bkg_score"] <= sf_range_max
            scaled = proc.loc[in_range, "weight_tot"].sum() * sf
            unscaled = proc.loc[~in_range, "weight_tot"].sum()
            yields_mc[sample] = scaled + unscaled
            range_str = f" in is_nonRes_bkg_score [{sf_range_min}, {sf_range_max}]"
        else:
            yields_mc[sample] *= sf
            range_str = " (all events)"
        print(f"Applying SF={sf} to {sample}{range_str}")

    total_mc = sum(yields_mc.values())
    data_yield = len(data_df)  # data: unweighted event count

    print_yields(yields_mc, total_mc, data_yield)


if __name__ == "__main__":
    main()
