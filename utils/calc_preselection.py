import argparse
import csv
import os
from collections import defaultdict

import awkward as ak
import numpy as np
import yaml

COLS = ["weight", "mass", "nonResReg_vbfpair_dijet_mass", "lead_mvaID", "sublead_mvaID"]

# Using mH = 125.4 GeV
# for kl samples and H BRs: https://gitlab.cern.ch/hh/recommendations/-/blob/master/CrossSections.md?ref_type=heads
# for singleH at 13.6 TeV: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap
# for singleH at 13 TeV: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#gluon_gluon_Fusion_Process

# factor for scaling mass for Run 2 VBF samples from 125.09 to 125.38, from above link for kl samples and H BRs
k_mass = 1.676/1.684
DICT_XSEC_13TEV = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.030649e3 * 0.00227 * 0.576 * 2, # cross sectio of GluGluToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct. 
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.068317e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.013422e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.090488e3 * 0.00227 * 0.576 * 2,

    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": 0.132486e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": 0.016068e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": 0.009427e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": 2.617158e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": 1.791638e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": 1.752648e3 * 0.00227 * 0.576 * 2, 

    "VBFHH_CV_1_C2V_1_C3_1": 0.0017260e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_1_C2V_0_C3_1": 0.0270800e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_1p74_C2V_1p37_C3_14p4": 0.3777832e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": 0.6322811e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": 0.0000120e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": 0.3340766e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": 0.0009976e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": 0.0033739e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": 0.0105109e3 * 0.00227 * 0.576 * 2 * k_mass,
    "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": 0.0149850e3 * 0.00227 * 0.576 * 2 * k_mass,

    # For singleH, XS(process) * BR(HtoGG)
    # Using mH = 125.38 for XS from: https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections
    # mH = 125.4 for BR
    "ttHtoGG_M_125": 0.525e3 * 0.00227,
    "BBHToGG_M_125": 0.522e3 * 0.00227,
    "GluGluHToGG_M_125": 47.84e3 * 0.00227,
    "VBFHToGG_M_125": 3.802e3 * 0.00227,
    "VHtoGG_M_125": 2.250e3 * 0.00227, # XS is sum of WH and ZH

    "DDQCDGJET": 1.0,
    "TTG_10_100": 4.216e3,
    "TTG_100_200": 0.4114e3,
    "TTG_200": 0.1284e3,
    "TT": 762.3e3,
    "GGJets": 88.75e3,
    "TTGG": 0.02391e3,
}

DICT_XSEC_13P6TEV = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.033969e3 * 0.00227 * 0.576 * 2,#0.033969e3 * 0.00227 * 0.576 * 2,  # cross sectio of GluGluToHH * BR(HToGG) * BR(HTobb) * 2 for two combination ### have to recheck if this is correct. 
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": 0.075495e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": 0.014864e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": 0.099298e3 * 0.00227 * 0.576 * 2,

    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": 2.900686e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": 0.010448e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": 0.146839e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": 0.017809e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": 1.985733e3 * 0.00227 * 0.576 * 2,
    "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": 1.752648e3 * 1.108268907 * 0.00227 * 0.576 * 2, 

    "VBFHH_CV_1_C2V_1_C3_1": 0.0019292e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_1_C2V_0_C3_1": 0.0296772e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_1p74_C2V_1p37_C3_14p4": 0.4002163e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": 0.6800842e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": 0.0000127e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": 0.3593242e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": 0.0011275e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": 0.0037987e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": 0.0117008e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": 0.0168528e3 * 0.00227 * 0.576 * 2,

    # For singleH, XS(process) * BR(HtoGG)
    # Using mH = 125.38 for XS from: https://gitlab.cern.ch/LHCHIGGSXS/LHCHXSWG1/crosssections
    # mH = 125.4 for BR
    "ttHtoGG_M_125": 0.589e3 * 0.00227,
    "BBHToGG_M_125": 0.563e3 * 0.00227,
    "GluGluHToGG_M_125": 51.45e3 * 0.00227,
    "VBFHToGG_M_125": 4.10e3 * 0.00227,
    "VHtoGG_M_125": 2.394e3 * 0.00227, # XS is sum of WH and ZH
    "WmHtoGG": 0.562e3 * 0.00227,
    "WpHtoGG": 0.891e3 * 0.00227,
    "ZHtoGG": 0.941e3 * 0.00227,

    "DDQCDGJET": 1.0,
    "TTG_10_100": 4.216e3,
    "TTG_100_200": 0.4114e3,
    "TTG_200": 0.1284e3,
    "TT": 762.3e3,
    "GGJets": 87.51e3,
    "GJetPt20To40": 242.5e3,
    "GJetPt40": 919.1e3,
    "TTGG": 0.02391e3,
}

LUMINOSITIES = {
    "2016preVFP":  19.5,
    "2016postVFP": 16.8,
    "2017":        42.07,
    "2018":        59.56,
    "preEE":       7.99, # We don't use era B
    "postEE":      26.68, # We don't use era B
    "preBPix":     17.96, # We don't use era B
    "postBPix":    9.68, # We don't use era B
    "2024":        109.82, # We don't use era B
    "2025":        110.58, # We don't use era B
}

RUN2_ERAS = {"2016preVFP", "2016postVFP", "2017", "2018"}
RUN3_ERAS = {"preEE", "postEE", "preBPix", "postBPix", "2024", "2025"}

SAMPLE_TO_ERA = {
    "2016preVFP_EraBv1": "2016preVFP",
    "2016preVFP_EraBv2": "2016preVFP",
    "2016preVFP_EraC": "2016preVFP",
    "2016preVFP_EraD": "2016preVFP",
    "2016preVFP_EraE": "2016preVFP",
    "2016preVFP_EraF": "2016preVFP",
    "2016postVFP_EraF": "2016postVFP",
    "2016postVFP_EraG": "2016postVFP",
    "2016postVFP_EraH": "2016postVFP",
    "2017_EraB": "2017",
    "2017_EraC": "2017",
    "2017_EraD": "2017",
    "2017_EraE": "2017",
    "2017_EraF": "2017",
    "2018_EraA": "2018",
    "2018_EraB": "2018",
    "2018_EraC": "2018",
    "2018_EraD": "2018",
    "2017": "2017",
    "2018": "2018",
    "2022_EraE": "postEE", 
    "2022_EraF": "postEE", 
    "2022_EraG": "postEE", 
    "2022_EraC": "preEE", 
    "2022_EraD": "preEE",
    "2023_EraC": "preBPix",
    "2023_EraD": "postBPix",
    "2023_EraCv1_EG0": "preBPix",
    "2023_EraCv1_EG1": "preBPix",
    "2023_EraCv2_EG0": "preBPix",
    "2023_EraCv2_EG1": "preBPix",
    "2023_EraCv3_EG0": "preBPix",
    "2023_EraCv3_EG1": "preBPix",
    "2023_EraCv4_EG0": "preBPix",
    "2023_EraCv4_EG1": "preBPix",
    "2023_EraDv1_EG0": "postBPix",
    "2023_EraDv1_EG1": "postBPix",
    "2023_EraDv2_EG0": "postBPix",
    "2023_EraDv2_EG1": "postBPix",
    "2024_EraC_EG0": "2024",
    "2024_EraC_EG1": "2024",
    "2024_EraD_EG0": "2024",
    "2024_EraD_EG1": "2024",
    "2024_EraE_EG0": "2024",
    "2024_EraE_EG1": "2024",
    "2024_EraF_EG0": "2024",
    "2024_EraF_EG1": "2024",
    "2024_EraG_EG0": "2024",
    "2024_EraG_EG1": "2024",
    "2024_EraH_EG0": "2024",
    "2024_EraH_EG1": "2024",
    "2024_EraIv1_EG0": "2024",
    "2024_EraIv1_EG1": "2024",
    "2024_EraIv2_EG0": "2024",
    "2024_EraIv2_EG1": "2024",
    "2025_EraCv1_EG0": "2025",
    "2025_EraCv1_EG1": "2025",
    "2025_EraCv1_EG2": "2025",
    "2025_EraCv1_EG3": "2025",
    "2025_EraCv2_EG0": "2025",
    "2025_EraCv2_EG1": "2025",
    "2025_EraCv2_EG2": "2025",
    "2025_EraCv2_EG3": "2025",
    "2025_EraDv1_EG0": "2025",
    "2025_EraDv1_EG1": "2025",
    "2025_EraDv1_EG2": "2025",
    "2025_EraDv1_EG3": "2025",
    "2025_EraEv1_EG0": "2025",
    "2025_EraEv1_EG1": "2025",
    "2025_EraEv1_EG2": "2025",
    "2025_EraEv1_EG3": "2025",
    "2025_EraFv1_EG0": "2025",
    "2025_EraFv1_EG1": "2025",
    "2025_EraFv1_EG2": "2025",
    "2025_EraFv1_EG3": "2025",
    "2025_EraFv2_EG0": "2025",
    "2025_EraFv2_EG1": "2025",
    "2025_EraFv2_EG2": "2025",
    "2025_EraFv2_EG3": "2025",
    "2025_EraGv1_EG0": "2025",
    "2025_EraGv1_EG1": "2025",
    "2025_EraGv1_EG2": "2025",
    "2025_EraGv1_EG3": "2025",
}


def get_xsec(sample, era):
    if era in RUN2_ERAS:
        return DICT_XSEC_13TEV[sample]
    elif era in RUN3_ERAS:
        return DICT_XSEC_13P6TEV[sample]
    raise ValueError(f"Unknown era: {era}")


def compute_weight_tot(events, sample, era):
    lumi = 1.0 if sample == "DDQCDGJET" else LUMINOSITIES[era]
    xsec = get_xsec(sample, era)
    return ak.to_numpy(events["weight"]) * xsec * lumi


def compute_weight_tot_data(events):
    return ak.to_numpy(events["weight"])


def filter_nonfinite_weight_events(events, weight_tot, label):
    finite_mask = np.isfinite(weight_tot)
    n_removed = int(np.count_nonzero(~finite_mask))
    if n_removed:
        print(
            f"  WARNING: removing {n_removed} event(s) with "
            f"non-finite weight_tot from {label}."
        )
    return events[finite_mask]


# Cut stages applied cumulatively (cut-flow order)
STAGES = [
    ("no_presel",       lambda ev: ev),
    ("mvaID_sel",       lambda ev: ev[
        (ev["lead_mvaID"] > -0.7) & (ev["sublead_mvaID"] > -0.7)
    ]),
    ("diphoton_mass_sel", lambda ev: ev[
        (ev["lead_mvaID"] > -0.7) & (ev["sublead_mvaID"] > -0.7)
        & (ev["mass"] > 100) & (ev["mass"] < 180)
    ]),
    ("dijet_mass_sel",  lambda ev: ev[
        (ev["lead_mvaID"] > -0.7) & (ev["sublead_mvaID"] > -0.7)
        & (ev["mass"] > 100) & (ev["mass"] < 180)
        & (ev["nonResReg_vbfpair_dijet_mass"] > 70)
        & (ev["nonResReg_vbfpair_dijet_mass"] < 190)
    ]),
]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["era", "sample", "n_events", "sum_weight_tot"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Cut-flow yields from parquets.")
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    parser.add_argument("--outdir", default=".", help="Directory for output CSVs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples_info = cfg["samples_info"]
    samples_path = samples_info["samples_path"]
    eras = samples_info["eras"]

    # rows_per_stage[stage_name] -> list of CSV rows
    rows = {name: [] for name, _ in STAGES}
    totals = {name: defaultdict(lambda: {"n_events": 0, "sum_weight_tot": 0.0})
              for name, _ in STAGES}

    for era in eras:
        era_str = str(era)
        if era_str not in samples_info:
            print(f"WARNING: era '{era_str}' not found in config, skipping.")
            continue

        print(f"\n=== {era_str} ===")
        for sample, parquet_rel in samples_info[era_str].items():
            parquet_full = f"{samples_path}/{parquet_rel}"

            try:
                events = ak.from_parquet(parquet_full, columns=COLS)
            except Exception as e:
                print(f"  ERROR loading {sample}: {e}")
                continue

            events = filter_nonfinite_weight_events(
                events,
                compute_weight_tot(events, sample, era_str),
                f"{era_str}/{sample}",
            )

            line_parts = [f"  {sample:55s}"]
            for stage_name, cut_fn in STAGES:
                ev = cut_fn(events)
                n = len(ev)
                s = float(np.sum(compute_weight_tot(ev, sample, era_str)))
                rows[stage_name].append([era_str, sample, n, s])
                totals[stage_name][sample]["n_events"] += n
                totals[stage_name][sample]["sum_weight_tot"] += s
                line_parts.append(f"{stage_name}: n={n:8d} yield={s:.4f}")

            print("  |  ".join(line_parts))

    if "data" in samples_info:
        print(f"\n=== data ===")
        for data_key, parquet_rel in samples_info["data"].items():
            era_str = SAMPLE_TO_ERA.get(data_key)
            if era_str is None:
                print(f"  WARNING: no era mapping for data sample '{data_key}', skipping.")
                continue

            parquet_full = f"{samples_path}/{parquet_rel}"
            try:
                events = ak.from_parquet(parquet_full, columns=COLS)
            except Exception as e:
                print(f"  ERROR loading {data_key}: {e}")
                continue

            events = filter_nonfinite_weight_events(
                events,
                compute_weight_tot_data(events),
                data_key,
            )

            sample_label = f"Data_{era_str}"
            line_parts = [f"  {data_key:30s} -> {era_str}"]
            for stage_name, cut_fn in STAGES:
                ev = cut_fn(events)
                n = len(ev)
                s = float(np.sum(compute_weight_tot_data(ev)))
                rows[stage_name].append([era_str, sample_label, n, s])
                totals[stage_name][sample_label]["n_events"] += n
                totals[stage_name][sample_label]["sum_weight_tot"] += s
                line_parts.append(f"{stage_name}: n={n:8d} yield={s:.4f}")
            print("  |  ".join(line_parts))

    # append per-sample ALL-era totals
    print("\n=== Totals across all eras ===")
    all_samples = sorted({s for name, _ in STAGES for s in totals[name]})
    for sample in all_samples:
        parts = [f"  {sample:55s}"]
        for stage_name, _ in STAGES:
            t = totals[stage_name][sample]
            rows[stage_name].append(["ALL", sample, t["n_events"], t["sum_weight_tot"]])
            parts.append(f"{stage_name}: yield={t['sum_weight_tot']:.4f}")
        print("  |  ".join(parts))

    os.makedirs(args.outdir, exist_ok=True)
    for stage_name, _ in STAGES:
        path = os.path.join(args.outdir, f"yield_{stage_name}.csv")
        write_csv(rows[stage_name], path)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
