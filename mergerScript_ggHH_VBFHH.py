import numpy as np
import os
import gc
import sys
import argparse
import pandas as pd
import awkward as ak
import pyarrow.parquet as pq

ff_sampledict = {
    "GGJets": "GGJets", 
    "DDQCDGJET": "DDQCDGJets",
    "TTGG": "TTGG",
    "TT": "TT",
    "TTG_10_100": "TTG_10_100",
    "TTG_100_200": "TTG_100_200",
    "TTG_200": "TTG_200",
    "ttHtoGG_M_125": "ttHToGG",
    "BBHToGG_M_125": "BBHToGG",
    "GluGluHToGG_M_125": "GluGluHToGG",
    "VBFHToGG_M_125": "VBFHToGG",
    "VHtoGG_M_125": "VHToGG",
    "WmHtoGG": "VHToGG",
    "WpHtoGG": "VHToGG",
    "ZHtoGG": "VHToGG",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGluToHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGluToHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGluToHH_kl-5p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": "GluGluToHH_kl-1p00_kt-1p00_c2-0p10",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": "GluGluToHH_kl-1p00_kt-1p00_c2-0p35",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": "GluGluToHH_kl-1p00_kt-1p00_c2-3p00",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": "GluGluToHH_kl-1p00_kt-1p00_c2-m2p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": "GluGluToHH_kl-0p00_kt-1p00_c2-1p00",
    "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": "GluGluToHH_kl-m20p00_kt-1p00_c2-2p24",
    "VBFHH_CV_1_C2V_1_C3_1": "VBFToHH_CV-1p000_C2V-1p000_C3-1p000",
    "VBFHH_CV_1_C2V_0_C3_1": "VBFToHH_CV-1p000_C2V-0p000_C3-1p000",
    "VBFHH_CV_1p74_C2V_1p37_C3_14p4": "VBFToHH_CV-1p74_C2V-1p37_C3-14p4",
    "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": "VBFToHH_CV-2p12_C2V-3p87_C3-m5p96",
    "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": "VBFToHH_CV-m0p012_C2V-0p030_C3-10p2",
    "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": "VBFToHH_CV-m0p758_C2V-1p44_C3-m19p3",
    "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": "VBFToHH_CV-m0p962_C2V-0p959_C3-m1p43",
    "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": "VBFToHH_CV-m1p21_C2V-1p94_C3-m0p94",
    "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": "VBFToHH_CV-m1p60_C2V-2p72_C3-m1p36",
    "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": "VBFToHH_CV-m1p83_C2V-3p57_C3-m3p39",
}

def load_samples(base_path, eras, samples, data=False, syst="", boosted_folder=None):
    """Load predictions and weights, scaling weights by luminosity.

    If ``boosted_folder`` is given, the ggHH ``events.parquet`` is read from
    ``boosted_folder`` (file named ``events_boostedCat.parquet``) instead of the
    ggHH folder under ``base_path``. The boosted folder is expected to mirror the
    same ``individual_samples/<era>/<sample>/<syst>`` subfolder structure. The
    ggHH ``y.npy`` is always read from ``base_path``.
    """
    # def load_events(events_dir, cols):
    #     """Load the events for a directory under base_path.

    #     If ``boosted_folder`` is set, read ``events_boostedCat.parquet`` from the
    #     mirrored directory under ``boosted_folder`` (same individual_samples/...
    #     subpath) instead of ``events.parquet`` under base_path. The boosted parquet
    #     is cross-checked against the original ``events.parquet`` (the one y.npy
    #     corresponds to): same length AND identical run/event/lumi ordering, else
    #     the scores would be silently misaligned -> abort.
    #     """
    #     orig_file = os.path.join(events_dir, 'events.parquet')
    #     if boosted_folder is None:
    #         print(f"Loading events from {orig_file}")
    #         return ak.from_parquet(orig_file, columns=cols)

    #     boosted_file = os.path.join(events_dir.replace(base_path, boosted_folder, 1), 'events_boostedCat.parquet')
    #     print(f"Loading events from {boosted_file}")
    #     events_ = ak.from_parquet(boosted_file, columns=cols)
    #     orig_ = ak.from_parquet(orig_file, columns=["run", "event", "lumi"])
    #     if len(events_) != len(orig_):
    #         raise ValueError(
    #             f"Boosted events length mismatch at {boosted_file}: "
    #             f"boosted={len(events_)}, original={len(orig_)}. Aborting.")
    #     for key in ["run", "event", "lumi"]:
    #         if not np.array_equal(np.asarray(events_[key]), np.asarray(orig_[key])):
    #             raise ValueError(
    #                 f"Boosted events '{key}' does not match original events.parquet at "
    #                 f"{boosted_file}. Not aligned with y.npy; aborting.")
    #     return events_
    def load_events(events_dir, cols):
        """Load regular columns from A and boosted/AK8 columns from B."""

        def unique(items):
            return list(dict.fromkeys(items))

        def missing_default(col):
            # Missing weight/SF columns use the original neutral convention.
            if col == "weight" or col.startswith("weight_"):
                return 1.0
            return -999.0

        def add_missing_columns(events, missing_cols, source_file):
            expected_missing = []
            other_era_btag_columns = set()
            if not data:
                for other_era in eras:
                    if other_era == era:
                        continue
                    other_btag_era = btag_sf_era_map[other_era]
                    other_era_btag_columns.update([
                        f"weight_btagSFbc_{other_btag_era}Down",
                        f"weight_btagSFbc_{other_btag_era}Up",
                        f"weight_btagSFlight_{other_btag_era}Down",
                        f"weight_btagSFlight_{other_btag_era}Up",
                    ])
            for col in missing_cols:
                default = missing_default(col)
                if col in other_era_btag_columns:
                    expected_missing.append(col)
                # else:
                    # print(f"[WARNING] Missing '{col}' in {source_file}; filling with {default}")
                events = ak.with_field(
                    events,
                    np.full(len(events), default, dtype=np.float64),
                    where=col,
                )
            if expected_missing:
                print(f"[INFO] Filled {len(expected_missing)} non-applicable other-era b-tag SF columns with 1.0 for era '{era}'.")
            return events

        key_cols = ["run", "lumi", "event"]

        columns_from_b = [
            "y_proba",
            "weight_noAK8SF",
            "weight_tot_noAK8SF",
            "weight_btagSFAK8Up",
            "weight_btagSFAK8Down",
        ]

        requested = unique(cols)
        file_a = os.path.join(events_dir, "events.parquet")

        if not os.path.exists(file_a):
            raise FileNotFoundError(f"Missing parquet A: {file_a}")

        fields_a = set(pq.ParquetFile(file_a).schema_arrow.names)

        for key in key_cols:
            if key not in fields_a:
                raise ValueError(f"Key '{key}' missing from A: {file_a}")

        # No B folder: read only A and fill missing columns.
        if boosted_folder is None:
            present_from_a = [
                col for col in requested
                if col in fields_a
            ]
            missing_from_a = [
                col for col in requested
                if col not in fields_a
            ]

            # print(f"Loading A: {file_a}")
            events_a = ak.from_parquet(
                file_a,
                columns=unique(key_cols + present_from_a),
            )

            events_a = add_missing_columns(
                events_a,
                missing_from_a,
                file_a,
            )

            return events_a[requested]

        # B exists: normal columns from A, boosted/AK8 columns from B.
        relative_dir = os.path.relpath(events_dir, base_path)
        file_b = os.path.join(
            boosted_folder,
            relative_dir,
            "events_boostedCat.parquet",
        )

        if not os.path.exists(file_b):
            raise FileNotFoundError(f"Missing parquet B: {file_b}")

        fields_b = set(pq.ParquetFile(file_b).schema_arrow.names)

        for key in key_cols:
            if key not in fields_b:
                raise ValueError(f"Key '{key}' missing from B: {file_b}")

        requested_from_b = [
            col for col in columns_from_b
            if col in requested
        ]

        requested_from_a = [
            col for col in requested
            if col not in columns_from_b
        ]

        present_from_a = [
            col for col in requested_from_a
            if col in fields_a
        ]
        missing_from_a = [
            col for col in requested_from_a
            if col not in fields_a
        ]

        present_from_b = [
            col for col in requested_from_b
            if col in fields_b
        ]
        missing_from_b = [
            col for col in requested_from_b
            if col not in fields_b
        ]

        # print(f"Loading A: {file_a}")
        events_a = ak.from_parquet(
            file_a,
            columns=unique(key_cols + present_from_a),
        )

        # print(f"Loading B: {file_b}")
        events_b = ak.from_parquet(
            file_b,
            columns=unique(key_cols + present_from_b),
        )

        if len(events_a) != len(events_b):
            raise ValueError(
                f"A/B event-count mismatch:\n"
                f"A: {len(events_a)} events — {file_a}\n"
                f"B: {len(events_b)} events — {file_b}"
            )

        # Strict event-order validation.
        for key in key_cols:
            values_a = np.asarray(events_a[key])
            values_b = np.asarray(events_b[key])

            mismatch = np.flatnonzero(values_a != values_b)

            if mismatch.size:
                index = int(mismatch[0])
                key_a = tuple(
                    np.asarray(events_a[k])[index]
                    for k in key_cols
                )
                key_b = tuple(
                    np.asarray(events_b[k])[index]
                    for k in key_cols
                )

                raise ValueError(
                    f"A/B event ordering mismatch at row {index}:\n"
                    f"A (run, lumi, event) = {key_a}\n"
                    f"B (run, lumi, event) = {key_b}\n"
                    f"A file: {file_a}\n"
                    f"B file: {file_b}"
                )

        # Fill columns missing from A.
        events_a = add_missing_columns(
            events_a,
            missing_from_a,
            file_a,
        )

        # Attach available B columns.
        for col in present_from_b:
            events_a = ak.with_field(
                events_a,
                events_b[col],
                where=col,
            )

        # Fill requested columns missing from B.
        events_a = add_missing_columns(
            events_a,
            missing_from_b,
            file_b,
        )

        print(
            f"[OK] A/B alignment verified: {len(events_a)} events; "
            f"attached {len(present_from_b)} columns from B."
        )

        return events_a[requested]

    # Example MC file to get the weight columns
    parquet_file = pq.ParquetFile(base_path+"/individual_samples/2024/ttHtoGG_M_125/"+syst+"/events.parquet")
    all_columns = parquet_file.schema.names
    # weight_columns = [col for col in all_columns if 'weight' in col]
    weight_columns = ["weight_tot", "weight", ] #\
            #  'weight_EFT_kl_m2p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m2p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m9p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m9p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_15p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_15p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_5p24_kt_m4p00_c2_m1p24_cg_m3p35_c2g_m0p62', 'weight_EFT_unc_kl_5p24_kt_m4p00_c2_m1p24_cg_m3p35_c2g_m0p62', 'weight_EFT_kl_13p51_kt_m2p19_c2_0p08_cg_3p27_c2g_0p43', 'weight_EFT_unc_kl_13p51_kt_m2p19_c2_0p08_cg_3p27_c2g_0p43', 'weight_EFT_kl_29p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_29p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m28p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m28p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_m1p20', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_m1p20', 'weight_EFT_kl_m25p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m25p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_20p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_20p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p32_kt_4p63_c2_3p23_cg_m2p89_c2g_2p81', 'weight_EFT_unc_kl_1p32_kt_4p63_c2_3p23_cg_m2p89_c2g_2p81', 'weight_EFT_kl_m3p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m3p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m7p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m7p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_4p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_4p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_15p00_kt_1p00_c2_0p00_cg_m1p00_c2g_1p00', 'weight_EFT_unc_kl_15p00_kt_1p00_c2_0p00_cg_m1p00_c2g_1p00', 'weight_EFT_kl_m2p50_kt_1p00_c2_0p00_cg_0p00_c2g_m1p20', 'weight_EFT_unc_kl_m2p50_kt_1p00_c2_0p00_cg_0p00_c2g_m1p20', 'weight_EFT_kl_11p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_11p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_m0p20_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_m0p20_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_1p10', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_1p10', 'weight_EFT_kl_m12p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m12p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m14p00_kt_1p14_c2_5p00_cg_m0p28_c2g_m2p19', 'weight_EFT_unc_kl_m14p00_kt_1p14_c2_5p00_cg_m0p28_c2g_m2p19', 'weight_EFT_kl_3p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_3p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_17p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_17p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_7p05_kt_m2p05_c2_3p61_cg_1p89_c2g_2p20', 'weight_EFT_unc_kl_7p05_kt_m2p05_c2_3p61_cg_1p89_c2g_2p20', 'weight_EFT_kl_19p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_19p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m14p00_kt_m0p74_c2_3p05_cg_2p64_c2g_m1p57', 'weight_EFT_unc_kl_m14p00_kt_m0p74_c2_3p05_cg_2p64_c2g_m1p57', 'weight_EFT_kl_m8p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m8p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_20p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_20p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p70_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p70_cg_0p00_c2g_0p00', 'weight_EFT_kl_m22p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m22p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m18p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m18p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m16p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m16p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_14p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_14p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_25p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_25p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_3p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_3p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_m2p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_m2p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_28p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_28p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_0p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_0p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m11p71_kt_4p15_c2_4p22_cg_1p84_c2g_3p33', 'weight_EFT_unc_kl_m11p71_kt_4p15_c2_4p22_cg_1p84_c2g_3p33', 'weight_EFT_kl_2p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_2p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m14p00_kt_0p75_c2_5p00_cg_1p62_c2g_1p56', 'weight_EFT_unc_kl_m14p00_kt_0p75_c2_5p00_cg_1p62_c2g_1p56', 'weight_EFT_kl_5p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_5p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m29p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m29p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m18p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m18p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m25p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m25p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m13p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m13p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_6p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_6p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m10p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m10p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m5p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m5p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_14p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_14p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m11p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m11p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_4p13_kt_1p40_c2_m3p56_cg_m2p32_c2g_m2p07', 'weight_EFT_unc_kl_4p13_kt_1p40_c2_m3p56_cg_m2p32_c2g_m2p07', 'weight_EFT_kl_m0p07_kt_m2p40_c2_1p20_cg_9p53_c2g_m1p55', 'weight_EFT_unc_kl_m0p07_kt_m2p40_c2_1p20_cg_9p53_c2g_m1p55', 'weight_EFT_kl_1p00_kt_1p00_c2_1p00_cg_6p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_1p00_cg_6p00_c2g_0p00', 'weight_EFT_kl_9p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_9p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_8p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_8p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_22p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_22p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_13p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_13p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m26p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m26p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_26p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_26p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m2p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m2p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m14p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m14p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m23p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m23p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m15p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m15p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m5p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m5p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m24p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m24p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_7p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_7p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m9p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m9p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_21p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_21p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_25p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_25p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_2p45_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_2p45_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m23p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m23p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m16p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m16p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m10p00_kt_1p00_c2_1p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m10p00_kt_1p00_c2_1p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m22p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m22p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_10p00_cg_1p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_10p00_cg_1p00_c2g_0p00', 'weight_EFT_kl_m20p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m20p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m11p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m11p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m20p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m20p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m14p00_kt_m1p26_c2_m1p92_cg_m5p00_c2g_0p95', 'weight_EFT_unc_kl_m14p00_kt_m1p26_c2_m1p92_cg_m5p00_c2g_0p95', 'weight_EFT_kl_m6p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m6p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m19p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m19p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_12p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_12p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_27p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_27p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_24p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_24p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_6p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_6p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_28p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_28p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_5p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_5p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m29p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m29p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m13p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m13p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_11p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_11p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m1p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m1p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_16p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_16p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_0p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_0p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_1p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_0p00_c2g_1p00', 'weight_EFT_kl_22p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_22p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_7p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_7p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m4p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m4p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m10p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m10p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_2p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_2p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_0p00', 'weight_EFT_kl_m27p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m27p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_30p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_30p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p35_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p35_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_m3p00_c2g_3p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_m3p00_c2g_3p00', 'weight_EFT_kl_1p00_kt_5p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_5p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_21p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_21p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_m2p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_1p00_c2g_m2p00', 'weight_EFT_kl_23p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_23p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m3p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m3p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_0p00_kt_1p00_c2_1p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_0p00_kt_1p00_c2_1p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p50_cg_m0p80_c2g_0p60', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p50_cg_m0p80_c2g_0p60', 'weight_EFT_kl_19p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_19p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m24p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m24p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_1p00_cg_1p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_1p00_cg_1p00_c2g_0p00', 'weight_EFT_kl_7p01_kt_6p00_c2_3p97_cg_m3p74_c2g_m1p43', 'weight_EFT_unc_kl_7p01_kt_6p00_c2_3p97_cg_m3p74_c2g_m1p43', 'weight_EFT_kl_4p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_4p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_10p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_10p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_3p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_3p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_24p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_24p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m12p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m12p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_13p00_kt_4p59_c2_1p27_cg_2p83_c2g_2p14', 'weight_EFT_unc_kl_13p00_kt_4p59_c2_1p27_cg_2p83_c2g_2p14', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_m0p50_c2g_m0p70', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_m0p50_c2g_m0p70', 'weight_EFT_kl_10p73_kt_5p96_c2_4p36_cg_m2p05_c2g_m1p81', 'weight_EFT_unc_kl_10p73_kt_5p96_c2_4p36_cg_m2p05_c2g_m1p81', 'weight_EFT_kl_9p79_kt_6p00_c2_5p00_cg_m8p00_c2g_1p91', 'weight_EFT_unc_kl_9p79_kt_6p00_c2_5p00_cg_m8p00_c2g_1p91', 'weight_EFT_kl_1p00_kt_1p00_c2_0p10_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p10_cg_0p00_c2g_0p00', 'weight_EFT_kl_m8p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m8p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_13p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_13p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_14p10_kt_m3p98_c2_11p06_cg_3p59_c2g_m1p83', 'weight_EFT_unc_kl_14p10_kt_m3p98_c2_11p06_cg_3p59_c2g_m1p83', 'weight_EFT_kl_18p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_18p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_8p28_kt_6p00_c2_m0p28_cg_1p21_c2g_m3p68', 'weight_EFT_unc_kl_8p28_kt_6p00_c2_m0p28_cg_1p21_c2g_m3p68', 'weight_EFT_kl_m14p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m14p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_26p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_26p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_0p00_cg_1p50_c2g_m0p50', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_0p00_cg_1p50_c2g_m0p50', 'weight_EFT_kl_m3p10_kt_1p45_c2_3p46_cg_m3p38_c2g_m2p57', 'weight_EFT_unc_kl_m3p10_kt_1p45_c2_3p46_cg_m3p38_c2g_m2p57', 'weight_EFT_kl_8p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_8p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m0p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m0p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m21p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m21p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_9p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_9p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_18p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_18p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_17p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_17p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_10p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_10p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m15p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m15p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_12p97_kt_m0p97_c2_m4p08_cg_4p69_c2g_2p70', 'weight_EFT_unc_kl_12p97_kt_m0p97_c2_m4p08_cg_4p69_c2g_2p70', 'weight_EFT_kl_23p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_23p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m19p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m19p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m7p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m7p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_1p00_cg_m0p60_c2g_0p60', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_1p00_cg_m0p60_c2g_0p60', 'weight_EFT_kl_m1p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m1p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m30p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m30p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_2p40_kt_1p00_c2_0p00_cg_0p20_c2g_m0p20', 'weight_EFT_unc_kl_2p40_kt_1p00_c2_0p00_cg_0p20_c2g_m0p20', 'weight_EFT_kl_m6p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m6p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_0p01_kt_m1p59_c2_5p00_cg_m5p00_c2g_m0p55', 'weight_EFT_unc_kl_0p01_kt_m1p59_c2_5p00_cg_m5p00_c2g_m0p55', 'weight_EFT_kl_m17p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m17p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m28p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m28p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_12p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_12p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_27p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_27p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_29p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_29p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_15p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_15p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_1p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_16p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_16p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_1p00_kt_1p00_c2_m0p20_cg_0p00_c2g_m1p20', 'weight_EFT_unc_kl_1p00_kt_1p00_c2_m0p20_cg_0p00_c2g_m1p20', 'weight_EFT_kl_34p45_kt_m0p39_c2_1p79_cg_m1p78_c2g_4p65', 'weight_EFT_unc_kl_34p45_kt_m0p39_c2_1p79_cg_m1p78_c2g_4p65', 'weight_EFT_kl_m17p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m17p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_0p00_kt_0p00_c2_0p00_cg_0p00_c2g_1p00', 'weight_EFT_unc_kl_0p00_kt_0p00_c2_0p00_cg_0p00_c2g_1p00', 'weight_EFT_kl_m21p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m21p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_15p41_kt_2p83_c2_m0p34_cg_m5p00_c2g_0p38', 'weight_EFT_unc_kl_15p41_kt_2p83_c2_m0p34_cg_m5p00_c2g_0p38', 'weight_EFT_kl_m14p00_kt_m2p86_c2_2p55_cg_4p20_c2g_1p87', 'weight_EFT_unc_kl_m14p00_kt_m2p86_c2_2p55_cg_4p20_c2g_1p87', 'weight_EFT_kl_m27p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m27p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_m4p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m4p00_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_kl_20p00_kt_1p00_c2_0p00_cg_0p00_c2g_5p00', 'weight_EFT_unc_kl_20p00_kt_1p00_c2_0p00_cg_0p00_c2g_5p00', 'weight_EFT_kl_m26p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_EFT_unc_kl_m26p50_kt_1p00_c2_0p00_cg_0p00_c2g_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_m20p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_m20p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_m20p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_m20p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_unc_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_unc_CH_0p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_CH_m20p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_m20p00_CHBox_20p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_10p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_10p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_CH_2p13_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_2p13_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_20p00_CHD_10p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_20p00_CHD_10p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p10', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_40p00_CHG_0p10', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_unc_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_m5p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_m5p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p10', 'weight_SMEFT_CH_0p00_CHBox_20p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_20p00_CHD_0p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_10p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_10p00_CuH_40p00_CHG_0p00', 'weight_SMEFT_CH_m8p50_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_m8p50_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_m0p05', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_m0p05', 'weight_SMEFT_CH_0p00_CHBox_m10p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_m10p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_m20p00_CHBox_0p00_CHD_0p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_CH_0p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p00', 'weight_SMEFT_unc_CH_0p00_CHBox_0p00_CHD_10p00_CuH_0p00_CHG_0p00']
    dijet_mass_key = "nonResReg_vbfpair_dijet_mass"
    HH_mass_key = "nonResReg_vbfpair_M_X"

    #"nonResReg_lead_bjet_hFlav", "nonResReg_sublead_bjet_hFlav", "event", "run", "lumi"]#, "is_boosted", "y_proba"] 
    columns = [
        "mass",
        dijet_mass_key,
        HH_mass_key,
        "nonResReg_vbfpair_M_X",
        "run",
        "lumi",
        "event",
        "n_jets",
        "n_electrons",
        "n_muons",
        "jet1_pt",
        "jet2_pt",
        "jet3_pt",
        "jet4_pt",
        "jet5_pt",
        "jet6_pt",
        "jet7_pt",
        "jet8_pt",
        "jet9_pt",
        "jet10_pt",
        "nBTight",
        "lead_mvaID",
        "sublead_mvaID",
        "nonResReg_vbfpair_VBF_dijet_mass",
        ]

    col_syst_common = [
        "weight_ElectronVetoSFDown",
        "weight_ElectronVetoSFUp",
        "weight_PileupDown",
        "weight_PileupUp",
        "weight_TriggerSFDown",
        "weight_TriggerSFUp",
        "weight_btagSFbc_correlatedDown",
        "weight_btagSFbc_correlatedUp",
        "weight_btagSFlight_correlatedDown",
        "weight_btagSFlight_correlatedUp",
    ]
    col_syst_run3 = [
        "weight_LoosePhoIDSFDown",
        "weight_LoosePhoIDSFUp",
        "weight_PreselSFDown",
        "weight_PreselSFUp",
    ]
    btag_sf_era_map = {
        "2016preVFP": "2016preVFP",
        "2016postVFP": "2016postVFP",
        "2017": "2017",
        "2018": "2018",
        "preEE": "2022preEE",
        "postEE": "2022postEE",
        "preBPix": "2023preBPix",
        "postBPix": "2023postBPix",
        "2024": "2024",
        "2025": "2025",
    }

    if syst == "":
        # Include every SF column in every load_samples output.
        columns += col_syst_common
        columns += col_syst_run3

        for btag_era in dict.fromkeys(btag_sf_era_map.values()):
            columns += [
                f"weight_btagSFbc_{btag_era}Down",
                f"weight_btagSFbc_{btag_era}Up",
                f"weight_btagSFlight_{btag_era}Down",
                f"weight_btagSFlight_{btag_era}Up",
            ]

    if boosted_folder is not None:
        columns += [
            "y_proba",
            "weight_noAK8SF",
            "weight_tot_noAK8SF",
            "weight_btagSFAK8Up",
            "weight_btagSFAK8Down",
        ]

    columns = list(dict.fromkeys(columns))

    columns_gen = ["lead_genPartFlav", "sublead_genPartFlav"]
    # columns_gen_ggHH = []
    columns_gen_ggHH = ["gen_mHH_hardProc", "gen_pT_HH_hardProc", "gen_CosThetaStar_HH_hardProc"]

    samples_input = {
            "mass": [], 
            "dijet_mass": [], 
            "HHbbggCandidate_mass": [],
            "nonResReg_vbfpair_VBF_dijet_mass": [],
            # "nonResReg_vbfpair_dijet_mass": [],
            "lead_mvaID" : [],
            "sublead_mvaID" : [],
            "sample": [],
            "year": [],
            "era": [],
            "score": [],
            "nonRes_score": [],
            # "nonRes_singleH_score": [],
            "ttH_score": [],
            "singleH_score" :[],
            "ggHH_score":[],
            "VBFMVA_score": [],
            "VBFMVA_VBFHH": [],
            "VBFMVA_ggHH": [],
            "VBFMVA_ResBkg": [],
            "VBFMVA_nonResBkg": [],
    }

    for col in columns + columns_gen + columns_gen_ggHH:
        if col not in [dijet_mass_key, HH_mass_key]:
            samples_input[col] = []

    for weight in weight_columns:
        samples_input.update({weight: []})

    for era in eras:
        print("###########")
        print(era)
        print("###########")
        print()
        for sample in samples:
            print(sample)
            if (sample in ["GGJets", "DDQCDGJET", "TTGG", "TT", "TTG_10_100", "TTG_100_200", "TTG_200"]) and (syst != ""):
                continue
            
            if ("GluGlutoHHto2B2G_EFTReweighted" in sample):
                continue

            if (era == "2024" or era == "2025") & (sample == "VHtoGG_M_125"):
                VHsample = "WmHtoGG"
                path_VH = os.path.join(base_path, "individual_samples"+"/", era, VHsample, syst)
                y_path_VH = os.path.join(path_VH, 'y.npy')
                w_path_VH = os.path.join(path_VH, 'rel_w.npy')
                events = load_events(path_VH, columns+weight_columns+columns_gen)
                y = np.load(y_path_VH)
                
                path_VH_VBFMVA = os.path.join(base_path_VBFMVA, "individual_samples"+"/", era, VHsample, syst)
                y_path_VH_VBFMVA = os.path.join(path_VH_VBFMVA, 'y.npy')
                y_VBFMVA = np.load(y_path_VH_VBFMVA)

                for VHsample in ["WpHtoGG", "ZHtoGG"]:
                    path_VH = os.path.join(base_path, "individual_samples"+"/", era, VHsample, syst)
                    y_path_VH = os.path.join(path_VH, 'y.npy')
                    w_path_VH = os.path.join(path_VH, 'rel_w.npy')
                    events_VH = load_events(path_VH, columns+weight_columns+columns_gen)
                    y_VH = np.load(y_path_VH)
                    
                    path_VH_VBFMVA = os.path.join(base_path_VBFMVA, "individual_samples"+"/", era, VHsample, syst)
                    y_path_VH_VBFMVA = os.path.join(path_VH_VBFMVA, 'y.npy')
                    y_VH_VBFMVA = np.load(y_path_VH_VBFMVA)

                    events = ak.concatenate([events, events_VH])
                    y = np.concatenate([y, y_VH])
                    y_VBFMVA = np.concatenate([y_VBFMVA, y_VH_VBFMVA])
                
            else:
                if era != "postEE":
                    if ("TTG_" in sample) or (sample == "TT"):
                        continue
                if data:
                    path = os.path.join(base_path, "individual_samples_data", era, sample)
                    y_path = os.path.join(path, 'y.npy')
                    w_path = os.path.join(path, 'rel_w.npy')
                    events = load_events(path, columns+weight_columns)
                else:
                    if not os.path.exists(os.path.join(os.path.join(base_path, "individual_samples"+"/", era, sample, syst), 'events.parquet')):
                        print(f"{era} {sample} does not exist, skip.")
                        continue
                    if sample == "DDQCDGJET":
                        path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
                        y_path = os.path.join(path, 'y.npy')
                        w_path = os.path.join(path, 'rel_w.npy')
                        events = load_events(path, columns+weight_columns)
                    elif "GluGlutoHHto2B2G" in sample:
                        path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
                        y_path = os.path.join(path, 'y.npy')
                        w_path = os.path.join(path, 'rel_w.npy')
                        events = load_events(path, columns+weight_columns+columns_gen+columns_gen_ggHH)
                    else:
                        path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
                        y_path = os.path.join(path, 'y.npy')
                        w_path = os.path.join(path, 'rel_w.npy')
                        events = load_events(path, columns+weight_columns+columns_gen)

                # Check if files exist
                if not (os.path.exists(y_path)):
                    print(f"Missing y for {path}. Skipping.")
                    continue
                y = np.load(y_path)

                if data:
                    path_VBFMVA = os.path.join(base_path_VBFMVA, "individual_samples_data", era, sample)
                else:
                    path_VBFMVA = os.path.join(base_path_VBFMVA, "individual_samples"+"/", era, sample, syst)
                y_path_VBFMVA = os.path.join(path_VBFMVA, 'y.npy')
                # Check if files exist
                if not (os.path.exists(y_path_VBFMVA)):
                    print(f"Missing y for {path_VBFMVA}. Skipping.")
                    continue
                y_VBFMVA = np.load(y_path_VBFMVA)

            # All three must have the same length: ggHH y.npy, VBFMVA y.npy, and events
            # (which, when --boosted-folder is set, comes from the boosted parquet).
            n_y = y.shape[0]
            n_yvbf = y_VBFMVA.shape[0]
            n_events = len(events)
            if not (n_y == n_yvbf == n_events):
                raise ValueError(
                    f"Length mismatch for {era}/{sample} syst='{syst}': "
                    f"y(ggHH)={n_y}, y(VBFMVA)={n_yvbf}, events={n_events}. Aborting.")

            samples_input["score"].append(y)
            samples_input["VBFMVA_score"].append(y_VBFMVA)
            
            # samples_input["lumi"].append(np.array(events['lumi']))
            # samples_input["event"].append(np.array(events['event']))
            # samples_input["run"].append(np.array(events['run']))

            #samples_input["nonResReg_lead_bjet_hFlav"].append(np.array(events['nonResReg_lead_bjet_hFlav']))
            #samples_input["nonResReg_sublead_bjet_hFlav"].append(np.array(events['nonResReg_sublead_bjet_hFlav']))

            # samples_input["mass"].append(np.array(events['mass']))
            samples_input["dijet_mass"].append(np.array(events[dijet_mass_key]))
            samples_input["HHbbggCandidate_mass"].append(np.array(events[HH_mass_key]))
            for col in columns + columns_gen:
                if (col != dijet_mass_key) & (col != HH_mass_key):
                    if not(("gen" in col) & ((sample == "DDQCDGJET") | (data))):
                        samples_input[col].append(np.array(events[col]))

            if (data | (sample == "DDQCDGJET")):
                samples_input["lead_genPartFlav"].append(np.array([-999] * len(events[dijet_mass_key])))
                samples_input["sublead_genPartFlav"].append(np.array([-999] * len(events[dijet_mass_key])))

            for col in columns_gen_ggHH:
                if ((not data) and ("GluGlutoHHto2B2G" in sample) and (col in events.fields)):
                    samples_input[col].append(np.array(events[col]))
                else:
                    samples_input[col].append(np.array([-999.0] * len(events[dijet_mass_key])))

            if sample == "":
                sample = "Data"
            if sample in ff_sampledict.keys():
                sample = ff_sampledict[sample]
            print(sample)
            samples_input["sample"].append(np.full(y.shape[0], sample))

            if "16" in era:
                year = 2016
            elif "17" in era:
                year = 2017
            elif "18" in era:
                year = 2018
            elif "22" in era or "EE" in era:
                year = 2022
            elif "23" in era or "BPix" in era:
                year = 2023
            elif "24" in era:
                year = 2024
            elif "25" in era:
                year = 2025
            else:
                raise ValueError(f"Unknown era: {era}")
            samples_input["year"].append(np.full(y.shape[0], year))
            samples_input["era"].append(np.full(y.shape[0], era))

            # samples_input["is_boosted"].append(np.array(events["is_boosted"]))  
            # samples_input["y_proba"].append(np.array(events['y_proba']))

            for weight in weight_columns:
                if weight in events.fields:
                    samples_input[weight].append(np.array(events[weight]))
                else:
                    samples_input[weight].append(np.array(ak.ones_like(events['mass'])))  # Default weight if not provided

    # Concatenate all data
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["HHbbggCandidate_mass"] = np.concatenate(samples_input["HHbbggCandidate_mass"], axis=0)

    for col in columns + columns_gen + columns_gen_ggHH:
        if (col != dijet_mass_key) & (col != HH_mass_key):
            samples_input[col] = np.concatenate(samples_input[col], axis=0)

    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["year"] = np.concatenate(samples_input["year"], axis=0)
    samples_input["era"] = np.concatenate(samples_input["era"], axis=0)
    VBFMVA_scores = np.concatenate(samples_input["VBFMVA_score"], axis=0)
    samples_input["VBFMVA_score"] = [row for row in VBFMVA_scores]
    samples_input["VBFMVA_VBFHH"] = [row[0] for row in VBFMVA_scores]
    samples_input["VBFMVA_ggHH"] = [row[1] for row in VBFMVA_scores]   
    samples_input["VBFMVA_ResBkg"] = [row[2] for row in VBFMVA_scores]
    samples_input["VBFMVA_nonResBkg"] = [row[3] for row in VBFMVA_scores]
    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["score"] = [row for row in scores]
    samples_input["nonRes_score"] = [row[0] for row in scores]
    # samples_input["nonRes_singleH_score"] = [row[0] for row in scores]
    samples_input["ttH_score"] = [row[1] for row in scores]   
    samples_input["singleH_score"] = [row[2] for row in scores]
    samples_input["ggHH_score"] = [row[3] for row in scores]
    # samples_input["ggHH_score"] = [row[2] for row in scores]

    # Binary discriminators built from multiclass outputs.
    nonres_score = np.asarray(samples_input["nonRes_score"])
    # nonres_singleH_score = np.asarray(samples_input["nonRes_singleH_score"])
    tth_score = np.asarray(samples_input["ttH_score"])
    singleh_score = np.asarray(samples_input["singleH_score"])
    gghh_score = np.asarray(samples_input["ggHH_score"])

    denom_ttH = gghh_score + tth_score
    denom_nonres = gghh_score + singleh_score + nonres_score
    # denom_nonres = gghh_score + nonres_score

    samples_input["Dsig_vs_ttH"] = np.where(denom_ttH != 0, gghh_score / denom_ttH, 0.0)
    samples_input["Dsig_vs_nonres"] = np.where(denom_nonres != 0, gghh_score / denom_nonres, 0.0)

    # samples_input["is_boosted"] = np.concatenate(samples_input["is_boosted"], axis=0)
    # samples_input["y_proba"] = np.concatenate(samples_input["y_proba"], axis=0)
    for weight in weight_columns:
        samples_input[weight] = np.concatenate(samples_input[weight], axis=0)

    # convert to pandas dataframe
    samples_input = pd.DataFrame(samples_input)

    if boosted_folder is not None:
        samples_input.rename(columns={"y_proba": "boosted_score"}, inplace=True)


    return samples_input

def load_samples_VBFMVA_only(base_path, base_path_VBFMVA, eras, samples, data=False, syst=""):
    """Lightweight merge: only loads run/event/lumi + VBFMVA y.npy scores.
    Output is a thin parquet suitable as parquet B for attach_boosted_score.py.
    """
    key_cols = ["run", "event", "lumi"]
    rows = {col: [] for col in key_cols + ["sample", "year", "era",
                                            "VBFMVA_VBFHH", "VBFMVA_ggHH",
                                            "VBFMVA_ResBkg", "VBFMVA_nonResBkg"]}

    for era in eras:
        print(f"### {era} ###")
        for sample in samples:
            if (sample in ["GGJets", "DDQCDGJET", "TTGG", "TT", "TTG_10_100", "TTG_100_200", "TTG_200"]) and syst != "":
                continue
            if "GluGlutoHHto2B2G_EFTReweighted" in sample:
                continue

            if (era in ["2024", "2025"]) and (sample == "VHtoGG_M_125"):
                events_list, y_vbf_list = [], []
                ok = True
                for VHsample in ["WmHtoGG", "WpHtoGG", "ZHtoGG"]:
                    path = os.path.join(base_path, "individual_samples", era, VHsample, syst)
                    path_vbf = os.path.join(base_path_VBFMVA, "individual_samples", era, VHsample, syst)
                    parquet_f = os.path.join(path, "events.parquet")
                    y_f = os.path.join(path_vbf, "y.npy")
                    if not os.path.exists(parquet_f) or not os.path.exists(y_f):
                        print(f"Missing {VHsample} for {era}, skipping VHtoGG_M_125.")
                        ok = False
                        break
                    events_list.append(ak.from_parquet(parquet_f, columns=key_cols))
                    y_vbf_list.append(np.load(y_f))
                if not ok:
                    continue
                events = ak.concatenate(events_list)
                y_vbfmva = np.concatenate(y_vbf_list, axis=0)
            else:
                if era != "postEE" and (("TTG_" in sample) or (sample == "TT")):
                    continue

                if data:
                    path = os.path.join(base_path, "individual_samples_data", era, sample)
                    path_vbf = os.path.join(base_path_VBFMVA, "individual_samples_data", era, sample)
                else:
                    path = os.path.join(base_path, "individual_samples", era, sample, syst)
                    path_vbf = os.path.join(base_path_VBFMVA, "individual_samples", era, sample, syst)

                parquet_f = os.path.join(path, "events.parquet")
                y_f = os.path.join(path_vbf, "y.npy")

                if not os.path.exists(parquet_f):
                    print(f"Missing parquet for {era}/{sample}. Skipping.")
                    continue
                if not os.path.exists(y_f):
                    print(f"Missing VBFMVA y.npy for {era}/{sample}. Skipping.")
                    continue

                events = ak.from_parquet(parquet_f, columns=key_cols)
                y_vbfmva = np.load(y_f)

            n = len(events)
            if y_vbfmva.shape[0] != n:
                print(f"WARNING: event count mismatch for {era}/{sample}: parquet={n}, y.npy={y_vbfmva.shape[0]}. Skipping.")
                continue

            for col in key_cols:
                rows[col].append(np.array(events[col]))

            sample_name = ff_sampledict.get(sample, sample) if sample != "" else "Data"
            rows["sample"].append(np.full(n, sample_name))

            if "16" in era:           year = 2016
            elif "17" in era:         year = 2017
            elif "18" in era:         year = 2018
            elif "22" in era or "EE" in era:   year = 2022
            elif "23" in era or "BPix" in era: year = 2023
            elif "24" in era:         year = 2024
            elif "25" in era:         year = 2025
            else: raise ValueError(f"Unknown era: {era}")
            rows["year"].append(np.full(n, year))
            rows["era"].append(np.full(n, era))

            rows["VBFMVA_VBFHH"].append(y_vbfmva[:, 0])
            rows["VBFMVA_ggHH"].append(y_vbfmva[:, 1])
            rows["VBFMVA_ResBkg"].append(y_vbfmva[:, 2])
            rows["VBFMVA_nonResBkg"].append(y_vbfmva[:, 3])

            print(f"  loaded {n} events for {sample_name} / {era}")

    for col in rows:
        rows[col] = np.concatenate(rows[col], axis=0)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ggHH/VBFHH samples and optionally build a global all-era parquet.")
    parser.add_argument("base_path", type=str, help="Path to multiclass input with individual_samples")
    parser.add_argument("base_path_VBFMVA", type=str, help="Path to VBFMVA input with individual_samples")
    parser.add_argument("out_path", type=str, help="Path to output directory")
    parser.add_argument("--merge-all-eras", action="store_true", default=True, help="If set, also write one parquet merging all eras together.")
    parser.add_argument("--VBFMVA-only", action="store_true", default=False,
                        help="Only merge VBFMVA y.npy with key columns (run/event/lumi/sample/year/era) for use with attach_boosted_score.py.")
    parser.add_argument("--boosted-folder", type=str, default=None,
                        help="If set, read the ggHH events from this folder (file named events_boostedCat.parquet) "
                             "instead of base_path's ggHH folder. Expected to mirror the same "
                             "individual_samples/<era>/<sample>/<syst> structure. ggHH y.npy is still read from base_path.")
    args = parser.parse_args()

    # Output multiclass folder, one folder up from individual_samples.
    base_path = args.base_path
    print(base_path)

    base_path_VBFMVA = args.base_path_VBFMVA
    print(base_path_VBFMVA)

    samples = [
            "GGJets",
            "DDQCDGJET",
            "TTGG",
            # "TT",
            # "TTG_10_100",
            # "TTG_100_200",
            # "TTG_200",
            "ttHtoGG_M_125",
            "BBHToGG_M_125",
            "GluGluHToGG_M_125",
            "VBFHToGG_M_125",
            "VHtoGG_M_125",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00",
            "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24",
            "VBFHH_CV_1_C2V_1_C3_1",
            "VBFHH_CV_1_C2V_0_C3_1",
            "VBFHH_CV_2p12_C2V_3p87_C3_m5p96",
            "VBFHH_CV_m0p012_C2V_0p030_C3_10p2",
            "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94",
            "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39",
            "VBFHH_CV_1p74_C2V_1p37_C3_14p4",
            "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3",
            "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43",
            "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36",
            # # "GluGlutoHHto2B2G_EFTReweighted_1D_public",
            # # "GluGlutoHHto2B2G_EFTReweighted_1D_private",
            # # "GluGlutoHHto2B2G_EFTReweighted_2D_private",
            # # "GluGlutoHHto2B2G_EFTReweighted_3D_private",
    ]

    # Split up eras to merge for memory
    dict_run_eras = {}
    dict_run_eras["2016"] = {"mc" : ["2016preVFP", "2016postVFP"], "data": [
        # "2016preVFP_EraBv1", #no events?
        "2016preVFP_EraBv2", "2016preVFP_EraC", "2016preVFP_EraD", "2016preVFP_EraE", "2016preVFP_EraF", "2016postVFP_EraF", "2016postVFP_EraG", "2016postVFP_EraH",]}
    dict_run_eras["2017"] = {"mc" : ["2017"], "data": ["2017_EraB", "2017_EraC", "2017_EraD", "2017_EraE", "2017_EraF",]}
    dict_run_eras["2018"] = {"mc" : ["2018"], "data": ["2018_EraA", "2018_EraB", "2018_EraC", "2018_EraD",]}
    dict_run_eras["2022"] = {"mc" : ["preEE", "postEE"], "data": ["2022_EraC","2022_EraD","2022_EraE","2022_EraF","2022_EraG"]}
    dict_run_eras["2023"] = {"mc" : ["preBPix", "postBPix"], "data": [
        # "2023_EraC","2023_EraD",
        "2023_EraCv1_EG0", "2023_EraCv1_EG1", "2023_EraCv2_EG0", "2023_EraCv2_EG1", "2023_EraCv3_EG0", "2023_EraCv3_EG1", "2023_EraCv4_EG0", "2023_EraCv4_EG1", "2023_EraDv1_EG0", "2023_EraDv1_EG1", "2023_EraDv2_EG0", "2023_EraDv2_EG1"
        ]}
    dict_run_eras["2024"] = {"mc" : ["2024"], "data": ["2024_EraC_EG0", "2024_EraC_EG1", "2024_EraD_EG0", "2024_EraD_EG1", "2024_EraE_EG0", "2024_EraE_EG1", "2024_EraF_EG0", "2024_EraF_EG1", "2024_EraG_EG0", "2024_EraG_EG1", "2024_EraH_EG0", "2024_EraH_EG1", "2024_EraIv1_EG0", "2024_EraIv1_EG1", "2024_EraIv2_EG0", "2024_EraIv2_EG1"]}
    dict_run_eras["2025"] = {"mc" : ["2025"], "data": ["2025_EraCv1_EG0", "2025_EraCv1_EG1", "2025_EraCv1_EG2", "2025_EraCv1_EG3", "2025_EraCv2_EG0", "2025_EraCv2_EG1", "2025_EraCv2_EG2", "2025_EraCv2_EG3", "2025_EraDv1_EG0", "2025_EraDv1_EG1", "2025_EraDv1_EG2", "2025_EraDv1_EG3", "2025_EraEv1_EG0", "2025_EraEv1_EG1", "2025_EraEv1_EG2", "2025_EraEv1_EG3", "2025_EraFv1_EG0", "2025_EraFv1_EG1", "2025_EraFv1_EG2", "2025_EraFv1_EG3", "2025_EraFv2_EG0", "2025_EraFv2_EG1", "2025_EraFv2_EG2", "2025_EraFv2_EG3", "2025_EraGv1_EG0", "2025_EraGv1_EG1", "2025_EraGv1_EG2", "2025_EraGv1_EG3",]}

    # systs = []
    systs = [
        "ScaleEB_Zee_down",
        "ScaleEB_Zee_up",
        "ScaleEE_Zee_down",
        "ScaleEE_Zee_up",
        "ScaleEB_Zmmg_down",
        "ScaleEB_Zmmg_up",
        "ScaleEE_Zmmg_down",
        "ScaleEE_Zmmg_up",
        "Smearing_down",
        "Smearing_up",
        "jec_syst_Total_down",
        "jec_syst_Total_up",
        "jer_syst_down",
        "jer_syst_up",
        "jec_AK8_syst_Total_down",
        "jec_AK8_syst_Total_up",
        "jer_AK8_syst_down",
        "jer_AK8_syst_up"
    ]

    outpath = args.out_path
    os.makedirs(outpath, exist_ok=True)

    merged_all_eras = []

    if args.VBFMVA_only:
        for era in dict_run_eras.keys():
            print(f"Loading VBFMVA-only samples for {era}.")
            dict_era = dict_run_eras[era]
            df_mc = load_samples_VBFMVA_only(base_path, base_path_VBFMVA, dict_era["mc"], samples)
            merged_all_eras.append(df_mc)

        if merged_all_eras:
            merged_all_eras_df = pd.concat(merged_all_eras, ignore_index=True)
            out_file = f"{outpath}/merged_VBFMVA_only.parquet"
            merged_all_eras_df.to_parquet(out_file, engine='pyarrow')
            print(f"Wrote VBFMVA-only parquet: {out_file}")
            print(f"Columns: {list(merged_all_eras_df.columns)}")
            print(f"Shape: {merged_all_eras_df.shape}")
    else:
        for era in dict_run_eras.keys():
            print(f"Loading samples for {era}.")
            dict_era = dict_run_eras[era]

            merged_samples_MC = load_samples(base_path, dict_era["mc"], samples, boosted_folder=args.boosted_folder)
            merged_samples_data = load_samples(base_path, dict_era["data"], [""] ,data=True, boosted_folder=args.boosted_folder)

            merged_samples = pd.concat([merged_samples_MC, merged_samples_data], ignore_index=True)
            # merged_samples = merged_samples_MC
            # merged_samples.to_parquet(f"{outpath}/merged_samples_{era}_ggHHVBFHH.parquet", engine='pyarrow')
            merged_all_eras.append(merged_samples)

            # for syst in systs:
            #     print(syst)
            #     merged_samples_MC = load_samples(base_path, dict_era["mc"], samples, syst=syst, boosted_folder=args.boosted_folder)
            #     merged_samples_MC.to_parquet(f"{outpath}/merged_samples_{syst}_{era}_ggHHVBFHH.parquet", engine='pyarrow')
            #     print()

        # if args.merge_all_eras and merged_all_eras:
            # merged_all_eras_df = pd.concat(merged_all_eras, ignore_index=True)
            # merged_all_eras_df.to_parquet(f"{outpath}/merged_samples_all_ggHHVBFHH.parquet", engine='pyarrow')
            # print(f"Wrote all-era parquet: {outpath}/merged_samples_all_ggHHVBFHH.parquet")

        # Release nominal DataFrames before processing systematics.
        if "merged_all_eras_df" in locals():
            del merged_all_eras_df
        merged_all_eras.clear()
        del merged_all_eras
        if "merged_samples_MC" in locals():
            del merged_samples_MC
        if "merged_samples_data" in locals():
            del merged_samples_data
        if "merged_samples" in locals():
            del merged_samples
        gc.collect()
        print("[DEBUG] Released nominal DataFrames from memory.")

        # Process one object-level systematic at a time and merge all years.
        for syst in systs:
            print(f"\n{'=' * 80}")
            print(f"Processing all years for systematic: {syst}")
            print(f"{'=' * 80}")

            merged_syst_all_eras = []
            for year_label, dict_era in dict_run_eras.items():
                print(f"Loading {syst} for {year_label}")
                merged_syst_MC = load_samples(
                    base_path,
                    dict_era["mc"],
                    samples,
                    syst=syst,
                    boosted_folder=args.boosted_folder,
                )
                # Optional: retain the individual-year output.
                per_year_output = (f"{outpath}/merged_samples_{syst}_{year_label}_ggHHVBFHH.parquet")
                # merged_syst_MC.to_parquet(per_year_output, engine="pyarrow", index=False)
                # print(f"Wrote: {per_year_output}")

                merged_syst_all_eras.append(merged_syst_MC)

            if merged_syst_all_eras:
                merged_syst_all_df = pd.concat(merged_syst_all_eras, ignore_index=True)

                all_years_output = (f"{outpath}/merged_samples_{syst}_all_ggHHVBFHH.parquet")

                merged_syst_all_df.to_parquet(all_years_output, engine="pyarrow", index=False)

                print(f"Wrote all-year systematic parquet: {all_years_output}")

                del merged_syst_all_df
                del merged_syst_all_eras