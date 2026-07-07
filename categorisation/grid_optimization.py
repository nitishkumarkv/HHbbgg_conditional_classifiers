#!/usr/bin/env python3
"""
Grid-search-based categorization for HHbbgg analysis.

Replaces Optuna TPE optimization with exhaustive multi-dimensional grid search
using cumulative histograms for O(1) per-evaluation cost.

Algorithm:
  Phase 1: Coarse grid (n_coarse_bins pts/dim) -> build 3D histograms ->
           cumulative sums -> evaluate all n_coarse_bins^3 threshold
           combinations via O(1) lookups.
  Phase 2: Determine feasible score-space volume from sideband constraints.
  Phase 3: Per coarse bin, fine histogram scan: n_fine_bins equal bins within
           the coarse bin + 1 extra bin covering [coarse_upper, overall_max].
           Build 3D histograms, compute reverse cumsums, evaluate all
           (n_fine_bins+1)^3 approx 1M threshold combinations via fully vectorized
           numpy operations.

Usage:
 # base can be /eos/cms/store/group/phys_higgs/nonresonant_HH/PrivateProd/Yuxiang/manos_bbgg_verify/correct_data/scored_samples/merged
 # input file can be ${base}/merged_scored_events_replaced_2024_2025.parquet
  conda run -n XXX python3 grid_search_categorization.py   \
  --input_file XXX_merged.parquet   \
  --output_dir .   --base_path XXXX --optuna_folder grid_search_test_lowTh8  \
  --signal_samples '[GluGluToHH_kl-1p00_kt-1p00_c2-0p00]' --vbf_thresholds 0.774   --apply_boosted_veto \
  --n_categories 3   --n_coarse_bins 500  \
  --n_fine_bins 100 --cat_sideband_max 15,100,150 --side_band_threshold_low 10
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

# -- Pre-computed interpolation coefficients ------------------------------
# Mass windows (GeV):
#   left  sideband: [100.0, 120.0]  width_L  = 20.0, centroid_L  = 110.0
#   signal region:   [120.0, 130.0]  width_SR = 10.0, centroid_SR = 125.0
#   right sideband:  [130.0, 180.0]  width_R  = 50.0, centroid_R  = 155.0
#
# dens_SR = w_L * (sumW_L / width_L) + w_R * (sumW_R / width_R)
#   with w_L = (centroid_R - centroid_SR) / (centroid_R - centroid_L)
#        w_R = (centroid_SR - centroid_L) / (centroid_R - centroid_L)
# b_SR_interp = dens_SR * width_SR
#             = coeff_L * sumW_L + coeff_R * sumW_R

# -- Mass window definitions ----------------------------------------------
MASS_LEFT_SB = (100.0, 120.0)
MASS_SR = (120, 130)
MASS_RIGHT_SB = (130.0, 180.0)

_WIDTH_L = MASS_LEFT_SB[1] - MASS_LEFT_SB[0]  # 20.0
_WIDTH_SR = MASS_SR[1] - MASS_SR[0]  # 10.0
_WIDTH_R = MASS_RIGHT_SB[1] - MASS_RIGHT_SB[0]
_CENTROID_L = 0.5 * (MASS_LEFT_SB[0] + MASS_LEFT_SB[1])  # 110.0
_CENTROID_SR = 0.5 * (MASS_SR[0] + MASS_SR[1])  # 125.0
_CENTROID_R = 0.5 * (MASS_RIGHT_SB[0] + MASS_RIGHT_SB[1])  # 155.0
_SPAN = _CENTROID_R - _CENTROID_L  # 45.0

_W_L = (_CENTROID_R - _CENTROID_SR) / _SPAN  # 2/3
_W_R = (_CENTROID_SR - _CENTROID_L) / _SPAN  # 1/3

COEFF_L = (_W_L * _WIDTH_SR) / _WIDTH_L  # 1/3
COEFF_R = (_W_R * _WIDTH_SR) / _WIDTH_R  # 1/15

# -- Interpolation samples (smoothly-falling backgrounds) -----------------
INTERP_SAMPLES = {"TTGG", "GGJets", "DDQCDGJets"}

# -- Score column names ---------------------------------------------------
SCORE_COLS = [
    "is_nonRes_bkg_score",
    "is_Res_bkg_score",
    "is_ggHH_sig_score",
    "is_VBFHH_sig_score",
]


# --------------------------------------------------------------------------
#                          GRID SEARCH CATEGORIZER                         
# --------------------------------------------------------------------------

class GridSearchCategorizer:
    """Exhaustive grid-search categorization using cumulative histograms."""

    def __init__(
        self,
        base_path,
        cat_folder=None,
        signal_class=2,
        signal_class_name="ggHH",
        signal_samples=None,
        bkg_samples=None,
        bkg_classes=None,
        bkg_class_names=None,
        n_categories=5,
        side_band_threshold_low=10,
        side_band_threshold_high=15,
        cat_sideband_max=None,
        sig_type=None,
        vbfhh_class=None,
        vbfhh_samples=None,
        vbfhh_n_scan_points=1000,
        vbfhh_sideband_threshold=10.0,
        vbfhh_n_categories=1,
        output_dir=None,
        input_file=None,
        use_individual_samples=False,
        vbf_thresholds=None,
        n_coarse_bins=100,
        n_fine_bins=100,
        require_sb_ratio=False,
    ):
        # -- Paths and I/O --------------------------------------------
        self.base_path = base_path
        self.cat_folder = cat_folder or "grid_search_categorization"
        self.output_dir = output_dir or base_path
        self.input_file = input_file
        self.use_individual_samples = use_individual_samples

        # -- Signal / background definitions --------------------------
        self.signal_class = signal_class
        self.signal_class_name = signal_class_name
        self.bkg_classes = bkg_classes if bkg_classes is not None else [0, 1]
        if bkg_class_names is not None:
            self.bkg_class_names = bkg_class_names
        else:
            self.bkg_class_names = [f"bkg_{i}" for i in self.bkg_classes]

        # -- Categorization parameters --------------------------------
        self.n_categories = n_categories
        self.side_band_threshold_low = side_band_threshold_low
        self.side_band_threshold_high = side_band_threshold_high
        if cat_sideband_max is not None:
            self.cat_sideband_max = [int(x.strip()) for x in cat_sideband_max.split(",")]
        else:
            self.cat_sideband_max = [25, 100, 500]  # defaults for cat1, cat2, cat3
        self.sig_type = sig_type

        # -- Grid search parameters -----------------------------------
        self.n_coarse_bins = n_coarse_bins
        self.n_fine_bins = n_fine_bins
        self.require_sb_ratio = require_sb_ratio

        # -- VBFHH parameters -----------------------------------------
        self.vbfhh_class = vbfhh_class
        self.vbfhh_samples = vbfhh_samples
        self.vbfhh_n_scan_points = vbfhh_n_scan_points
        self.vbfhh_sideband_threshold = vbfhh_sideband_threshold
        self.vbfhh_n_categories = vbfhh_n_categories
        self.vbf_thresholds = vbf_thresholds

        # -- Pre-selection flags --------------------------------------
        self.apply_preselection = True
        self.apply_boosted_veto = False

        # -- Parse sample lists ---------------------------------------
        self.signal_samples = self._parse_bracket_list(signal_samples, "signal_samples")
        self.bkg_samples = self._parse_bracket_list(
            bkg_samples, "bkg_samples",
            default=[
                "VBFHToGG", "VHToGG", "ttHToGG", "GluGluHToGG",
                "TTGG", "GGJets", "DDQCDGJets",
            ],
        )
        if vbfhh_samples is not None:
            self.vbfhh_samples = self._parse_bracket_list(
                vbfhh_samples, "vbfhh_samples"
            )
        else:
            self.vbfhh_samples = None

        # -- Data arrays (filled by load_samples) ---------------------
        self.dijet_mass_key = "nonResReg_vbfpair_dijet_mass_DNNreg"
        self.scores_all = None
        self.mass_all = None
        self.dijet_mass_all = None
        self.weights_all = None
        self.weights_unscaled_all = None
        self.labels_all = None
        self.samples_all = None
        self.weight_scale = 1.44

        # -- Grid search state ----------------------------------------
        self.coarse_edges = None       # list of 3 edge arrays
        self.coarse_histograms = {}    # dict of 3D histogram arrays
        self.coarse_cumsums = {}       # dict of 3D cumulative sum arrays
        self.coarse_results = None     # structured array of all valid combos

    # ----------------------------------------------------------------------
    #  STATIC HELPERS
    # ----------------------------------------------------------------------

    @staticmethod
    def _parse_bracket_list(raw, name, default=None):
        """Parse '[A,B,C]' -> ['A','B','C']."""
        if raw is None:
            if default is not None:
                return list(default)
            raise ValueError(f"Missing required argument: {name}")
        raw = raw.strip()
        if not (raw.startswith("[") and raw.endswith("]")):
            raise ValueError(f"{name} must be bracket-enclosed, e.g. '[A,B,C]'")
        inner = raw[1:-1]
        items = [s.strip() for s in inner.split(",") if s.strip()]
        if not items:
            raise ValueError(f"{name} is empty inside brackets.")
        return items

    @staticmethod
    def asymptotic_significance(s, b):
        """Asimov Z = sqrt(2 * ((s+b)*ln(1+s/b) - s))."""
        if s <= 0 or b <= 0:
            return 0.0
        return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))

    @staticmethod
    def _compute_reverse_cumsum(hist):
        """Compute reverse cumulative sum (survival function).

        C[i,j,k] = sum_{a>=i, b>=j, c>=k} H[a,b,c]

        Makes one contiguous reversed copy then cumsums forward on
        contiguous memory (much faster than repeated np.flip calls).
        """
        # One contiguous reversed copy, then forward cumsum on dense memory
        rev = np.ascontiguousarray(hist[::-1, ::-1, ::-1])
        for axis in range(hist.ndim):
            rev = np.cumsum(rev, axis=axis)
        # Reverse view back - O(1), no copy
        return rev[::-1, ::-1, ::-1]

    @staticmethod
    def _query_region(cumsum, i0, i1, i2):
        """Sum over region: dim0 >= i0, dim1 >= i1, dim2 >= i2.

        With reverse cumsum, this is just a direct lookup.
        All cuts are '>' after 1-bg_score transformation.
        """
        n0, n1, n2 = cumsum.shape
        if i0 >= n0 or i1 >= n1 or i2 >= n2:
            return 0.0
        return float(cumsum[i0, i1, i2])

    @staticmethod
    def _zero_region_inplace(hist, i0, i1, i2):
        """Zero out region dim0 >= i0, dim1 >= i1, dim2 >= i2 (in-place)."""
        hist[i0:, i1:, i2:] = 0.0

    # ----------------------------------------------------------------------
    #  DATA LOADING  (reused from bayesian_categorization_withVBFHH.py)
    # ----------------------------------------------------------------------

    def preselection(self, events, scores):
        """Apply pre-selection cuts."""
        mass_bool = (events.mass > 100) & (events.mass < 180)
        dijet_mass_bool = (events[self.dijet_mass_key] > 80) & (
            events[self.dijet_mass_key] < 190
        )
        ggHH_bool = scores[:, self.signal_class] > 0

        mask = mass_bool & dijet_mass_bool & ggHH_bool
        try:
            lead_mvaID_bool = events.lead_mvaID > -0.7
            sublead_mvaID_bool = events.sublead_mvaID > -0.7
            mask = mask & lead_mvaID_bool & sublead_mvaID_bool
        except Exception:
            pass
        if self.apply_boosted_veto:
            mask = mask & (events.is_boosted == False)  # noqa: E712
        events = events[mask]
        scores = scores[mask]
        return events, scores

    def load_samples_merged(self):
        """Load scored events from a single merged parquet file."""
        if self.input_file is None:
            parquet_file = os.path.join(
                self.base_path,
                "scored_samples",
                "merged",
                "merged_scored_events_boostedCat.parquet",
            )
        else:
            parquet_file = self.input_file

            # mergerScript.py writes the _boostedCat file. Flag the common
            # failure mode where an older pre-existing sibling is selected.
            if os.path.basename(parquet_file) == "merged_scored_events.parquet":
                boosted_file = os.path.join(
                    os.path.dirname(parquet_file),
                    "merged_scored_events_boostedCat.parquet",
                )
                if (
                    os.path.exists(parquet_file)
                    and os.path.exists(boosted_file)
                    and os.path.getmtime(boosted_file) > os.path.getmtime(parquet_file)
                ):
                    print(
                        "WARNING: The selected merged_scored_events.parquet is older "
                        "than mergerScript.py's output:\n"
                        f"  selected: {parquet_file}\n"
                        f"  newer:    {boosted_file}\n"
                        "Use the newer _boostedCat file to match the per-era inputs."
                    )

        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Merged parquet file not found: {parquet_file}")

        print(f"[load_samples_merged] Reading {parquet_file} ...")

        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc

        pq_schema = pq.read_schema(parquet_file)
        pq_cols = {c for c in pq_schema.names}

        columns_needed = [
            "sample", "mass", self.dijet_mass_key, "weight_tot",
        ]
        extra_cols = {
            "lead_mvaID", "sublead_mvaID",
            "lead_genPartFlav", "sublead_genPartFlav", "is_boosted",
        }
        for c in sorted(extra_cols):
            if c in pq_cols:
                columns_needed.append(c)

        for c in SCORE_COLS:
            if c not in pq_cols:
                raise RuntimeError(f"Required score column '{c}' not found.")
        columns_needed.extend(SCORE_COLS)

        table = pq.read_table(parquet_file, columns=columns_needed)
        N = len(table)
        print(f"[load_samples_merged] Loaded {N} events.")

        weights_unscaled = table.column("weight_tot").to_numpy()

        # apply scale to GGJets and DDQCDGJets with nonRes_score < 0.05: 1.44
        def scale_weight(events, cut_mask, scale_factor = 1, scale_samples = []):
            """Scale weights for a specific sample."""
            if len(scale_samples) > 0:
                mask = pc.is_in(events["sample"], pa.array(scale_samples))
            mask = pc.and_(mask, cut_mask)
            rw = events.column("weight_tot")
            scaled_weights = pc.if_else(mask, pc.multiply(rw, scale_factor), rw)
            return events.set_column(events.schema.get_field_index("weight_tot"), "weight_tot", scaled_weights)
        nonres_cut = pc.less(table["is_nonRes_bkg_score"], 0.05)
        scale_samples = ["GGJets", "DDQCDGJets"]
        table = scale_weight(table, nonres_cut, scale_factor=self.weight_scale, scale_samples=scale_samples)
        print(f"  Scaled weights for {scale_samples} with nonRes_score < 0.05 by {self.weight_scale}")

        scores = np.column_stack([
            table.column(c).to_numpy() for c in SCORE_COLS
        ])
        # make a temp df without score columns for preselection
        non_score_cols = [c for c in columns_needed if c not in SCORE_COLS]
        table_filtered = table.select(non_score_cols)
        data = table_filtered.to_pandas()

        if self.apply_preselection:
            data, scores = self.preselection(data, scores)
            weights_unscaled = weights_unscaled[data.index.to_numpy()]

            # TTG / TTGG-like samples filter
            if "lead_mvaID" in pq_cols and "sublead_mvaID" in pq_cols:
                sel = (
                    data["sample"].str.startswith("TT")
                    & (data["lead_genPartFlav"] == 1)
                    & (data["sublead_genPartFlav"] == 1)
                )
                data = data[~sel]
                weights_unscaled = weights_unscaled[~sel.to_numpy()]
                scores = scores[~sel.to_numpy()]

        if len(scores) == 0:
            raise RuntimeError("[load_samples_merged] No events remain after selection.")

        # Assign labels
        data["labels"] = 0
        data.loc[data["sample"].isin(self.signal_samples), "labels"] = 1
        data.loc[
            data["sample"].str.contains("GluGlutoHHto2B2G") & (data["labels"] != 1),
            "labels",
        ] = -1

        df = pd.DataFrame({
            "diphoton_mass": data["mass"],
            "dijet_mass": data[self.dijet_mass_key],
            "weights": data["weight_tot"],
            "labels": data["labels"],
            "sample": data["sample"],
        })

        self.scores_all = scores  # (N,4) original: [nonRes, Res, ggHH, VBFHH]
        # Transformed scores for histogram: [1-nonRes, 1-Res, ggHH, VBFHH]
        # After transform, ALL cuts are '>' direction (larger is better)
        self.scores_trans = scores.astype(np.float64).copy()
        self.scores_trans[:, 0] = 1.0 - self.scores_trans[:, 0]  # nonRes -> inv
        self.scores_trans[:, 1] = 1.0 - self.scores_trans[:, 1]  # Res -> inv
        # scores_trans[:, 2] = ggHH (unchanged)
        # scores_trans[:, 3] = VBFHH (unchanged)
        self.mass_all = df["diphoton_mass"].to_numpy()
        self.dijet_mass_all = df["dijet_mass"].to_numpy()
        self.weights_all = df["weights"].to_numpy()
        self.weights_unscaled_all = weights_unscaled
        self.labels_all = df["labels"].to_numpy()
        self.samples_all = df["sample"].to_numpy()

        in_peak = (df["diphoton_mass"] > MASS_SR[0]) & (df["diphoton_mass"] < MASS_SR[1])
        print(f"Background weight: {df.loc[df['labels'] == 0, 'weights'].sum():.3g}")
        print(f"Signal weight:     {df.loc[df['labels'] == 1, 'weights'].sum():.3g}")
        print(
            f"Bkg weight {MASS_SR} GeV: "
            f"{df.loc[(df['labels'] == 0) & in_peak, 'weights'].sum():.3g}"
        )
        print(
            f"Sig weight {MASS_SR} GeV: "
            f"{df.loc[(df['labels'] == 1) & in_peak, 'weights'].sum():.3g}"
        )
        return df

    # ----------------------------------------------------------------------
    #  BACKGROUND ESTIMATION  (reused from original)
    # ----------------------------------------------------------------------

    def _sr_from_sidebands_linear(self, mass, weights):
        """Estimate SR background from sideband densities.

        Returns (b_sr, var_b_sr).
        """
        m = mass
        w = weights

        L = (m >= MASS_LEFT_SB[0]) & (m < MASS_LEFT_SB[1])
        R = (m >= MASS_RIGHT_SB[0]) & (m < MASS_RIGHT_SB[1])

        sumw_L = w[L].sum()
        sumw2_L = (w[L] ** 2).sum()
        sumw_R = w[R].sum()
        sumw2_R = (w[R] ** 2).sum()

        dens_L = sumw_L / _WIDTH_L if _WIDTH_L > 0 else 0.0
        var_dens_L = sumw2_L / (_WIDTH_L ** 2) if _WIDTH_L > 0 else 0.0
        dens_R = sumw_R / _WIDTH_R if _WIDTH_R > 0 else 0.0
        var_dens_R = sumw2_R / (_WIDTH_R ** 2) if _WIDTH_R > 0 else 0.0

        have_L = sumw_L > 0.0
        have_R = sumw_R > 0.0

        if have_L and have_R:
            dens = _W_L * dens_L + _W_R * dens_R
            var_dens = _W_L ** 2 * var_dens_L + _W_R ** 2 * var_dens_R
        elif have_L:
            dens = dens_L
            var_dens = var_dens_L
        elif have_R:
            dens = dens_R
            var_dens = var_dens_R
        else:
            return 0.0, 0.0

        b_sr = dens * _WIDTH_SR
        var_b_sr = var_dens * (_WIDTH_SR ** 2)
        return float(b_sr), float(var_b_sr)

    def _indices_to_thresholds(self, i0, i1, i2):
        """Convert bin indices to score thresholds.

        After 1-bg_score transformation, ALL cuts are '>' (lower bound).
        Threshold = edge[index].
        Returns (th_nonRes_inv, th_Res_inv, th_ggHH).
        """
        th0 = float(self.coarse_edges[0][i0])
        th1 = float(self.coarse_edges[1][i1])
        th2 = float(self.coarse_edges[2][i2])
        return th0, th1, th2

    def _trans_thresholds_to_original(self, th_inv0, th_inv1, th_sig):
        """Convert inverted thresholds back to original score space.

        Original: nonRes_score < th_nonRes, Res_score < th_Res, ggHH_score > th_ggHH
        Inverted: (1-nonRes) > th_inv0, (1-Res) > th_inv1, ggHH > th_sig to
          th_nonRes = 1 - th_inv0, th_Res = 1 - th_inv1, th_ggHH = th_sig
        """
        return 1.0 - th_inv0, 1.0 - th_inv1, th_sig

    # ----------------------------------------------------------------------
    #  PHASE 1: COARSE GRID HISTOGRAMS + EVALUATION
    # ----------------------------------------------------------------------

    def _build_coarse_histograms(self):
        """Build 3D weighted histograms (n_bins X n_bins X n_bins).

        Uses transformed scores: [1-nonRes, 1-Res, ggHH].
        After transformation, ALL cuts are '>' (larger is better).
        Histogram axes: dim0=nonRes_inv, dim1=Res_inv, dim2=ggHH.

        Histograms built (key -> content):
          'sig_SR'          -> signal weight in SR
          'interp_LSB'      -> interp-sample bkg weight in left SB
          'interp_RSB'      -> interp-sample bkg weight in right SB
          'noninterp_SR'    -> non-interp bkg weight in SR
          'bkg_sideband'    -> bkg weight in sidebands (excl. Data, for constraints)
        """
        n_bins = self.n_coarse_bins
        # Adaptive bin ranges: bg_inv scores in [0, 1], sig in [0.8, 1.0]
        bin_range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        bins = [n_bins, n_bins, n_bins]

        # Build boolean masks for efficient slicing
        in_LSB = (self.mass_all >= MASS_LEFT_SB[0]) & (self.mass_all < MASS_LEFT_SB[1])
        in_SR = (self.mass_all >= MASS_SR[0]) & (self.mass_all < MASS_SR[1])
        in_RSB = (self.mass_all >= MASS_RIGHT_SB[0]) & (self.mass_all < MASS_RIGHT_SB[1])
        in_sideband = in_LSB | in_RSB

        is_signal = self.labels_all == 1
        is_bkg = self.labels_all == 0
        is_data = self.samples_all == "Data"
        is_interp = np.isin(self.samples_all, list(INTERP_SAMPLES))
        is_noninterp_bkg = np.isin(self.samples_all, self.bkg_samples) & ~is_interp

        # Use transformed scores: [nonRes_inv, Res_inv, ggHH]
        scores_3d = self.scores_trans[:, [0, 1, 2]]
        weights = self.weights_all

        print(f"[Phase 1] Building 3D histograms ({n_bins}^3 = {n_bins**3:,} cells) ...")
        print(f"  Axes: nonRes_inv in {bin_range[0]}, Res_inv in {bin_range[1]}, ggHH in {bin_range[2]}")
        data_sideband_mask = is_data & in_sideband
        bkg_sideband_mask = is_bkg & ~is_data & in_sideband
        print("  Sideband summary after selection:")
        print(f"    Data sideband events: {data_sideband_mask.sum():,}")
        print(f"    Background sideband events: {bkg_sideband_mask.sum():,}")
        print(f"    Background sideband sumW: {weights[bkg_sideband_mask].sum():.6g}")
        t0 = time.time()

        # -- H_sig_SR --
        mask = is_signal & in_SR
        print(f"  sig_SR: {mask.sum():,} events")
        H_sig_SR, self.coarse_edges = np.histogramdd(
            scores_3d[mask], bins=bins, range=bin_range, weights=weights[mask]
        )

        # -- H_interp_LSB --
        mask = is_interp & in_LSB
        print(f"  interp_LSB: {mask.sum():,} events")
        H_interp_LSB, _ = np.histogramdd(
            scores_3d[mask], bins=bins, range=bin_range, weights=weights[mask]
        )

        # -- H_interp_RSB --
        mask = is_interp & in_RSB
        print(f"  interp_RSB: {mask.sum():,} events")
        H_interp_RSB, _ = np.histogramdd(
            scores_3d[mask], bins=bins, range=bin_range, weights=weights[mask]
        )

        # -- H_noninterp_SR --
        mask = is_noninterp_bkg & in_SR
        print(f"  noninterp_SR: {mask.sum():,} events")
        H_noninterp_SR, _ = np.histogramdd(
            scores_3d[mask], bins=bins, range=bin_range, weights=weights[mask]
        )

        # -- H_bkg_sideband -- (exclude Data, as in original code)
        mask = is_bkg & ~is_data & in_sideband
        print(f"  bkg_sideband: {mask.sum():,} events")
        H_bkg_sideband, _ = np.histogramdd(
            scores_3d[mask], bins=bins, range=bin_range, weights=weights[mask]
        )

        # -- H_data_LSB / H_data_RSB -- split sideband for SB ratio constraint
        mask_L = is_data & in_LSB
        print(f"  data_LSB: {mask_L.sum():,} events")
        H_data_LSB, _ = np.histogramdd(
            scores_3d[mask_L], bins=bins, range=bin_range, weights=weights[mask_L]
        )
        mask_R = is_data & in_RSB
        print(f"  data_RSB: {mask_R.sum():,} events")
        H_data_RSB, _ = np.histogramdd(
            scores_3d[mask_R], bins=bins, range=bin_range, weights=weights[mask_R]
        )

        # Event COUNT histogram (unweighted) for per-bin event retrieval
        mask = ~is_data # should not include Data for counting either
        H_count, _ = np.histogramdd(scores_3d[mask], bins=bins, range=bin_range)

        self.coarse_histograms = {
            "sig_SR": H_sig_SR,
            "interp_LSB": H_interp_LSB,
            "interp_RSB": H_interp_RSB,
            "noninterp_SR": H_noninterp_SR,
            "bkg_sideband": H_bkg_sideband,
            "data_LSB": H_data_LSB,
            "data_RSB": H_data_RSB,
            "count": H_count,
        }

        dt = time.time() - t0
        print(f"[Phase 1] Histograms built in {dt:.1f}s")

    def _subtract_removed_events(self, mask):
        """Incrementally subtract removed events from existing histograms.

        Builds histograms only for the masked subset (typically << total),
        then subtracts them from self.coarse_histograms in-place.
        Much faster than rebuilding all histograms from scratch.
        The raw arrays (scores, weights, masses, etc.) must NOT have been
        trimmed yet when this is called.
        """
        n_bins = self.n_coarse_bins
        # Use existing coarse_edges for exact bin alignment
        bins = self.coarse_edges

        # Slice to masked events
        s3d = self.scores_trans[mask][:, [0, 1, 2]]
        w_sub = self.weights_all[mask]
        m_sub = self.mass_all[mask]
        l_sub = self.labels_all[mask]
        sp_sub = self.samples_all[mask]

        # Mass-region masks
        in_LSB = (m_sub >= MASS_LEFT_SB[0]) & (m_sub < MASS_LEFT_SB[1])
        in_SR = (m_sub >= MASS_SR[0]) & (m_sub < MASS_SR[1])
        in_RSB = (m_sub >= MASS_RIGHT_SB[0]) & (m_sub < MASS_RIGHT_SB[1])
        in_sideband = in_LSB | in_RSB

        is_signal = l_sub == 1
        is_bkg = l_sub == 0
        is_data = sp_sub == "Data"
        is_interp = np.isin(sp_sub, list(INTERP_SAMPLES))
        is_noninterp_bkg = np.isin(sp_sub, self.bkg_samples) & ~is_interp

        # -- Build and subtract each layer -----------------------------
        H = self.coarse_histograms

        m = is_signal & in_SR
        H["sig_SR"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_interp & in_LSB
        H["interp_LSB"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_interp & in_RSB
        H["interp_RSB"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_noninterp_bkg & in_SR
        H["noninterp_SR"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_bkg & ~is_data & in_sideband
        H["bkg_sideband"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_data & in_LSB
        H["data_LSB"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = is_data & in_RSB
        H["data_RSB"] -= np.histogramdd(s3d[m], bins=bins, weights=w_sub[m])[0]

        m = ~is_data
        H["count"] -= np.histogramdd(s3d[m], bins=bins)[0]

    def _compute_coarse_cumsums(self):
        """Compute 3D reverse cumulative sums (survival function).

        Merges interp_LSB, interp_RSB, noninterp_SR into a single 'bkg_SR'
        layer before cumsum (they are only used as a linear combination),
        reducing cumsum calls from 6 -> 4.
        """
        print("[Phase 1] Computing reverse cumulative sums ...")
        t0 = time.time()

        H = self.coarse_histograms
        C = {}

        # Merge: bkg_SR = COEFF_L * interp_LSB + COEFF_R * interp_RSB + noninterp_SR
        H_bkg_SR = (COEFF_L * H["interp_LSB"]
                    + COEFF_R * H["interp_RSB"]
                    + H["noninterp_SR"])

        # Compute cumsums only for the 4 independent layers
        for key, data in [
            ("sig_SR", H["sig_SR"]),
            ("bkg_SR", H_bkg_SR),
            ("bkg_sideband", H["bkg_sideband"]),
            ("data_LSB", H["data_LSB"]),
            ("data_RSB", H["data_RSB"]),
            ("count", H["count"]),
        ]:
            C[key] = self._compute_reverse_cumsum(data)

        self.coarse_cumsums = C
        dt = time.time() - t0
        print(f"[Phase 1] Cumsums computed in {dt:.1f}s")

    def _evaluate_single_threshold(self, i0, i1, i2):
        """Evaluate Z for a single threshold combination.

        All cuts are '>': nonRes_inv > edge[0][i0], Res_inv > edge[1][i1],
        ggHH > edge[2][i2].

        Returns (s, b, b_sideband, Z) or (0, 0, 0, 0) if invalid.
        """
        C = self.coarse_cumsums

        s = self._query_region(C["sig_SR"], i0, i1, i2)
        if s <= 0:
            return 0.0, 0.0, 0.0, 0.0

        b_interp_L = self._query_region(C["interp_LSB"], i0, i1, i2)
        b_interp_R = self._query_region(C["interp_RSB"], i0, i1, i2)
        b_interp = COEFF_L * b_interp_L + COEFF_R * b_interp_R

        b_noninterp = self._query_region(C["noninterp_SR"], i0, i1, i2)
        b_total = max(b_interp + b_noninterp, 1e-9)

        b_side = self._query_region(C["bkg_sideband"], i0, i1, i2)

        Z = self.asymptotic_significance(s, b_total)
        return float(s), float(b_total), float(b_side), float(Z)

    def _phase1_coarse_evaluation(self):
        """Evaluate threshold combinations, skipping empty grid cells.

        Only cells with H_count > 0 (actual events in the 3D bin) are
        considered. Cells with count=0 have cumsum values identical to
        a neighboring cell, so they are duplicates for Z evaluation.

        All cuts are '>' direction (after 1-bg_score transformation).
        Fully vectorized - no Python loops.
        """
        n = self.n_coarse_bins
        total_combos = n * n * n
        print(f"[Phase 1] Evaluating threshold combinations (vectorized) ...")
        t0 = time.time()

        C = self.coarse_cumsums

        # -- Pre-filter: find cells with events AND S>0 AND sideband OK -
        H_cnt = self.coarse_histograms["count"]  # sparse: ~few% non-zero
        candidate = (
            (H_cnt > 0)
            & (C["sig_SR"] > 0)
            & (C["bkg_sideband"] >= self.side_band_threshold_low)
        )
        if self.require_sb_ratio:
            candidate = (
                candidate
                & (C["data_LSB"] > 0)
                & (C["data_RSB"] / C["data_LSB"] < _WIDTH_R / _WIDTH_L)
            )
        idx = np.where(candidate)
        n_cand = len(idx[0])
        pct = 100.0 * n_cand / total_combos
        sb_ratio_str = f", RSB/LSB<{_WIDTH_R/_WIDTH_L:.2f}" if self.require_sb_ratio else ""
        print(f"  Candidates (has events, S>0, sb>={self.side_band_threshold_low}{sb_ratio_str}): "
              f"{n_cand:,} ({pct:.2f}% of {total_combos:,})")

        if n_cand == 0:
            raise RuntimeError(
                "[Phase 1] No valid threshold combinations found. "
                "Check sideband thresholds."
            )

        # -- Compute B_total and Z only for candidate cells -------------
        S_cand = C["sig_SR"][idx]
        B_side_cand = C["bkg_sideband"][idx]
        B_total_cand = np.maximum(C["bkg_SR"][idx], 1e-9)

        with np.errstate(invalid="ignore", divide="ignore"):
            Z_cand = np.sqrt(
                2.0 * ((S_cand + B_total_cand)
                       * np.log(1.0 + S_cand / B_total_cand)
                       - S_cand)
            )

        # -- Filter to Z > 0 ------------------------------------------
        valid = Z_cand > 0
        n_valid = np.count_nonzero(valid)

        dt = time.time() - t0
        print(f"[Phase 1] Evaluation done in {dt:.1f}s, {n_valid:,} valid combos")

        if n_valid == 0:
            raise RuntimeError(
                "[Phase 1] No valid threshold combinations found. "
                "Check sideband thresholds."
            )

        # -- Gather valid results --------------------------------------
        i0_arr = idx[0][valid].astype(np.int32)
        i1_arr = idx[1][valid].astype(np.int32)
        i2_arr = idx[2][valid].astype(np.int32)
        Z_valid = Z_cand[valid]
        S_valid = S_cand[valid]
        B_valid = B_total_cand[valid]
        BS_valid = B_side_cand[valid]

        # -- Sort by Z descending --------------------------------------
        sort_idx = np.argsort(Z_valid)[::-1]

        dtype = np.dtype([
            ("i0", np.int32), ("i1", np.int32), ("i2", np.int32),
            ("Z", np.float64), ("s", np.float64), ("b", np.float64),
            ("b_side", np.float64),
        ])
        self.coarse_results = np.empty(n_valid, dtype=dtype)
        self.coarse_results["i0"] = i0_arr[sort_idx]
        self.coarse_results["i1"] = i1_arr[sort_idx]
        self.coarse_results["i2"] = i2_arr[sort_idx]
        self.coarse_results["Z"] = Z_valid[sort_idx]
        self.coarse_results["s"] = S_valid[sort_idx]
        self.coarse_results["b"] = B_valid[sort_idx]
        self.coarse_results["b_side"] = BS_valid[sort_idx]

        # -- Top-10 report ---------------------------------------------
        top = self.coarse_results[:10]
        print(f"[Phase 1] Top 10 combinations:")
        for rec in top:
            th0, th1, th2 = self._indices_to_thresholds(
                int(rec["i0"]), int(rec["i1"]), int(rec["i2"])
            )
            th_nonRes, th_Res, th_ggHH = self._trans_thresholds_to_original(th0, th1, th2)
            print(f"  Z={rec['Z']:.4f}  s={rec['s']:.3g}  b={rec['b']:.3g}  "
                  f"b_side={rec['b_side']:.3g}  "
                  f"th_nonRes<{th_nonRes:.4f}  th_Res<{th_Res:.4f}  th_ggHH>{th_ggHH:.4f}")

    # ----------------------------------------------------------------------
    #  PHASE 2: FEASIBLE SPACE DETERMINATION
    # ----------------------------------------------------------------------

    def _phase2_feasible_space(self):
        """Determine feasible score-space volume from valid coarse combinations.

        Computes per-dimension bounds and counts of active coarse cells.
        """
        print("[Phase 2] Determining feasible space ...")

        res = self.coarse_results

        # Per-dimension bounds (bin indices in transformed space)
        i0_min, i0_max = res["i0"].min(), res["i0"].max()
        i1_min, i1_max = res["i1"].min(), res["i1"].max()
        i2_min, i2_max = res["i2"].min(), res["i2"].max()

        self.feasible_bounds = {
            "inv0": (i0_min, i0_max),
            "inv1": (i1_min, i1_max),
            "sig": (i2_min, i2_max),
        }

        unique_i0 = len(np.unique(res["i0"]))
        unique_i1 = len(np.unique(res["i1"]))
        unique_i2 = len(np.unique(res["i2"]))

        self.feasible_n_cells = unique_i0 * unique_i1 * unique_i2

        print(f"[Phase 2] Feasible range per dimension (coarse bin indices):")
        print(f"  nonRes_inv: [{i0_min}, {i0_max}]  ({unique_i0} unique values)")
        print(f"  Res_inv:    [{i1_min}, {i1_max}]  ({unique_i1} unique values)")
        print(f"  ggHH:       [{i2_min}, {i2_max}]  ({unique_i2} unique values)")
        print(f"  Estimated active cells: ~{self.feasible_n_cells:,}")

    # ----------------------------------------------------------------------
    #  FINE HISTOGRAM SCAN  (unequal-width histograms, vectorized)
    # ----------------------------------------------------------------------

    def _get_cat_sb_max(self, cat_num):
        """Get sideband upper limit for a category."""
        idx = cat_num - 1
        if idx < len(self.cat_sideband_max):
            return float(self.cat_sideband_max[idx])
        return float("inf")

    def _fine_histogram_scan(self, ci0, ci1, ci2, cat_num):
        """Fine scan within one coarse bin using unequal-width 3D histograms.

        Per dimension: n_fine_bins equal bins dividing the coarse bin range,
        plus 1 extra bin covering [coarse_upper, overall_max]. This extra bin
        naturally captures all events from coarser bins, so no separate
        "constant base" handling is needed.

        Builds 3D histograms of the region (scores >= coarse bin lower bounds),
        computes reverse cumulative sums, then evaluates all (n_fine_bins+1)^3
        threshold combinations via fully vectorized numpy operations.

        Replaces both the old _event_driven_3d_scan (O(n_cell X n_active))
        and _sig_1d_scan (1D-only), providing true 3D fine scanning with
        O(1) per threshold evaluation.
        """
        n_fine = self.n_fine_bins
        e = self.coarse_edges
        s3d = self.scores_trans[:, [0, 1, 2]]

        # All events with scores >= coarse bin lower bounds in all 3 dims.
        # This includes: events within the coarse bin, events in coarser bins,
        # and "mixed" events (coarser in some dims, within-range in others).
        mask = (
            (s3d[:, 0] >= e[0][ci0])
            & (s3d[:, 1] >= e[1][ci1])
            & (s3d[:, 2] >= e[2][ci2])
        )
        n_region = mask.sum()
        if n_region == 0:
            return None

        # -- Build unequal-width fine bin edges per dimension ----------
        overall_max = [1.0, 1.0, 1.0]
        fine_edges = []
        for d, ci in enumerate([ci0, ci1, ci2]):
            lo = float(e[d][ci])
            hi = float(e[d][ci + 1])
            # n_fine equal bins within the coarse bin
            inner = np.linspace(lo, hi, n_fine + 1)  # n_fine+1 edges
            # 1 extra bin: [hi, overall_max]
            outer = np.array([overall_max[d]])
            fine_edges.append(np.concatenate([inner, outer]))

        n_ft = n_fine + 1  # total fine bins per dim (e.g. 101)

        # -- Slice out the region --------------------------------------
        r_s3d = s3d[mask]
        r_w = self.weights_all[mask]
        r_m = self.mass_all[mask]
        r_l = self.labels_all[mask]
        r_sp = self.samples_all[mask]

        # Pre-compute mass-region masks (same for all histograms)
        in_LSB = (r_m >= MASS_LEFT_SB[0]) & (r_m < MASS_LEFT_SB[1])
        in_SR = (r_m >= MASS_SR[0]) & (r_m < MASS_SR[1])
        in_RSB = (r_m >= MASS_RIGHT_SB[0]) & (r_m < MASS_RIGHT_SB[1])
        in_sideband = in_LSB | in_RSB

        is_signal = r_l == 1
        is_bkg = r_l == 0
        is_data = r_sp == "Data"
        is_interp = np.isin(r_sp, list(INTERP_SAMPLES))
        is_noninterp_bkg = np.isin(r_sp, self.bkg_samples) & ~is_interp

        bins_3d = [fine_edges[0], fine_edges[1], fine_edges[2]]

        # -- Build 3D histograms ---------------------------------------
        m = is_signal & in_SR
        H_sig, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        m = is_interp & in_LSB
        H_il, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        m = is_interp & in_RSB
        H_ir, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        m = is_noninterp_bkg & in_SR
        H_ni, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        m = is_bkg & ~is_data & in_sideband
        H_bs, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        # -- Split sideband for SB ratio constraint --
        m = is_data & in_LSB
        H_bs_L, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])
        m = is_data & in_RSB
        H_bs_R, _ = np.histogramdd(r_s3d[m], bins=bins_3d, weights=r_w[m])

        # -- Reverse cumulative sums (survival function) ---------------
        C_sig = self._compute_reverse_cumsum(H_sig)
        C_il = self._compute_reverse_cumsum(H_il)
        C_ir = self._compute_reverse_cumsum(H_ir)
        C_ni = self._compute_reverse_cumsum(H_ni)
        C_bs = self._compute_reverse_cumsum(H_bs)
        C_bs_L = self._compute_reverse_cumsum(H_bs_L)
        C_bs_R = self._compute_reverse_cumsum(H_bs_R)

        # -- Sideband constraints --------------------------------------
        sb_min = self.side_band_threshold_low
        sb_max = self._get_cat_sb_max(cat_num)

        # -- Vectorized Z evaluation over all (n_ft)^3 combos ----------
        S = C_sig
        B_interp = COEFF_L * C_il + COEFF_R * C_ir
        B = np.maximum(B_interp + C_ni, 1e-9)
        B_side = C_bs

        valid_mask = (S > 0) & (B_side >= sb_min) & (B_side <= sb_max)
        if self.require_sb_ratio:
            valid_mask = (
                valid_mask
                & (C_bs_L > 0)
                & (C_bs_R / C_bs_L < _WIDTH_R / _WIDTH_L)
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.where(
                valid_mask,
                np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)),
                0.0,
            )

        best_flat = np.argmax(Z.ravel())
        best_i, best_j, best_k = np.unravel_index(best_flat, Z.shape)
        best_Z = float(Z[best_i, best_j, best_k])

        if best_Z <= 0:
            return None

        return {
            "th_inv0": float(fine_edges[0][best_i]),
            "th_inv1": float(fine_edges[1][best_j]),
            "th_sig": float(fine_edges[2][best_k]),
            "ci0": ci0, "ci1": ci1, "ci2": ci2,
            "Z": best_Z,
            "s": float(S[best_i, best_j, best_k]),
            "b": float(B[best_i, best_j, best_k]),
            "b_side": float(B_side[best_i, best_j, best_k]),
            "n_cell": n_region,
            "method": "fine_hist",
            "source": "fine",
        }

    # ----------------------------------------------------------------------
    #  SEQUENTIAL CATEGORY OPTIMIZATION
    # ----------------------------------------------------------------------

    def _sequential_grid_optimization(self):
        """Sequential grid-search optimization.

        1. Build coarse histograms, compute cumsums, evaluate all coarse combos.
        2. For each category: collect valid bins (sideband in range), run
           fine histogram scan on each, select best.
        3. Remove selected events, rebuild coarse grid, repeat.
        """
        cat_path = os.path.join(self.output_dir, self.cat_folder)
        os.makedirs(cat_path, exist_ok=True)

        self._build_coarse_histograms()
        self._compute_coarse_cumsums()
        self._phase1_coarse_evaluation()

        best_cut_params_list = []
        best_sig_values = []
        sig_peak_list = []
        gghh_sig_peak_list = []
        bkg_side_list = []
        bkg_side_unscaled_list = []
        data_side_event_list = []
        bkg_side_event_list = []
        data_side_weight_list = []

        for cat in range(1, self.n_categories + 1):
            print(f"\n{'='*60}")
            print(f"Category {cat} of {self.n_categories}")
            print(f"{'='*60}")

            sb_min = self.side_band_threshold_low
            sb_max = self._get_cat_sb_max(cat)
            print(f"  Sideband: [{sb_min}, {sb_max}]")

            # -- Separate valid bins into those with/without events ---
            # Vectorized: extract columns, compute tighter_bs, filter
            H_cnt = self.coarse_histograms["count"]
            n0, n1, n2 = self.coarse_cumsums["bkg_sideband"].shape
            C_bs = self.coarse_cumsums["bkg_sideband"]

            res = self.coarse_results
            i0 = res["i0"]
            i1 = res["i1"]
            i2 = res["i2"]
            bs_arr = res["b_side"]

            # Clamped i+1 for tighter-sideband lookup
            ni0 = np.where(i0 + 1 < n0, i0 + 1, i0)
            ni1 = np.where(i1 + 1 < n1, i1 + 1, i1)
            ni2 = np.where(i2 + 1 < n2, i2 + 1, i2)
            tighter_bs = C_bs[ni0, ni1, ni2]

            valid_mask = (bs_arr >= sb_min) & (tighter_bs <= sb_max)
            has_events = H_cnt[i0, i1, i2] > 0

            valid_with_events = res[valid_mask & has_events]
            valid_empty = res[valid_mask & ~has_events]

            n_with = len(valid_with_events)
            n_empty = len(valid_empty)
            print(f"  Valid bins: {n_with} with events, {n_empty} empty")

            if n_with + n_empty == 0:
                print(f"  No valid bins. Stopping.")
                break

            # -- Fine histogram scan on populated bins -----------------
            best_result = None

            if n_with > 0:
                if n_with > 100:
                    n_with = 100
                    valid_with_events = valid_with_events[:n_with]
                    print(f"  Limiting to top {n_with} populated bins for fine scan ...")
                print(f"  Fine-scanning {n_with} populated bins "
                      f"(n_fine_bins={self.n_fine_bins}, "
                      f"{(self.n_fine_bins+1)**3:,} combos/bin) ...")
                n_scanned = 0
                for rec in valid_with_events:
                    i0, i1, i2 = int(rec["i0"]), int(rec["i1"]), int(rec["i2"])

                    fine = self._fine_histogram_scan(i0, i1, i2, cat)

                    n_scanned += 1
                    if fine is not None:
                        th_nonRes, th_Res, th_ggHH = \
                            self._trans_thresholds_to_original(
                                fine["th_inv0"], fine["th_inv1"], fine["th_sig"])
                        fine["th_nonRes"] = th_nonRes
                        fine["th_Res"] = th_Res
                        fine["th_ggHH"] = th_ggHH
                        fine["coarse_Z"] = float(rec["Z"])

                        if best_result is None or fine["Z"] > best_result["Z"]:
                            best_result = fine

                    if n_scanned % 100 == 0:
                        print(f"    ... {n_scanned}/{n_with} scanned, "
                              f"best Z={best_result['Z'] if best_result else 0:.4f}")

            # Fall back to best empty bin if no fine result
            if best_result is None and n_empty > 0:
                best_coarse = valid_empty[0]  # sorted by Z descending
                th0, th1, th2 = self._indices_to_thresholds(
                    int(best_coarse["i0"]), int(best_coarse["i1"]),
                    int(best_coarse["i2"]))
                th_nonRes, th_Res, th_ggHH = \
                    self._trans_thresholds_to_original(th0, th1, th2)
                best_result = {
                    "th_inv0": th0, "th_inv1": th1, "th_sig": th2,
                    "th_nonRes": th_nonRes, "th_Res": th_Res,
                    "th_ggHH": th_ggHH,
                    "Z": float(best_coarse["Z"]),
                    "s": float(best_coarse["s"]),
                    "b": float(best_coarse["b"]),
                    "b_side": float(best_coarse["b_side"]),
                    "source": "coarse-empty",
                }
            elif best_result is None and n_with > 0:
                # Fine scanned all populated bins but none valid -
                # take best coarse among them
                best_coarse = valid_with_events[0]
                th0, th1, th2 = self._indices_to_thresholds(
                    int(best_coarse["i0"]), int(best_coarse["i1"]),
                    int(best_coarse["i2"]))
                th_nonRes, th_Res, th_ggHH = \
                    self._trans_thresholds_to_original(th0, th1, th2)
                best_result = {
                    "th_inv0": th0, "th_inv1": th1, "th_sig": th2,
                    "th_nonRes": th_nonRes, "th_Res": th_Res,
                    "th_ggHH": th_ggHH,
                    "Z": float(best_coarse["Z"]),
                    "s": float(best_coarse["s"]),
                    "b": float(best_coarse["b"]),
                    "b_side": float(best_coarse["b_side"]),
                    "source": "coarse-fallback",
                }

            # -- Report and save ------------------------------------
            meth = best_result.get("method", best_result.get("source", "?"))
            print(f"  Best ({meth}): Z={best_result['Z']:.4f}  "
                  f"s={best_result['s']:.3g}  b={best_result['b']:.3g}  "
                  f"b_side={best_result['b_side']:.3g}")
            print(f"  th_nonRes < {best_result['th_nonRes']:.4f}  "
                  f"th_Res < {best_result['th_Res']:.4f}  "
                  f"th_ggHH > {best_result['th_ggHH']:.4f}")

            params = {
                "th_signal": float(best_result["th_ggHH"]),
                f"th_bg_{self.bkg_classes[0]}": float(best_result["th_nonRes"]),
                f"th_bg_{self.bkg_classes[1]}": float(best_result["th_Res"]),
            }
            best_cut_params_list.append(params)
            best_sig_values.append(best_result["Z"])

            # Raw-event bookkeeping
            s3d = self.scores_trans[:, [0, 1, 2]]
            mask_local = (
                (s3d[:, 0] > best_result["th_inv0"])
                & (s3d[:, 1] > best_result["th_inv1"])
                & (s3d[:, 2] > best_result["th_sig"])
            )

            if mask_local.sum() > 0:
                ml = self.labels_all[mask_local]
                mm = self.mass_all[mask_local]
                mw = self.weights_all[mask_local]
                mw_unscaled = self.weights_unscaled_all[mask_local]
                ms = self.samples_all[mask_local]
                in_peak = (mm > MASS_SR[0]) & (mm < MASS_SR[1])
                # Match the sideband convention used to build the optimization
                # histograms and by make_yield_table_5cats_autodetect.py.
                in_left_sideband = (
                    (mm >= MASS_LEFT_SB[0]) & (mm < MASS_LEFT_SB[1])
                )
                in_right_sideband = (
                    (mm >= MASS_RIGHT_SB[0]) & (mm < MASS_RIGHT_SB[1])
                )
                sideband_mask = in_left_sideband | in_right_sideband
                sig_peak = mw[in_peak & (ml == 1)].sum()
                is_gghh_sample = np.array([
                    ("GluGluToHH" in s) or ("GluGlutoHH" in s)
                    for s in ms
                ])
                gghh_sig_peak = mw[in_peak & is_gghh_sample].sum()
                bkg_side_mask = sideband_mask & (ml == 0) & (ms != "Data")
                data_side_mask = sideband_mask & (ml == 0) & (ms == "Data")
                bkg_side = mw[bkg_side_mask].sum()
                bkg_side_unscaled = mw_unscaled[bkg_side_mask].sum()
                data_side = mw[data_side_mask].sum()
                bkg_side_events = bkg_side_mask.sum()
                data_side_events = data_side_mask.sum()
            else:
                sig_peak = 0.0
                gghh_sig_peak = 0.0
                bkg_side = 0.0
                bkg_side_unscaled = 0.0
                data_side = 0.0
                bkg_side_events = 0
                data_side_events = 0

            sig_peak_list.append(sig_peak)
            gghh_sig_peak_list.append(gghh_sig_peak)
            bkg_side_list.append(bkg_side)
            bkg_side_unscaled_list.append(bkg_side_unscaled)
            data_side_event_list.append(data_side_events)
            bkg_side_event_list.append(bkg_side_events)
            data_side_weight_list.append(data_side)
            print(f"  Signal in SR: {sig_peak:.3g}  Bkg in SB: {bkg_side:.3g}  Data in SB: {data_side:.3g} "
                  f"Removed: {mask_local.sum():,}")
            print(f"  Optimized-threshold sideband events: Data={data_side_events:,}  "
                  f"Background={bkg_side_events:,}  Background sumW unscaled={bkg_side_unscaled:.6g}  "
                  f"Background sumW scaled={bkg_side:.6g}")
            
            # if this is the last category, we don't need to remove events and rebuild
            if cat == self.n_categories:
                break

            # -- Incremental update: subtract removed events from histograms -
            # Build histograms of removed events BEFORE trimming raw arrays
            self._subtract_removed_events(mask_local)

            # Trim raw arrays (for fine scanning in subsequent categories)
            keep = ~mask_local
            self.scores_all = self.scores_all[keep]
            self.scores_trans = self.scores_trans[keep]
            self.mass_all = self.mass_all[keep]
            self.dijet_mass_all = self.dijet_mass_all[keep]
            self.weights_all = self.weights_all[keep]
            self.weights_unscaled_all = self.weights_unscaled_all[keep]
            self.labels_all = self.labels_all[keep]
            self.samples_all = self.samples_all[keep]

            if len(self.scores_all) == 0:
                print("  No events remaining. Stopping.")
                break

            self._compute_coarse_cumsums()
            self._phase1_coarse_evaluation()

        # -- Summary --------------------------------------------------
        if not best_sig_values:
            raise RuntimeError("No successful categories found.")

        z_sum_quad = np.sqrt(np.sum(np.array(best_sig_values) ** 2))
        print(f"\n{'='*60}")
        print(f"Grid Search Categorization Summary")
        print(f"{'='*60}")
        print(f"Per-category Z: {best_sig_values}")
        print(f"Quadrature sum:  {z_sum_quad:.4f}")
        print(f"Signal in SR:    {sig_peak_list}")
        print(f"Bkg in SB:       {bkg_side_list}")
        print("Optimized-threshold sideband summary per category:")
        print("  Category  ggHH SR yield  Data events  Bkg events  Data weight  Bkg sumW unscaled  Bkg sumW scaled")
        for i, (gghh_sig, data_evt, bkg_evt, data_w, bkg_w_unscaled, bkg_w) in enumerate(
            zip(gghh_sig_peak_list, data_side_event_list, bkg_side_event_list,
                data_side_weight_list, bkg_side_unscaled_list, bkg_side_list),
            start=1,
        ):
            print(f"  cat{i:<5} {gghh_sig:>13.6g} {data_evt:>11,} {bkg_evt:>11,} "
                  f"{data_w:>12.6g} {bkg_w_unscaled:>17.6g} {bkg_w:>16.6g}")
        print("  Total    "
              f"{sum(gghh_sig_peak_list):>13.6g} "
              f"{sum(data_side_event_list):>11,} {sum(bkg_side_event_list):>11,} "
              f"{sum(data_side_weight_list):>12.6g} {sum(bkg_side_unscaled_list):>17.6g} "
              f"{sum(bkg_side_list):>16.6g}")

        self._write_output_files(
            best_cut_params_list, best_sig_values,
            sig_peak_list, bkg_side_list, cat_path)

        return best_cut_params_list, best_sig_values
    
    def _remove_fake_digits(self, value, place=12):
        # assume precision is less than 12 digits, so we can round to 12 digits to remove fake digits
        return round(value, place) if isinstance(value, float) else value

    def _write_output_files(self, best_cut_params_list, best_sig_values,
                            sig_peak_list, bkg_side_list, cat_path):
        """Write txt, json, metric outputs and summary plot."""
        z_sum_quad = np.sqrt(np.sum(np.array(best_sig_values) ** 2))

        for p in best_cut_params_list:
            for k in p.keys():
                p[k] = self._remove_fake_digits(p[k])

        sig_col = SCORE_COLS[self.signal_class]
        base_cuts = []
        for p in best_cut_params_list:
            parts = [f"{sig_col} > {p['th_signal']}"]
            for idx, bg_idx in enumerate(self.bkg_classes):
                bg_col = SCORE_COLS[bg_idx]
                parts.append(f"{bg_col} < {p[f'th_bg_{bg_idx}']}")
            base_cuts.append("(" + " & ".join(parts) + ")")

        cat_strings = {}
        for i, base in enumerate(base_cuts, start=1):
            parts = [base]
            if i >= 2:
                for j in range(i - 1):
                    parts.append(f"not({base_cuts[j]})")
            cat_strings[f"cat{i}"] = " & ".join(parts)
        
        for i, base in enumerate(base_cuts, start=1):
            if self.apply_boosted_veto:
                cat_strings[f"cat{i}"] += " & (is_boosted == 0)"
            if self.vbf_thresholds is not None:
                vbf_col = SCORE_COLS[self.vbfhh_class]
                cat_strings[f"cat{i}"] += f" & ({vbf_col} < {self.vbf_thresholds})"
            cat_strings[f"cat{i}"] += " & (dijet_mass > 80) & (dijet_mass < 190)"

        txt_path = os.path.join(cat_path, "best_cut_params.txt")
        with open(txt_path, "w") as f:
            f.write(json.dumps(cat_strings, indent=2))
        print(f"Category strings saved to {txt_path}")

        json_path = os.path.join(cat_path, "best_cut_params.json")
        with open(json_path, "w") as f:
            json.dump(best_cut_params_list, f, indent=4)
        print(f"Best parameters saved to {json_path}")

        metric_path = os.path.join(cat_path, "best_cut_metric.json")
        with open(metric_path, "w") as f:
            json.dump({
                "Z_score_values": best_sig_values,
                "Z_sum_quad": z_sum_quad,
                "sig_peak_list": sig_peak_list,
                "bkg_side_list": bkg_side_list,
                "best_cut_params_list": best_cut_params_list
            }, f, indent=4)
        print(f"Metric values saved to {metric_path}")

        self._plot_category_summary(
            best_sig_values, sig_peak_list, bkg_side_list,
            best_cut_params_list, cat_path,
        )

    # ----------------------------------------------------------------------
    #  PLOTTING
    # ----------------------------------------------------------------------

    def _plot_category_summary(
        self, best_sig_values, sig_peak_list, bkg_side_list,
        best_cut_params_list, save_path,
    ):
        """4-panel summary plot (same style as original)."""
        plt.style.use(hep.style.CMS)
        n_cats = len(best_sig_values)
        cat_indices = np.arange(n_cats)

        z_sum_quad = [
            np.sqrt(np.sum(np.array(best_sig_values[:i]) ** 2))
            for i in range(1, n_cats + 1)
        ]

        fig, axs = plt.subplots(
            4, 1, figsize=(10, 12),
            gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0},
        )

        # Build threshold name mapping
        th_name_to_class = {}
        for n, bi in enumerate(self.bkg_classes):
            th_name_to_class[f"th_bg_{bi}"] = f"th_{self.bkg_class_names[n]}"
        th_name_to_class["th_signal"] = f"th_{self.signal_class_name}"

        # 1) Asymptotic significance
        axs[0].plot(cat_indices, best_sig_values, ".b", markersize=8)
        axs[0].set_ylabel("Asymptotic Significance (Z)", fontsize=10)
        axs[0].grid(True)
        axs[0].tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        for i in range(n_cats):
            threshold_text = "\n".join(
                f"{th_name_to_class[k]}={v:.4f}"
                for k, v in best_cut_params_list[i].items()
            )
            axs[0].annotate(
                threshold_text,
                xy=(i, best_sig_values[i]),
                xytext=(i, best_sig_values[i] * 1.10),
                ha="center",
                arrowprops=dict(color="black", arrowstyle="->", lw=1),
                fontsize=10,
            )

        # 2) Z sum in quadrature
        axs[1].plot(cat_indices, z_sum_quad, ".b", markersize=8)
        axs[1].set_ylabel("Z sum in quadrature", fontsize=10)
        axs[1].grid(True)
        axs[1].tick_params(axis="x", bottom=False, top=False, labelbottom=False)

        # 3) Signal under the peak
        axs[2].plot(cat_indices, sig_peak_list, ".b", markersize=8)
        axs[2].set_ylabel("Signal under peak", fontsize=10)
        axs[2].grid(True)
        axs[2].tick_params(axis="x", bottom=False, top=False, labelbottom=False)

        # 4) Background in sidebands (log scale)
        axs[3].plot(cat_indices, bkg_side_list, ".b", markersize=8)
        axs[3].set_yscale("log")
        axs[3].set_ylabel("Bkg in sidebands", fontsize=10)
        axs[3].set_xlabel("Category Index", fontsize=10)
        axs[3].grid(True)
        for i in range(n_cats):
            axs[3].annotate(
                f"{bkg_side_list[i]:.5f}",
                xy=(i, bkg_side_list[i]),
                xytext=(i, bkg_side_list[i] * 1.5),
                fontsize=8,
                arrowprops=dict(color="black", arrowstyle="->"),
                ha="center",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "category_summary_new.png"))
        plt.close(fig)

    # ----------------------------------------------------------------------
    #  VBFHH OPTIMIZATION  (1D scan - identical to original)
    # ----------------------------------------------------------------------

    def optimize_vbfhh_sr(self, vbfhh_class, vbfhh_samples,
                           n_scan_points=1000, sideband_threshold=10.0):
        """Scan VBFHH score and return best threshold."""
        if self.scores_all is None:
            raise RuntimeError("scores_all is not set. Call load_samples() first.")

        thresholds = np.linspace(0.0, 1.0, n_scan_points + 1)[:-1]

        best_threshold = None
        best_z = -1.0
        best_s = 0.0
        best_b = 0.0

        sr_low, sr_high = MASS_SR

        for threshold in thresholds:
            mask = self.scores_all[:, vbfhh_class] > threshold
            if not np.any(mask):
                continue

            sel_masses = self.mass_all[mask]
            sel_weights = self.weights_all[mask]
            sel_samples = self.samples_all[mask]

            is_vbfhh = np.isin(sel_samples, vbfhh_samples)

            side_mask = ((sel_masses <= sr_low) | (sel_masses >= sr_high)) & ~is_vbfhh
            if sel_weights[side_mask].sum() < sideband_threshold:
                continue

            sr_mask = (sel_masses > sr_low) & (sel_masses < sr_high)
            s = sel_weights[sr_mask & is_vbfhh].sum()
            if s <= 0.0:
                continue

            b_total = 0.0
            interp_mask = ~is_vbfhh & np.isin(sel_samples, list(INTERP_SAMPLES))
            if np.any(interp_mask):
                b_interp, _ = self._sr_from_sidebands_linear(
                    sel_masses[interp_mask], sel_weights[interp_mask]
                )
                b_total += b_interp

            for sname in self.bkg_samples:
                if sname in INTERP_SAMPLES:
                    continue
                smask = ~is_vbfhh & (sel_samples == sname)
                if not np.any(smask):
                    continue
                b_total += sel_weights[smask & sr_mask].sum()

            if b_total <= 0.0:
                b_total = 1e-9

            z = self.asymptotic_significance(s, b_total)
            if z > best_z:
                best_z = z
                best_threshold = threshold
                best_s = s
                best_b = b_total

        return best_threshold, best_z, best_s, best_b

    # ----------------------------------------------------------------------
    #  MAIN ENTRY POINT
    # ----------------------------------------------------------------------

    def run_categorisation(self):
        """Full categorization pipeline: VBFHH (optional) + grid-search SR."""
        if self.use_individual_samples:
            raise NotImplementedError(
                "Individual sample loading not supported in grid search version. "
                "Use --input_file instead."
            )
        else:
            _ = self.load_samples_merged()

        # -- VBFHH SR optimization (optional) -------------------------
        if self.vbfhh_class is not None and self.vbfhh_samples is not None:
            sr_low, sr_high = MASS_SR
            all_vbfhh_sr_info = []

            if self.vbf_thresholds is not None:
                self.vbfhh_n_categories = 1

            for i_cat in range(self.vbfhh_n_categories):
                print(f"\n--- Optimising VBFHH SR {i_cat + 1} of "
                      f"{self.vbfhh_n_categories} ---")
                if self.vbf_thresholds is not None:
                    best_vbfhh_threshold = self.vbf_thresholds
                    best_vbfhh_z = 0.0001
                    best_vbfhh_s = 0.0001
                    best_vbfhh_b = 0.0001
                else:
                    (best_vbfhh_threshold, best_vbfhh_z, best_vbfhh_s,
                     best_vbfhh_b) = self.optimize_vbfhh_sr(
                        vbfhh_class=self.vbfhh_class,
                        vbfhh_samples=self.vbfhh_samples,
                        n_scan_points=self.vbfhh_n_scan_points,
                        sideband_threshold=self.vbfhh_sideband_threshold,
                    )

                if best_vbfhh_threshold is None:
                    print(f"No valid VBFHH SR found at iteration {i_cat + 1}, stopping.")
                    break

                self.vbf_thresholds = best_vbfhh_threshold

                print(f"VBFHH SR {i_cat + 1}: threshold = {best_vbfhh_threshold:.4f}, "
                      f"Z = {best_vbfhh_z:.4f}, s = {best_vbfhh_s:.4g}, "
                      f"b = {best_vbfhh_b:.4g}")

                vbfhh_mask = self.scores_all[:, self.vbfhh_class] > best_vbfhh_threshold
                sr_mass_mask = (self.mass_all > sr_low) & (self.mass_all < sr_high)
                is_vbfhh = np.isin(self.samples_all, self.vbfhh_samples)
                is_ggHH = np.array(
                    ["GluGlutoHHto2B2G" in s for s in self.samples_all]
                )
                is_interp = np.isin(self.samples_all, list(INTERP_SAMPLES))
                is_singleH = ~is_vbfhh & ~is_ggHH & ~is_interp

                best_ggHH = float(
                    self.weights_all[vbfhh_mask & sr_mass_mask & is_ggHH].sum()
                )
                best_H = float(
                    self.weights_all[vbfhh_mask & sr_mass_mask & is_singleH].sum()
                )
                sdb_mask = (
                    (self.mass_all <= sr_low) | (self.mass_all >= sr_high)
                ) & ~is_vbfhh
                best_SDB = float(self.weights_all[vbfhh_mask & sdb_mask].sum())

                print(f"VBFHH SR {i_cat + 1}: ggHH = {best_ggHH:.4g}, "
                      f"single-H = {best_H:.4g}, SDB = {best_SDB:.4g}")

                all_vbfhh_sr_info.append({
                    "best_threshold": float(best_vbfhh_threshold),
                    "best_z": float(best_vbfhh_z),
                    "best_s": float(best_vbfhh_s),
                    "best_b": float(best_vbfhh_b),
                    "best_ggHH": best_ggHH,
                    "best_H": best_H,
                    "best_SDB": best_SDB,
                })

                remaining = ~vbfhh_mask
                print(f"VBFHH SR {i_cat + 1} removes {vbfhh_mask.sum():,} events; "
                      f"{remaining.sum():,} remain for SR optimisation.")
                self.scores_all = self.scores_all[remaining]
                self.scores_trans = self.scores_trans[remaining]
                self.mass_all = self.mass_all[remaining]
                self.dijet_mass_all = self.dijet_mass_all[remaining]
                self.weights_all = self.weights_all[remaining]
                self.weights_unscaled_all = self.weights_unscaled_all[remaining]
                self.labels_all = self.labels_all[remaining]
                self.samples_all = self.samples_all[remaining]

                if not np.any(remaining):
                    print("No events remaining, stopping.")
                    break

            cat_path = os.path.join(self.output_dir, self.cat_folder)
            os.makedirs(cat_path, exist_ok=True)
            vbfhh_sr_json = os.path.join(cat_path, "vbfhh_sr_info.json")
            with open(vbfhh_sr_json, "w") as f:
                json.dump(all_vbfhh_sr_info, f, indent=4)
            print(f"VBFHH SR info saved to {vbfhh_sr_json}")

        # -- Grid search SR categorization ----------------------------
        if self.n_categories > 0:

            self._sequential_grid_optimization()

# --------------------------------------------------------------------------
#                                MAIN                                      
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search-based categorization for HHbbgg analysis."
    )

    # -- Required ----------------------------------------------------
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path for input samples")
    parser.add_argument("--signal_samples", type=str, required=True,
                        help="Bracket-enclosed signal sample list, "
                             "e.g. '[GluGluToHH_kl-1p00_kt-1p00_c2-0p00]'")

    # -- Categorization ----------------------------------------------
    parser.add_argument("--n_categories", type=int, default=5,
                        help="Number of categories to optimize")
    parser.add_argument("--side_band_threshold_low", type=float, default=10.0,
                        help="Minimum sideband background (weighted events)")
    parser.add_argument("--side_band_threshold_high", type=float, default=15,
                        help="Maximum sideband background for category 1")
    parser.add_argument("--cat_sideband_max", type=str, default="25,100,500",
                        help="Comma-separated per-category sideband upper "
                             "limits (cat1, cat2, cat3, ...)")

    # -- Grid search -------------------------------------------------
    parser.add_argument("--n_coarse_bins", type=int, default=100,
                        help="Bins per dimension for Phase 1 coarse grid")
    parser.add_argument("--n_fine_bins", type=int, default=100,
                        help="Fine bins per dimension within each coarse bin "
                             "(produces (n_fine_bins+1)^3 approx 1M threshold "
                             "combinations per coarse bin)")
    parser.add_argument("--require_sb_ratio", action="store_true", default=False,
                        help="Require right_sideband / left_sideband "
                             "< right_width / left_width (=%.2f)"
                             % (_WIDTH_R / _WIDTH_L))

    # -- Output ------------------------------------------------------
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory for output files")
    parser.add_argument("--optuna_folder", type=str,
                        default="grid_search_categorization",
                        help="Output sub-folder name")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to merged scored parquet file")

    # -- Background classes ------------------------------------------
    parser.add_argument("--signal_class", type=int, default=2,
                        help="Signal class index in score array")
    parser.add_argument("--signal_class_name", type=str, default="ggHH",
                        help="Signal class name for output")
    parser.add_argument("--bkg_classes", type=str, default="0,1",
                        help="Background class indices (comma-separated)")
    parser.add_argument("--bkg_class_names", type=str, default="nonRes,Res",
                        help="Background class names (comma-separated)")
    parser.add_argument("--bkg_samples", type=str,
                        default="[VBFHToGG,VHToGG,ttHToGG,GluGluHToGG,"
                                "TTGG,GGJets,DDQCDGJets]",
                        help="Bracket-enclosed background sample list")

    # -- VBFHH -------------------------------------------------------
    parser.add_argument("--vbfhh_class", type=int, default=3,
                        help="VBFHH score index")
    parser.add_argument("--vbfhh_samples", type=str,
                        default="[VBFHH_CV-1p000_C2V-1p000_C3-1p000]",
                        help="Bracket-enclosed VBFHH signal sample list")
    parser.add_argument("--vbfhh_n_scan_points", type=int, default=1000,
                        help="Number of scan points for VBFHH 1D scan")
    parser.add_argument("--vbfhh_sideband_threshold", type=float, default=10.0,
                        help="Minimum sideband background for VBFHH SR")
    parser.add_argument("--vbfhh_n_categories", type=int, default=1,
                        help="Number of VBFHH SR categories")
    parser.add_argument("--vbf_thresholds", type=float, default=None,
                        help="Fixed VBFHH threshold (skip scan if set)")

    # -- Pre-selection -----------------------------------------------
    parser.add_argument("--apply_boosted_veto", action="store_true", default=False,
                        help="Exclude boosted events")
    parser.add_argument("--use_individual_samples", action="store_true",
                        default=False,
                        help="Use per-era/per-sample loading (not recommended)")
    parser.add_argument("--weight_scale", type=float, default=1.44,
                        help="Scale factor for weight adjustment")

    args = parser.parse_args()

    # Parse list-type arguments
    bkg_classes = [int(x.strip()) for x in args.bkg_classes.split(",")
                   if x.strip()]

    bkg_class_names = [x.strip() for x in args.bkg_class_names.split(",")
                       if x.strip()]
    if len(bkg_class_names) != len(bkg_classes):
        raise ValueError(
            f"bkg_class_names ({len(bkg_class_names)}) must match "
            f"bkg_classes ({len(bkg_classes)})"
        )

    # Create categorizer
    categoriser = GridSearchCategorizer(
        base_path=args.base_path,
        cat_folder=args.optuna_folder,
        n_categories=args.n_categories,
        side_band_threshold_low=args.side_band_threshold_low,
        side_band_threshold_high=args.side_band_threshold_high,
        cat_sideband_max=args.cat_sideband_max,
        signal_samples=args.signal_samples,
        signal_class=args.signal_class,
        signal_class_name=args.signal_class_name,
        bkg_classes=bkg_classes,
        bkg_class_names=bkg_class_names,
        bkg_samples=args.bkg_samples,
        vbfhh_class=args.vbfhh_class,
        vbfhh_samples=args.vbfhh_samples,
        vbfhh_n_scan_points=args.vbfhh_n_scan_points,
        vbfhh_sideband_threshold=args.vbfhh_sideband_threshold,
        vbfhh_n_categories=args.vbfhh_n_categories,
        output_dir=args.output_dir,
        input_file=args.input_file,
        use_individual_samples=args.use_individual_samples,
        vbf_thresholds=args.vbf_thresholds,
        n_coarse_bins=args.n_coarse_bins,
        n_fine_bins=args.n_fine_bins,
        require_sb_ratio=args.require_sb_ratio,
    )
    categoriser.apply_boosted_veto = args.apply_boosted_veto
    categoriser.weight_scale = args.weight_scale
    categoriser.run_categorisation()
