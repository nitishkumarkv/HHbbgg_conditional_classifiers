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
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

# -- Pre-computed interpolation coefficients ------------------------------
# Mass windows (GeV):
#   left  sideband: [100.0, 120.0]  width_L  = 20.0, centroid_L  = 110.0
#   signal region:   [122.5, 127.0]  width_SR = 4.5,  centroid_SR = 124.75
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
_CENTROID_SR = 0.5 * (MASS_SR[0] + MASS_SR[1])  # 124.75
_CENTROID_R = 0.5 * (MASS_RIGHT_SB[0] + MASS_RIGHT_SB[1])  # 155.0
_SPAN = _CENTROID_R - _CENTROID_L  # 45.0

_W_L = (_CENTROID_R - _CENTROID_SR) / _SPAN  # ~0.6722
_W_R = (_CENTROID_SR - _CENTROID_L) / _SPAN  # ~0.3278

COEFF_L = (_W_L * _WIDTH_SR) / _WIDTH_L  # ~0.15125
COEFF_R = (_W_R * _WIDTH_SR) / _WIDTH_R  # ~0.02950

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
        self.vbf_2d_scan = False
        self.vbfhh_nonres_threshold = None
        self.skip_grid_search = False

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
        rev = np.ascontiguousarray(hist[::-1, ::-1, ::-1])
        for axis in range(hist.ndim):
            np.cumsum(rev, axis=axis, out=rev)
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
                self.base_path, "scored_samples", "merged", "merged_scored_events.parquet"
            )
        else:
            parquet_file = self.input_file

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

            # TTG / TTGG-like samples filter
            if "lead_mvaID" in pq_cols and "sublead_mvaID" in pq_cols:
                sel = (
                    data["sample"].str.startswith("TT")
                    & (data["lead_genPartFlav"] == 1)
                    & (data["sublead_genPartFlav"] == 1)
                )
                data = data[~sel]
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

    @staticmethod
    def _build_3d_hists(scores, weights, bins, configs):
        """Build multiple 3D histograms from pre-computed bins and configs.

        scores: (N, 3) array
        weights: (N,) array
        bins: list of 3 edge arrays
        configs: list of (key, mask, use_weights)

        Returns: dict of {key: 3D histogram}
        """
        n0, n1, n2 = [len(b) - 1 for b in bins]
        shape = (n0, n1, n2)
        total_cells = n0 * n1 * n2

        idx0 = np.searchsorted(bins[0], scores[:, 0], side='right') - 1
        idx1 = np.searchsorted(bins[1], scores[:, 1], side='right') - 1
        idx2 = np.searchsorted(bins[2], scores[:, 2], side='right') - 1
        in_range = (idx0 >= 0) & (idx0 < n0) & (idx1 >= 0) & (idx1 < n1) & (idx2 >= 0) & (idx2 < n2)
        idx0 = np.clip(idx0, 0, n0 - 1)
        idx1 = np.clip(idx1, 0, n1 - 1)
        idx2 = np.clip(idx2, 0, n2 - 1)
        linear_idx = idx0 * n1 * n2 + idx1 * n2 + idx2

        weights_f32 = weights.astype(np.float32) if weights is not None else None

        result = {}
        for key, mask, use_weights in configs:
            m = mask & in_range
            if m.sum() == 0:
                result[key] = np.zeros(shape, dtype=np.float32)
            else:
                h = np.bincount(linear_idx[m],
                                weights=weights_f32[m] if use_weights else None,
                                minlength=total_cells)
                if not use_weights:
                    h = h.astype(np.float32)
                result[key] = h.reshape(shape)
        return result

    def _build_histograms(self, scores, weights, masses, labels, samples, bin_edges=None):
        n_bins = self.n_coarse_bins
        bin_range = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        if bin_edges is not None:
            bins = bin_edges
        else:
            bins = [np.linspace(r[0], r[1], n_bins + 1) for r in bin_range]

        # Mass-region masks
        in_LSB = (masses >= MASS_LEFT_SB[0]) & (masses < MASS_LEFT_SB[1])
        in_SR = (masses >= MASS_SR[0]) & (masses < MASS_SR[1])
        in_RSB = (masses >= MASS_RIGHT_SB[0]) & (masses < MASS_RIGHT_SB[1])
        in_sideband = in_LSB | in_RSB

        # Event type masks
        is_signal = labels == 1
        is_bkg = labels == 0
        is_data = samples == "Data"
        is_interp = np.isin(samples, list(INTERP_SAMPLES))
        is_noninterp_bkg = np.isin(samples, self.bkg_samples) & ~is_interp

        hist_configs = [
            ("sig_SR", is_signal & in_SR, True),
            ("interp_LSB", is_interp & in_LSB, True),
            ("interp_RSB", is_interp & in_RSB, True),
            ("noninterp_SR", is_noninterp_bkg & in_SR, True),
            ("bkg_sideband", is_bkg & ~is_data & in_sideband, True),
            ("data_LSB", is_data & in_LSB, True),
            ("data_RSB", is_data & in_RSB, True),
            ("count", ~is_data, False),
        ]

        hist_dict = self._build_3d_hists(scores, weights, bins, hist_configs)

        l=""
        for k,m,_ in hist_configs:
            i=f"{k}: {m.sum():,} events"
            if l and len(l)+1+len(i)>120:print(l);l="  "+i
            else:l+=(" "if l else"  ")+i
        if l:print(l)

        return hist_dict, bins

    def _histogram_helper(self, mask = None, bin_edges = None):
        """Helper to build histograms for a subset of events."""
        if mask is None:
            mask = np.ones(len(self.scores_trans), dtype=bool)
        hist_dict, edges = self._build_histograms(
            scores=self.scores_trans[mask][:, [0, 1, 2]],
            weights=self.weights_all[mask],
            masses=self.mass_all[mask],
            labels=self.labels_all[mask],
            samples=self.samples_all[mask],
            bin_edges=bin_edges,
        )
        return hist_dict, edges
    
    def _histogram_modifier(self, A, B, action="subtract"):
        """Modify histogram A by adding/subtracting histogram B in-place."""
        if action == "subtract":
            for key in B:
                A[key] -= B[key]
        elif action == "add":
            for key in B:
                A[key] += B[key]
        else:
            raise ValueError(f"Unknown action: {action}")
        return A

    def _build_coarse_histograms(self):
        """Build 3D weighted histograms from scratch."""
        n_bins = self.n_coarse_bins
        print(f"[Phase 1] Building 3D histograms ({n_bins}^3 = {n_bins**3:,} cells) ...")
        print(f"  Axes: nonRes_inv in [0.0, 1.0], Res_inv in [0.0, 1.0], ggHH in [0.0, 1.0]")
        t0 = time.time()
        
        # Build histograms from all events
        hist_dict, edges = self._histogram_helper(mask=None, bin_edges=None)
        
        self.coarse_histograms = hist_dict
        self.coarse_edges = edges
        
        dt = time.time() - t0
        print(f"[Phase 1] Histograms built in {dt:.1f}s")

    def _subtract_removed_events(self, mask):
        """Incrementally subtract removed events from existing histograms.
        
        Builds histograms only for the masked subset, then subtracts them
        from self.coarse_histograms in-place.
        """
        # Build histograms for removed events using existing bin edges
        removed_hists, _ = self._histogram_helper(mask=mask, bin_edges=self.coarse_edges)
        
        # Subtract in-place
        H = self.coarse_histograms
        H = self._histogram_modifier(H, removed_hists, action="subtract")
        self.coarse_histograms = H

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
        cL = np.float32(COEFF_L)
        cR = np.float32(COEFF_R)
        H_bkg_SR = (cL * H["interp_LSB"]
                    + cR * H["interp_RSB"]
                    + H["noninterp_SR"])

        # Parallel cumsums for the 6 independent layers
        items = [
            ("sig_SR", H["sig_SR"]),
            ("bkg_SR", H_bkg_SR),
            ("bkg_sideband", H["bkg_sideband"]),
            ("data_LSB", H["data_LSB"]),
            ("data_RSB", H["data_RSB"]),
            ("count", H["count"]),
        ]
        n_workers = min(4, len(items))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_to_key = {ex.submit(self._compute_reverse_cumsum, data): key
                          for key, data in items}
            for f in as_completed(fut_to_key):
                C[fut_to_key[f]] = f.result()

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
            _sb_ratio = np.divide(C["data_RSB"], C["data_LSB"],
                                  out=np.full_like(C["data_RSB"], np.inf),
                                  where=C["data_LSB"] > 0)
            candidate = (
                candidate
                & (C["data_LSB"] > 0)
                & (_sb_ratio < _WIDTH_R / _WIDTH_L)
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
        B_total_cand = np.maximum(C["bkg_SR"][idx], np.float32(1e-9))

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

        fine_configs = [
            ("sig", is_signal & in_SR, True),
            ("il", is_interp & in_LSB, True),
            ("ir", is_interp & in_RSB, True),
            ("ni", is_noninterp_bkg & in_SR, True),
            ("bs", is_bkg & ~is_data & in_sideband, True),
            ("bs_L", is_data & in_LSB, True),
            ("bs_R", is_data & in_RSB, True),
        ]
        H = self._build_3d_hists(r_s3d, r_w, bins_3d, fine_configs)
        H_sig, H_il, H_ir, H_ni, H_bs, H_bs_L, H_bs_R = \
            [H[k] for k in ("sig", "il", "ir", "ni", "bs", "bs_L", "bs_R")]

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
        cL = np.float32(COEFF_L)
        cR = np.float32(COEFF_R)
        B_interp = cL * C_il + cR * C_ir
        B = np.maximum(B_interp + C_ni, np.float32(1e-9))
        B_side = C_bs

        valid_mask = (S > 0) & (B_side >= sb_min) & (B_side <= sb_max)
        if self.require_sb_ratio:
            _sb_ratio = np.divide(C_bs_R, C_bs_L,
                                  out=np.full_like(C_bs_R, np.inf),
                                  where=C_bs_L > 0)
            valid_mask = (
                valid_mask
                & (C_bs_L > 0)
                & (_sb_ratio < _WIDTH_R / _WIDTH_L)
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
        bkg_side_list = []

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
            best_coarse_indices = None

            if n_with > 0:
                if n_with > 100:
                    n_with = 100
                    valid_with_events = valid_with_events[:n_with]
                    print(f"  Limiting to top {n_with} populated bins for fine scan ...")
                print(f"  Fine-scanning {n_with} populated bins "
                      f"(n_fine_bins={self.n_fine_bins}, "
                      f"{(self.n_fine_bins+1)**3:,} combos/bin) ...")
                n_workers = min(8, n_with)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    _submits = {}
                    for rec in valid_with_events:
                        i0, i1, i2 = int(rec["i0"]), int(rec["i1"]), int(rec["i2"])
                        f = ex.submit(self._fine_histogram_scan, i0, i1, i2, cat)
                        _submits[f] = (rec, i0, i1, i2)

                    n_scanned = 0
                    for f in as_completed(_submits):
                        n_scanned += 1
                        fine = f.result()
                        rec, i0, i1, i2 = _submits[f]
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
                                best_coarse_indices = (i0, i1, i2)
                        if n_scanned % 25 == 0 or n_scanned == len(_submits):
                            print(f"    ... {n_scanned}/{len(_submits)} scanned, "
                                  f"best Z={best_result['Z'] if best_result else 0:.4f}")

            # Fall back to best empty bin if no fine result
            if best_result is None:
                if n_empty > 0:
                    best_coarse = valid_empty[0]
                    source = "coarse-empty"
                elif n_with > 0:
                    best_coarse = valid_with_events[0]
                    source = "coarse-fallback"
                else:
                    best_coarse = None
                    best_coarse_indices = None
                if best_coarse is not None:
                    best_coarse_indices = (int(best_coarse["i0"]), int(best_coarse["i1"]), int(best_coarse["i2"]))
                    th0, th1, th2 = self._indices_to_thresholds(
                        best_coarse_indices[0], best_coarse_indices[1], best_coarse_indices[2]
                    )
                    th_nonRes, th_Res, th_ggHH = self._trans_thresholds_to_original(th0, th1, th2)
                    
                    best_result = {
                        "th_inv0": th0, "th_inv1": th1, "th_sig": th2,
                        "th_nonRes": th_nonRes, "th_Res": th_Res, "th_ggHH": th_ggHH,
                        "Z": float(best_coarse["Z"]),
                        "s": float(best_coarse["s"]),
                        "b": float(best_coarse["b"]),
                        "b_side": float(best_coarse["b_side"]),
                        "source": source,
                    }
            if best_result is None:
                raise RuntimeError(f"Category {cat}: No valid bins found for both fine and coarse scans.")
            best_result["best_coarse_indices"] = best_coarse_indices
            
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
                (s3d[:, 0] >= best_result["th_inv0"])
                & (s3d[:, 1] >= best_result["th_inv1"])
                & (s3d[:, 2] >= best_result["th_sig"])
            )

            if mask_local.sum() > 0:
                ml = self.labels_all[mask_local]
                mm = self.mass_all[mask_local]
                mw = self.weights_all[mask_local]
                ms = self.samples_all[mask_local]
                in_peak = (mm > MASS_SR[0]) & (mm < MASS_SR[1])
                sig_peak = mw[in_peak & (ml == 1)].sum()
                bkg_side = mw[
                    ((mm < MASS_LEFT_SB[1]) | (mm > MASS_RIGHT_SB[0]))
                    & (ml == 0) & (ms != "Data")
                ].sum()
                data_side = mw[
                    ((mm < MASS_LEFT_SB[1]) | (mm > MASS_RIGHT_SB[0]))
                    & (ml == 0) & (ms == "Data")
                ].sum()
                is_data = ms == "Data"
                data_L = mw[is_data & (mm >= MASS_LEFT_SB[0]) & (mm < MASS_LEFT_SB[1])].sum()
                data_R = mw[is_data & (mm >= MASS_RIGHT_SB[0]) & (mm < MASS_RIGHT_SB[1])].sum()
            else:
                sig_peak = 0.0
                bkg_side = 0.0
                data_side = 0.0
                data_L = 0.0
                data_R = 0.0

            sig_peak_list.append(sig_peak)
            bkg_side_list.append(bkg_side)
            data_ratio_str = ""
            if self.require_sb_ratio and data_L > 0:
                data_ratio = data_R / data_L
                limit = _WIDTH_R / _WIDTH_L
                data_ratio_str = f"  Data SB ratio R/L={data_R:.0f}/{data_L:.0f}={data_ratio:.2f} (limit={limit:.2f})"
                if data_ratio >= limit:
                    data_ratio_str += " *** EXCEEDS LIMIT ***"
            elif self.require_sb_ratio:
                data_ratio_str = "  Data SB ratio: N/A (no data in LSB)"
            print(f"  Signal in SR: {sig_peak:.3g}  Bkg in SB: {bkg_side:.3g}  Data in SB: {data_side:.3g} "
                  f"Removed: {mask_local.sum():,}{data_ratio_str}")
            
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
        has_boosted = self.apply_boosted_veto
        has_vbf = self.vbf_thresholds is not None

        if has_boosted:
            cat_strings["cat0"] = (
                "is_boosted == 1"
            )

        if has_vbf:
            vbf_col = SCORE_COLS[self.vbfhh_class]
            vbf_parts= [f"({vbf_col} >= {self._remove_fake_digits(self.vbf_thresholds)})"]
            if self.vbf_2d_scan and self.vbfhh_nonres_threshold is not None:
                nonres_col = SCORE_COLS[self.bkg_classes[0]]
                vbf_parts.append(f"({nonres_col} < {self._remove_fake_digits(self.vbfhh_nonres_threshold)})")
            cat_strings["cat1"] = " & ".join(vbf_parts)

        grid_shift = 1 if has_vbf else 0
        for i, base in enumerate(base_cuts, start=1):
            parts = [base]
            if i >= 2:
                for j in range(i - 1):
                    parts.append(f"not({base_cuts[j]})")
            cat_idx = i + grid_shift
            cat_str = " & ".join(parts)
            if has_vbf:
                cat_str += f" & (not({cat_strings['cat1']}))"
            cat_strings[f"cat{cat_idx}"] = cat_str
        
        # post process: add is_boosted == 0 and (dijet_mass > 80) & (dijet_mass < 190) to all categories except cat0
        for key in cat_strings.keys():
            if key != "cat0":
                _base_cut = " & (dijet_mass > 80) & (dijet_mass < 190)"
                _base_cut += " & (is_boosted == 0)" if has_boosted else ""
                cat_strings[key] = f"{cat_strings[key]}{_base_cut}"

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
        """1D scan over VBFHH_score > threshold."""
        if self.scores_all is None:
            raise RuntimeError("scores_all is not set. Call load_samples() first.")

        scores = self.scores_all[:, vbfhh_class]
        masses = self.mass_all
        weights = self.weights_all
        samples = self.samples_all

        sr_low, sr_high = MASS_SR

        is_vbfhh = np.isin(samples, vbfhh_samples)
        is_interp = (~is_vbfhh) & np.isin(samples, list(INTERP_SAMPLES))
        is_data = samples == "Data"

        in_sr = (masses > sr_low) & (masses < sr_high)
        in_side = ((masses <= sr_low) | (masses >= sr_high))

        s_cond = is_vbfhh & in_sr
        side_data_cond = is_data & in_side
        interp_L_cond = is_interp & (masses <= sr_low)
        interp_R_cond = is_interp & (masses >= sr_high)

        bkg_noninterp = [sname for sname in self.bkg_samples if sname not in INTERP_SAMPLES]
        b_noninterp_conds = []
        for sname in bkg_noninterp:
            cond = (~is_vbfhh) & (~is_interp) & (samples == sname) & in_sr
            b_noninterp_conds.append(cond)

        scores_clipped = np.clip(scores, 0.0, 1.0)
        bins = np.linspace(0.0, 1.0, n_scan_points + 1)
        thresholds = bins[:-1]
        w = weights.astype(np.float64)

        def get_cumsum(cond):
            if not np.any(cond):
                return np.zeros(n_scan_points, dtype=np.float64)
            hist, _ = np.histogram(scores_clipped[cond], bins=bins, weights=w[cond])
            return np.cumsum(hist[::-1])[::-1]

        s_cum = get_cumsum(s_cond)
        side_data_cum = get_cumsum(side_data_cond)
        b_interp_L_cum = get_cumsum(interp_L_cond)
        b_interp_R_cum = get_cumsum(interp_R_cond)
        
        b_total = COEFF_L * b_interp_L_cum + COEFF_R * b_interp_R_cum
        for cond in b_noninterp_conds:
            b_total += get_cumsum(cond)

        best_threshold = None
        best_z = -1.0
        best_s = 0.0
        best_b = 0.0

        for i in range(n_scan_points):
            s = s_cum[i]
            side = b_interp_L_cum[i] + b_interp_R_cum[i]
            if side_data_cum[i] < sideband_threshold:
                continue
            if s <= 0.0 or side < sideband_threshold:
                continue

            b = b_total[i]
            if b <= 0.0:
                b = 1e-9

            z = self.asymptotic_significance(s, b)
            if z > best_z:
                best_z = z
                best_threshold = thresholds[i]
                best_s = s
                best_b = b
                best_side = side
        print(f"VBFHH SR 1D: best threshold = {best_threshold:.4f}, "
              f"Z = {best_z:.4f}, s = {best_s:.4g}, b = {best_b:.4g}, sideband = {best_side:.4g}")

        return best_threshold, best_z, best_s, best_b

    def optimize_vbfhh_2d(self, vbfhh_class, nonres_class, vbfhh_samples,
                           n_scan_vbfhh=1000, n_scan_nonres=500,
                           sideband_threshold=10.0, VBF_purity_threshold=0.85):
        """
        2D grid scan: VBFHH_score > th_vbf AND nonRes_score < th_nonres.

        Candidate points must also satisfy
        VBF purity = VBFHH / (VBFHH + ggHH) >= VBF_purity_threshold.

        Uses 2D histograms + 2D reverse cumulative sum for O(1) per-pair
        evaluation across all (n_scan_vbfhh x n_scan_nonres) threshold combos.
        """
        if self.scores_all is None:
            raise RuntimeError("scores_all is not set. Call load_samples() first.")

        scores = self.scores_all
        masses = self.mass_all
        weights = self.weights_all.astype(np.float32)
        samples = self.samples_all

        sr_low, sr_high = MASS_SR

        is_vbfhh = np.isin(samples, vbfhh_samples)
        is_interp = (~is_vbfhh) & np.isin(samples, list(INTERP_SAMPLES))
        is_data = samples == "Data"

        in_sr = (masses > sr_low) & (masses < sr_high)
        in_side = (masses <= sr_low) | (masses >= sr_high)
        data_in_side = (masses <= 115) | (masses >= 135)

        score_vbf = scores[:, vbfhh_class]
        score_inv_nonres = 1.0 - scores[:, nonres_class]
        is_ggHH = self.labels_all != 0

        bins0 = np.linspace(0.0, 1.0, n_scan_vbfhh + 1)
        bins1 = np.linspace(0.0, 1.0, n_scan_nonres + 1)

        i0 = np.searchsorted(bins0, score_vbf, side='right') - 1
        i1 = np.searchsorted(bins1, score_inv_nonres, side='right') - 1
        in_range = ((i0 >= 0) & (i0 < n_scan_vbfhh)
                    & (i1 >= 0) & (i1 < n_scan_nonres))
        i0c = np.clip(i0, 0, n_scan_vbfhh - 1)
        i1c = np.clip(i1, 0, n_scan_nonres - 1)
        lin = i0c * n_scan_nonres + i1c
        total = n_scan_vbfhh * n_scan_nonres
        shape = (n_scan_vbfhh, n_scan_nonres)

        def h2d(cond, use_w=True):
            m = cond & in_range
            if m.sum() == 0:
                return np.zeros(shape, dtype=np.float32)
            w = weights[m] if use_w else None
            h = np.bincount(lin[m], weights=w, minlength=total)
            if not use_w:
                h = h.astype(np.float32)
            return h.reshape(shape)

        H_s = h2d(is_vbfhh & in_sr)
        H_gg = h2d(is_ggHH & in_sr)
        H_iL = h2d(is_interp & (masses <= sr_low))
        H_iR = h2d(is_interp & (masses >= sr_high))
        bkg_ni = [s for s in self.bkg_samples
                   if s not in INTERP_SAMPLES]
        H_ni = sum(h2d(~is_vbfhh & ~is_interp & (samples == s) & in_sr) for s in bkg_ni)
        H_ds = h2d(is_data & data_in_side)

        def cumsum2d(hist):
            rev = np.ascontiguousarray(hist[::-1, ::-1])
            for ax in range(2):
                np.cumsum(rev, axis=ax, out=rev)
            return rev[::-1, ::-1]

        C_s = cumsum2d(H_s)
        C_gg = cumsum2d(H_gg)
        C_iL = cumsum2d(H_iL)
        C_iR = cumsum2d(H_iR)
        C_ni = cumsum2d(H_ni)
        C_ds = cumsum2d(H_ds)

        cL = np.float32(COEFF_L)
        cR = np.float32(COEFF_R)
        C_side = C_iL + C_iR
        C_b = cL * C_iL + cR * C_iR + C_ni
        C_purity_den = C_s + C_gg

        with np.errstate(invalid="ignore", divide="ignore"):
            C_purity = np.divide(
                C_s,
                C_purity_den,
                out=np.zeros_like(C_s, dtype=np.float32),
                where=C_purity_den > 0,
            )

        valid = (
            (C_s > 0)
            & (C_ds >= sideband_threshold)
            & (C_side >= sideband_threshold)
            & (C_b > 0)
            & (C_purity >= VBF_purity_threshold)
        )
        if not np.any(valid):
            print("VBFHH 2D: No valid cells found")
            return None, None, 0.0, 0.0, 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.where(
                valid,
                np.sqrt(2.0 * ((C_s + C_b) * np.log(1.0 + C_s / C_b) - C_s)),
                0.0,
            )

        # -- Diagnostics ------------------------------------------------
        print(f"  VBFHH signal: total events={is_vbfhh.sum()}, "
              f"total weight={weights[is_vbfhh].sum():.3f}, "
              f"SR weight={weights[is_vbfhh & in_sr].sum():.3f}")

        best_flat = np.argmax(Z.ravel())
        best_i, best_j = np.unravel_index(best_flat, Z.shape)
        th_vbf = float(bins0[best_i])
        th_nonres = float(1.0 - bins1[best_j])
        best_z = float(Z[best_i, best_j])
        best_s = float(C_s[best_i, best_j])
        best_b = float(C_b[best_i, best_j])
        best_purity = float(C_purity[best_i, best_j])

        print(f"  Best: th_vbf={th_vbf:.4f} th_nonres={th_nonres:.4f} "
              f"Z={best_z:.4f} s={best_s:.4g} b={best_b:.4g} "
              f"purity={best_purity:.4f} "
              f"Data_SB={float(C_ds[best_i, best_j]):.0f} "
              f"interp_SB={float(C_side[best_i, best_j]):.4g}")

        # Top-10 distinct combinations
        flat_Z = Z.ravel()
        top10 = np.argsort(flat_Z)[-10:][::-1]
        print(f"  Top Z thresholds (vbf_th, nonres_th, Z, s, b, Data_SB, interp_SB):")
        for idx in top10:
            if flat_Z[idx] <= 0:
                break
            i, j = np.unravel_index(idx, Z.shape)
            print(f"    th_v={bins0[i]:.4f} th_nr={1-bins1[j]:.4f} "
                  f"Z={flat_Z[idx]:.4f} s={C_s[i,j]:.4g} b={C_b[i,j]:.4g} "
                  f"purity={C_purity[i,j]:.4f} "
                  f"ds={C_ds[i,j]:.0f} isb={C_side[i,j]:.4g}")

        # 1D profile: Z vs VBFHH threshold (maximizing over nonRes per VBFHH bin)
        z_vs_vbf = Z.max(axis=1)
        best_vbf_idx = np.argmax(z_vs_vbf)
        best_nonres_for_best_vbf = np.argmax(Z[best_vbf_idx])
        print(f"  1D Z profile (max over nonRes per VBFHH bin): "
              f"peak at th_vbf={bins0[best_vbf_idx]:.4f} "
              f"Z={z_vs_vbf[best_vbf_idx]:.4f} "
              f"th_nonres={1-bins1[best_nonres_for_best_vbf]:.4f}")
        # Print a few points along the peak
        step = max(n_scan_vbfhh // 10, 1)
        for i in range(0, n_scan_vbfhh, step):
            if z_vs_vbf[i] > 0.01:
                best_jj = np.argmax(Z[i])
                print(f"    th_vbf={bins0[i]:.4f} "
                      f"Z_max={z_vs_vbf[i]:.4f} "
                      f"th_nonres@max={1-bins1[best_jj]:.4f} "
                      f"s={C_s[i,best_jj]:.4g} b={C_b[i,best_jj]:.4g} "
                      f"purity={C_purity[i,best_jj]:.4f}")

        return th_vbf, th_nonres, best_z, best_s, best_b


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
                elif self.vbf_2d_scan:
                    nonres_class = self.bkg_classes[0]
                    (best_vbfhh_threshold, best_vbfhh_nonres, best_vbfhh_z,
                     best_vbfhh_s, best_vbfhh_b) = self.optimize_vbfhh_2d(
                        vbfhh_class=self.vbfhh_class,
                        nonres_class=nonres_class,
                        vbfhh_samples=self.vbfhh_samples,
                        sideband_threshold=self.vbfhh_sideband_threshold,
                    )
                    self.vbfhh_nonres_threshold = best_vbfhh_nonres
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

                if self.vbf_2d_scan:
                    print(f"VBFHH SR {i_cat + 1}: th_vbf={best_vbfhh_threshold:.4f}, "
                          f"th_nonres={self.vbfhh_nonres_threshold:.4f}, "
                          f"Z={best_vbfhh_z:.4f}, s={best_vbfhh_s:.4g}, "
                          f"b={best_vbfhh_b:.4g}")
                    vbfhh_mask = ((self.scores_all[:, self.vbfhh_class] > best_vbfhh_threshold)
                                  & (self.scores_all[:, nonres_class] < best_vbfhh_nonres))
                else:
                    print(f"VBFHH SR {i_cat + 1}: threshold = {best_vbfhh_threshold:.4f}, "
                          f"Z = {best_vbfhh_z:.4f}, s = {best_vbfhh_s:.4g}, "
                          f"b = {best_vbfhh_b:.4g}")
                    vbfhh_mask = self.scores_all[:, self.vbfhh_class] > best_vbfhh_threshold
                sr_mass_mask = (self.mass_all > sr_low) & (self.mass_all < sr_high)
                sb_mask = (self.mass_all <= sr_low) | (self.mass_all >= sr_high)
                is_vbfhh = np.isin(self.samples_all, self.vbfhh_samples)
                is_ggHH = self.labels_all != 0  # SM signal (1) + non-SM ggHH (-1)
                is_interp = np.isin(self.samples_all, list(INTERP_SAMPLES))
                is_data = self.samples_all == "Data"
                is_mc = ~is_data

                best_ggHH = float(
                    self.weights_all[vbfhh_mask & sr_mass_mask & is_ggHH].sum()
                )
                best_SingleH_SR = float(self.weights_all[
                    vbfhh_mask & sr_mass_mask & ~is_vbfhh & ~is_ggHH & ~is_interp & is_mc
                ].sum())
                best_interp_SB = float(self.weights_all[
                    vbfhh_mask & sb_mask & is_interp
                ].sum())
                best_Data_SB = float(self.weights_all[
                    vbfhh_mask & sb_mask & is_data
                ].sum())

                print(f"VBFHH SR {i_cat + 1}: ggHH_SR={best_ggHH:.4g}, "
                      f"SingleHiggs_SR={best_SingleH_SR:.4g}, "
                      f"b_total(MC_estimate)={best_vbfhh_b:.4g}")
                print(f"  interp_SB={best_interp_SB:.4g}, Data_SB={best_Data_SB:.4g}")

                all_vbfhh_sr_info.append({
                    "best_threshold": float(best_vbfhh_threshold),
                    "best_z": float(best_vbfhh_z),
                    "best_s": float(best_vbfhh_s),
                    "best_b": float(best_vbfhh_b),
                    "best_ggHH": best_ggHH,
                    "best_SingleH_SR": best_SingleH_SR,
                    "best_interp_SB": best_interp_SB,
                    "best_Data_SB": best_Data_SB,
                })

                remaining = ~vbfhh_mask
                print(f"VBFHH SR {i_cat + 1} removes {vbfhh_mask.sum():,} events; "
                      f"{remaining.sum():,} remain for SR optimisation.")
                self.scores_all = self.scores_all[remaining]
                self.scores_trans = self.scores_trans[remaining]
                self.mass_all = self.mass_all[remaining]
                self.dijet_mass_all = self.dijet_mass_all[remaining]
                self.weights_all = self.weights_all[remaining]
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

        if self.skip_grid_search:
            print("\n--skip-grid-search set, stopping after VBFHH optimization.")
            return

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
    parser.add_argument("--vbfhh_2d_scan", action="store_true", default=False,
                        help="Use 2D scan (VBFHH+nonRes) instead of 1D VBFHH-only")
    parser.add_argument("--skip_grid_search", action="store_true", default=False,
                        help="Stop after VBFHH optimization, skip grid search")

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
    categoriser.vbf_2d_scan = args.vbfhh_2d_scan
    categoriser.skip_grid_search = args.skip_grid_search
    categoriser.weight_scale = args.weight_scale
    categoriser.run_categorisation()
