#!/usr/bin/env python3

import ast
import gc
import os
import awkward as ak
import numpy as np
import yaml
import matplotlib.pyplot as plt

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import json
import mplhep
from matplotlib.backends.backend_pdf import PdfPages

hep.style.use("CMS")


SAMPLE_STYLES = {
    # Backgrounds: colors follow the active plot_stacked_histogram palette in
    # the reference script, keyed by raw sample name so they never depend on
    # which other samples happen to be present.
    "VBFHToGG_M_125": {"label": "VBFH", "color": "#FF8A50"},
    "VHtoGG_M_125": {"label": "VH", "color": "#FFB300"},
    "ttHtoGG_M_125": {"label": "ttH", "color": "#66BB6A"},
    "BBHToGG_M_125": {"label": "bbH", "color": "#C0CA33"},
    "GluGluHToGG_M_125": {"label": "ggH", "color": "#26A69A"},
    "TTGG": {"label": "TTGG", "color": "#09529A"},
    "GGJets": {"label": "GGJets", "color": "#0B66C1"},
    "DDQCDGJET": {"label": "DDQCDGJets", "color": "#5E60CE"},
    # ggHH signals.
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": {
        "label": "ggHH SM", "color": "red"
    },
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": {
        "label": "ggHH kl=0.00", "color": "blue"
    },
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": {
        "label": "ggHH kl=2.45", "color": "orange"
    },
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": {
        "label": "ggHH kl=5.00", "color": "brown"
    },
    "VBFToHH_CV-1p000_C2V-1p000_C3-1p000": {
        "label": "VBFHH SM", "color": "green"
    },
}

# Names stored by the merged-parquet producer.  Reuse the exact same style as
# the corresponding per-sample directory name.
SAMPLE_STYLE_ALIASES = {
    "VBFHToGG": "VBFHToGG_M_125",
    "VHToGG": "VHtoGG_M_125",
    "ttHToGG": "ttHtoGG_M_125",
    "BBHToGG": "BBHToGG_M_125",
    "GluGluHToGG": "GluGluHToGG_M_125",
    "DDQCDGJets": "DDQCDGJET",
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p00": "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "GluGluToHH_kl-0p00_kt-1p00_c2-0p00": "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
    "GluGluToHH_kl-2p45_kt-1p00_c2-0p00": "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
    "GluGluToHH_kl-5p00_kt-1p00_c2-0p00": "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
}
for alias, source in SAMPLE_STYLE_ALIASES.items():
    SAMPLE_STYLES[alias] = SAMPLE_STYLES[source]

BACKGROUND_ORDER = [
    "VBFHToGG",
    "VBFHToGG_M_125",
    "VHToGG",
    "VHtoGG_M_125",
    "ttHToGG",
    "ttHtoGG_M_125",
    "BBHToGG",
    "BBHToGG_M_125",
    "GluGluHToGG",
    "GluGluHToGG_M_125",
    "TTGG",
    "GGJets",
    "DDQCDGJets",
    "DDQCDGJET",
]

SIGNAL_ORDER = [
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "GluGluToHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
    "GluGluToHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
    "GluGluToHH_kl-5p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
    "GluGluToHH_kl-1p00_kt-1p00_c2-3p00",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00",
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p35",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35",
    "GluGluToHH_kl-0p00_kt-1p00_c2-1p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00",
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p10",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10",
    "GluGluToHH_kl-1p00_kt-1p00_c2-m2p00",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00",
    "GluGluToHH_kl-m20p00_kt-1p00_c2-2p24",
    "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24",
]

CMS_LUMI_LABEL = "137.93 / 282.71"
CMS_COM_LABEL = "13 / 13.6"

# Category-dependent scale factors for the continuum nonresonant background.
# Edit only these two values when the fitted scale factors are available.
NONRES_BKG_SF_BY_CATEGORY_PREFIX = {
    "ggHH-low": 1.39,
    "ggHH-high": 1.31,
}

# Raw parquet sample names to which the nonresonant-background SF is applied.
NONRES_BKG_SAMPLES = {
    "GGJets",
    "DDQCDGJets",
    "DDQCDGJET",
    "TTGG",
}


def _ordered_samples(samples, preferred_order):
    """Return known samples in the configured order, then unknowns stably."""
    samples = set(samples)
    return [s for s in preferred_order if s in samples] + sorted(
        samples - set(preferred_order)
    )


def _sample_style(sample):
    return SAMPLE_STYLES.get(sample, {"label": sample, "color": "#757575"})


def _nonres_bkg_sf(category):
    """Return the configured nonresonant-background SF for one SR."""
    for prefix, scale_factor in NONRES_BKG_SF_BY_CATEGORY_PREFIX.items():
        if category.startswith(prefix):
            return float(scale_factor)
    return 1.0



# ============================================================
#  SR cut definition (hard-coded, frozen)
# ============================================================

def load_sr_cuts(json_path):
    """Load ordered category-name -> cut-expression entries from JSON."""
    with open(json_path, "r", encoding="utf-8") as handle:
        cuts = json.load(handle)

    if not isinstance(cuts, dict) or not cuts:
        raise ValueError("SR cuts JSON must be a non-empty object")

    for category, expression in cuts.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every SR category name must be a non-empty string")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(
                f"Cut for category {category!r} must be a non-empty string"
            )
        try:
            ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ValueError(
                f"Invalid cut expression for category {category!r}: {error}"
            ) from error

    return cuts


def evaluate_cut(expression, events):
    """Safely evaluate Python-style array cuts without using eval()."""
    parsed = ast.parse(expression, mode="eval")
    n_events = len(events)
    cut_fields = {
        node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)
    }
    valid_inputs = np.ones(n_events, dtype=bool)
    for field in cut_fields:
        if field not in events.fields:
            raise KeyError(
                f"Column {field!r} required by cut {expression!r} "
                "is missing from the input parquet"
            )
        valid_inputs &= ak.to_numpy(~ak.is_none(events[field], axis=0))

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Name):
            if node.id not in events.fields:
                raise KeyError(
                    f"Column {node.id!r} required by cut {expression!r} "
                    "is missing from the input parquet"
                )
            # Parquet columns may be nullable for samples where a quantity is
            # undefined.  NaN makes numeric comparisons false; valid_inputs
            # below also guarantees missing values fail every cut, including
            # expressions that use !=.
            return ak.to_numpy(
                ak.fill_none(events[node.id], np.nan, axis=0)
            )

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, bool))
        ):
            return node.value

        if isinstance(node, ast.BoolOp):
            # Combine operands one at a time.  For tens of millions of events,
            # retaining every comparison result in a list can consume several
            # hundred MB for a single category cut.
            operands = iter(node.values)
            result = np.array(visit(next(operands)), dtype=bool, copy=True)
            operation = (
                np.logical_and
                if isinstance(node.op, ast.And)
                else np.logical_or
            )
            for operand in operands:
                value = np.asarray(visit(operand), dtype=bool)
                operation(result, value, out=result)
            return result

        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.Not):
                return np.logical_not(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value

        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            operations = {
                ast.Add: np.add,
                ast.Sub: np.subtract,
                ast.Mult: np.multiply,
                ast.Div: np.divide,
                ast.Pow: np.power,
                ast.Mod: np.mod,
            }
            for operator_type, operation in operations.items():
                if isinstance(node.op, operator_type):
                    return operation(left, right)

        if isinstance(node, ast.Compare):
            left = visit(node.left)
            result = np.ones(n_events, dtype=bool)
            operations = {
                ast.Gt: np.greater,
                ast.GtE: np.greater_equal,
                ast.Lt: np.less,
                ast.LtE: np.less_equal,
                ast.Eq: np.equal,
                ast.NotEq: np.not_equal,
            }
            for operator, comparator in zip(node.ops, node.comparators):
                right = visit(comparator)
                operation = next(
                    (
                        function
                        for operator_type, function in operations.items()
                        if isinstance(operator, operator_type)
                    ),
                    None,
                )
                if operation is None:
                    raise ValueError(
                        f"Unsupported comparison in cut {expression!r}"
                    )
                result &= operation(left, right)
                left = right
            return result

        raise ValueError(
            f"Unsupported syntax in cut {expression!r}: {ast.dump(node)}"
        )

    value = np.asarray(visit(parsed))
    if value.ndim == 0:
        value = np.full(n_events, bool(value), dtype=bool)
    if len(value) != n_events:
        raise ValueError(
            f"Cut {expression!r} returned {len(value)} entries for "
            f"{n_events} events"
        )
    value = value.astype(bool, copy=False)
    np.logical_and(value, valid_inputs, out=value)
    return value


def plot_stacked_histogram_from_events(
    mc_events_dict,     # dict: sample -> awkward array (already SR-selected)
    data_events,        # awkward array (already SR-selected)
    variables,
    out_path,
    category,
    signal_scale=1000,
    mass_window=(120, 130),
    only_MC=False,
    blind=True,
):
    """
    Plot stacked MC vs Data histograms from SR-selected events.
    - SR cuts are assumed to be already applied
    - Data is blinded in the diphoton mass window
    - Plotting style follows the reference CMS Data/MC plotter
    """

    import os
    import numpy as np
    import awkward as ak
    import matplotlib.pyplot as plt
    import mplhep as hep

    os.makedirs(out_path, exist_ok=True)
    nonres_bkg_sf = _nonres_bkg_sf(category)
    if nonres_bkg_sf != 1.0:
        print(
            f"[info] {category}: applying nonres background SF "
            f"{nonres_bkg_sf:g}"
        )

    signal_scale = {
        "ggHH": 100,
        "VBFHH": 10000,
    }

    # --------------------------------------------------
    # Variable config (keep identical ranges / bins)
    # --------------------------------------------------
    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 24, "range": (100, 180), "log": True},
        "dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 190), "log": True},
        "HHbbggCandidate_mass": {"label": r"$M_X^{reg}$ [GeV]", "bin_edges": [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1125, 1300, 1500], "log": True, "normalized": False},
        "Dn": {"label": r"$D_{\mathrm{sig/nonres}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "Dt": {"label": r"$D_{\mathrm{sig/ttH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "DsH": {"label": r"$D_{\mathrm{sig/singleH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "nonRes_score": {"label": "nonRes_score", "bins": 30, "range": (0, 1), "log": True},
        "singleH_score": {"label": "singleH_score", "bins": 30, "range": (0, 1), "log": True},
        "ggHH_score": {"label": "ggHH_score", "bins": 30, "range": (0, 1), "log": True},
        "nonResReg_vbfpair_HHbbggCandidate_mass": {"label": r"$m_{\mathrm{HHbbgg}}^{reg}$ [GeV]", "bins": 40, "range": (0, 4000), "log": True},
        "nonResReg_vbfpair_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 190), "log": True},
        "nonResReg_vbfpair_M_X": {"label": r"$M_X^{reg}$ [GeV]", "bin_edges": [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1125, 1300, 1500], "log": True, "normalized": True},
        "D_sig_vs_ttH": {"label": r"$D_{\mathrm{sig/ttH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "D_sig_vs_nonres": {"label": r"$D_{\mathrm{sig/nonres}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "D_sig_vs_nonres_only": {"label": r"$D_{\mathrm{sig/nonres-only}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "D_sig_vs_singleH": {"label": r"$D_{\mathrm{sig/singleH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "D_sig_vs_ttH_singleH": {"label": r"$D_{\mathrm{sig/ttH/singleH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "D_singleH_vs_ttH": {"label": r"$D_{\mathrm{singleH/ttH}}$", "bins": 30, "range": (0.9, 1), "log": True},
        "non_resonant_bkg_score": {"label": "non_resonant_bkg_score", "bins": 30, "range": (0, 1), "log": True},
        "ttH_score": {"label": "ttH_score", "bins": 30, "range": (0, 1), "log": True},
        "other_single_H_score": {"label": "other_single_H_score", "bins": 30, "range": (0, 1), "log": True},
        "GluGluToHH_score": {"label": "GluGluToHH_score", "bins": 30, "range": (0, 1), "log": True},
        "boosted_score": {"label": "boosted_score", "bins": 50, "range": (0, 1), "log": True},
        "lead_mvaID": {"label": "lead photon MVA ID", "bins": 30, "range": (-0.7, 1), "log": True},
        "sublead_mvaID": {"label": "sublead photon MVA ID", "bins": 30, "range": (-0.7, 1), "log": True},
    }

    # --------------------------------------------------
    # Split signal / background (same logic as before)
    # --------------------------------------------------
    valid_samples = {}
    for sample, events in mc_events_dict.items():
        if events is None or len(events) == 0:
            continue
        if sample not in SAMPLE_STYLES:
            print(
                f"[warning] no fixed plotting style configured for '{sample}'; "
                "skipping this sample"
            )
            continue
        valid_samples[sample] = events

    background_samples = [s for s in valid_samples if "HH" not in s]
    signal_samples = [s for s in valid_samples if "HH" in s]
    stack_mc_dict = {
        sample: valid_samples[sample]
        for sample in _ordered_samples(background_samples, BACKGROUND_ORDER)
    }
    signal_mc_dict = {
        sample: valid_samples[sample]
        for sample in _ordered_samples(signal_samples, SIGNAL_ORDER)
    }

    # --------------------------------------------------
    # Blind DATA in mgg window (MC and data)
    # --------------------------------------------------
    if blind:
        if "mass" in data_events.fields:
            data_events = data_events[
                (data_events.mass < mass_window[0]) |
                (data_events.mass > mass_window[1])
            ]
        for sample in list(mc_events_dict.keys()):
            ev = mc_events_dict[sample]
            if ev is not None and len(ev) > 0 and "mass" in ev.fields:
                mc_events_dict[sample] = ev[
                    (ev.mass < mass_window[0]) |
                    (ev.mass > mass_window[1])
                ]

    # --------------------------------------------------
    # Loop over variables
    # --------------------------------------------------
    for variable in variables:
        if variable not in data_events.fields:
            print(f"[skip] variable {variable} not in data")
            continue

        cfg = var_config[variable]
        normalized = bool(cfg.get("normalized", False))
        if "bin_edges" in cfg:
            bin_edges = np.asarray(cfg["bin_edges"], dtype=float)
            if bin_edges.ndim != 1 or len(bin_edges) < 2:
                raise ValueError(f"[config error] {variable}: bin_edges must be a 1D array with at least 2 entries")
            if not np.all(np.diff(bin_edges) > 0):
                raise ValueError(f"[config error] {variable}: bin_edges must be strictly increasing")
            x_range = (bin_edges[0], bin_edges[-1])
        else:
            bin_edges = np.linspace(*cfg["range"], cfg["bins"] + 1)
            x_range = cfg["range"]

        mc_hist = []
        mc_err = np.zeros(len(bin_edges) - 1)
        mc_labels = []
        mc_colors_used = []

        # ----------------------------
        # MC histograms
        # ----------------------------
        for sample, events in stack_mc_dict.items():
            values = ak.to_numpy(events[variable])
            weights = ak.to_numpy(events["weight_tot"])
            sample_sf = (
                nonres_bkg_sf if sample in NONRES_BKG_SAMPLES else 1.0
            )
            weights = weights * sample_sf

            hist, _ = np.histogram(values, bins=bin_edges, weights=weights)
            err2, _ = np.histogram(values, bins=bin_edges, weights=weights**2)

            mc_hist.append(hist)
            mc_err += err2
            style = _sample_style(sample)
            mc_labels.append(
                f"{style['label']} x {sample_sf:g}"
                if sample_sf != 1.0 else style["label"]
            )
            mc_colors_used.append(style["color"])

        if mc_hist:
            mc_total = np.sum(np.asarray(mc_hist), axis=0)
        else:
            mc_total = np.zeros(len(bin_edges) - 1, dtype=float)
        mc_err = np.sqrt(mc_err)

        if normalized:
            mc_norm = np.sum(mc_total)
            if mc_norm > 0:
                mc_hist = [h / mc_norm for h in mc_hist]
                mc_total = mc_total / mc_norm
                mc_err = mc_err / mc_norm

        # ----------------------------
        # Data histogram
        # ----------------------------
        data_vals = ak.to_numpy(data_events[variable])
        data_hist, _ = np.histogram(data_vals, bins=bin_edges)
        data_err = np.sqrt(data_hist)

        if normalized:
            data_norm = np.sum(data_hist)
            if data_norm > 0:
                data_hist = data_hist / data_norm
                data_err = data_err / data_norm
        
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     ratio = data_hist / mc_total
        #     ratio_err = data_err / mc_total
        #     mc_ratio_err = mc_err / mc_total

        # template solution for those bins where mc_total <=0
        ####### begin ########
        ratio = np.zeros(len(bin_edges) - 1, dtype=float)
        ratio_err = np.zeros(len(bin_edges) - 1, dtype=float)
        mc_ratio_err = np.zeros(len(bin_edges) - 1, dtype=float)

        valid = mc_total > 0
        ratio[valid] = data_hist[valid] / mc_total[valid]
        ratio_err[valid] = data_err[valid] / mc_total[valid]
        mc_ratio_err[valid] = mc_err[valid] / mc_total[valid]
        ####### end #######

        # ----------------------------
        # Signal histograms (same style)
        # ----------------------------
        signal_hists = {}
        for sample, events in signal_mc_dict.items():
            if sample.startswith(("GluGluToHH", "GluGlutoHH")):
                sample_scale = signal_scale["ggHH"]
            elif sample.startswith("VBFToHH"):
                sample_scale = signal_scale["VBFHH"]
            else:
                sample_scale = 1

            values = ak.to_numpy(events[variable])
            weights = ak.to_numpy(events["weight_tot"])
            hist, _ = np.histogram(values, bins=bin_edges, weights=weights)
            err2, _ = np.histogram(values, bins=bin_edges, weights=weights**2)
            if normalized:
                sig_norm = np.sum(hist)
                if sig_norm > 0:
                    hist = hist / sig_norm
                    err2 = err2 / (sig_norm**2)
            signal_hists[sample] = {
                "hist": hist * sample_scale,
                "err": np.sqrt(err2) * sample_scale,
                "scale": sample_scale,
            }

        # ----------------------------
        # Plot using the reference Data/MC style.
        # ----------------------------
        if only_MC:
            fig, ax = plt.subplots(figsize=(10, 10))
        else: 
            fig, axs = plt.subplots(
                2, 1,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
                figsize=(10, 10),
                sharex=True
            )
            ax, ax_ratio = axs

        hep.cms.label(
            data=not only_MC,
            lumi=CMS_LUMI_LABEL,
            ax=ax,
            loc=0,
            fontsize=16,
            label="Private Work",
            com=CMS_COM_LABEL,
        )

        if mc_hist:
            hep.histplot(
                mc_hist,
                bin_edges,
                histtype="fill",
                stack=True,
                label=mc_labels,
                color=mc_colors_used,
                edgecolor="black",
                ax=ax,
            )

        ax.fill_between(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            mc_total - mc_err,
            mc_total + mc_err,
            color="gray",
            alpha=0.5,
            step="mid",
        )
        
        print("mc_total min =", np.nanmin(mc_total))
        print("mc_total <= 0 bins =", np.where(mc_total <= 0))
        print("mc_total values =", mc_total)
        print("ratio_err min =", np.nanmin(ratio_err))
        print("mc_ratio_err min =", np.nanmin(mc_ratio_err))

        if not only_MC:
            ax.errorbar(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                data_hist,
                yerr=data_err,
                fmt="o",
                color="black",
                label="Data",
                markersize=5,
            )

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for sample, sig in signal_hists.items():
            style = _sample_style(sample)
            hist = sig["hist"]
            err = sig["err"]

            ax.step(
                centers,
                hist,
                where="mid",
                linestyle="solid" if normalized else "dashed",
                linewidth=2,
                color=style["color"],
                label=(
                    f"{style['label']} x {sig['scale']}"
                    if sig["scale"] != 1 else style["label"]
                ),
            )

            if normalized:
                yerr_low = np.minimum(err, hist * 0.9999)
                yerr_low = np.maximum(yerr_low, 0)
                ax.errorbar(
                    centers,
                    hist,
                    yerr=[yerr_low, err],
                    fmt="none",
                    color=style["color"],
                    capsize=0,
                )

        sig_max = max((np.max(sig["hist"]) for sig in signal_hists.values()), default=0.0)
        y_max = max(np.max(mc_total), np.max(data_hist), sig_max)

        ax.set_xlim(x_range)
        if only_MC:
            ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Normalized Events" if normalized else "Events")
        if cfg["log"]:
            ax.set_yscale("log")
            if normalized:
                ymin = 1e-3
                ymax = max(1.5 * y_max, 1.0)
            else:
                ymin = 1e-3
                if variable == "dijet_mass":
                    ymax = max(5000 * y_max, 1.0)
                # elif "HHbbggCandidate_mass" in variable:
                #     print(f"HHbbggCandidate_mass: setting ymax to 100000 * y_max: {100000 * y_max}")
                #     ymax = max(100000 * y_max, 1.0)
                else:
                    ymax = max(1000 * y_max, 1.0)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0.0, max(1.7 * y_max, 1e-6 if normalized else 1.0))

        blind_title = (
            rf"$m_{{\gamma\gamma}}$ blinded in [{mass_window[0]},{mass_window[1]}] GeV"
            if blind else None
        )
        ax.legend(
            fontsize=14,
            ncol=2,
            title=blind_title,
            title_fontsize=14,
        )

        if not only_MC:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Data / MC points
            ax_ratio.errorbar(
                centers,
                ratio,
                yerr=ratio_err,
                fmt="o",
                color="black",
                markersize=5,
            )

            # MC uncertainty band
            ax_ratio.fill_between(
                centers,
                1 - mc_ratio_err,
                1 + mc_ratio_err,
                color="gray",
                alpha=0.3,
                hatch="xx",
                edgecolor="black",
                linewidth=0.0,
                step="mid",
            )

            ax_ratio.axhline(1.0, linestyle="--", color="gray")
            ax_ratio.set_ylim(0, 2)
            ax_ratio.set_ylabel("Data / MC")
            ax_ratio.set_xlabel(var_config[variable]["label"])
        
        # if variable == "nonResReg_M_X":
        #     ax.axvline(x=350, color="gray", linestyle="--")
        
        plt.tight_layout()
        plt.savefig(
            f"{out_path}/{variable}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"[done] plots saved in {out_path}")
# ============================================================
#  SR Plotter (lightweight, no Optuna)
# ============================================================

class SRPlotter:
    def __init__(self, input_parquet, sr_cuts_path):
        self.input_parquet = input_parquet
        self.SR_cuts = load_sr_cuts(sr_cuts_path)

    def load_events(self):
        """Load the merged MC+Data parquet once, with all columns intact."""
        print(f"[info] loading merged parquet: {self.input_parquet}")
        events = ak.from_parquet(self.input_parquet)

        cut_columns = set()
        for expression in self.SR_cuts.values():
            parsed = ast.parse(expression, mode="eval")
            cut_columns.update(
                node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)
            )

        required_columns = {"sample", "weight_tot"} | cut_columns
        missing_columns = sorted(required_columns - set(events.fields))
        if missing_columns:
            raise KeyError(
                "Input parquet is missing required columns: "
                + ", ".join(missing_columns)
            )

        print(f"[info] loaded {len(events)} events")
        print(f"[info] SR categories: {list(self.SR_cuts)}")
        return events

    def apply_SR(self, events):
        """Yield one selected SR at a time; earlier categories have priority."""
        remaining = np.ones(len(events), dtype=bool)

        for category, expression in self.SR_cuts.items():
            cut_mask = evaluate_cut(expression, events)
            np.logical_and(remaining, cut_mask, out=cut_mask)
            category_count = int(np.count_nonzero(cut_mask))
            category_events = events[cut_mask]
            remaining[cut_mask] = False
            print(
                f"[info] {category}: {category_count} events"
            )

            # Do not retain all SR event copies simultaneously.  The caller
            # plots this category before requesting the next one.
            yield category, category_events

            del category_events, cut_mask
            gc.collect()

    def plot_SR(
        self,
        variables,
        out_dir,
        plot_func,
        signal_scale=1000,
        only_MC=False,
        blind=False,
    ):
        os.makedirs(out_dir, exist_ok=True)
        events = self.load_events()

        available_variables = [
            variable for variable in variables if variable in events.fields
        ]
        missing_variables = [
            variable for variable in variables if variable not in events.fields
        ]
        for variable in missing_variables:
            print(
                f"[warning] optional plotting variable {variable!r} is absent; "
                "skipping it"
            )
        if not available_variables:
            raise ValueError("None of the configured plotting variables exist")

        for category, category_events in self.apply_SR(events):
            print(f"Plotting {category}")

            sample_values = np.asarray(category_events["sample"]).astype(str)
            data_mask = sample_values == "Data"
            data_events = category_events[data_mask]

            mc_events_dict = {}
            for sample in np.unique(sample_values[~data_mask]):
                mc_events_dict[sample] = category_events[
                    sample_values == sample
                ]

            plot_func(
                mc_events_dict=mc_events_dict,
                data_events=data_events,
                variables=available_variables,
                out_path=f"{out_dir}/{category}",
                category=category,
                signal_scale=signal_scale,
                only_MC=only_MC,
                blind=blind,
            )

            # Release all per-category event slices before evaluating the
            # next SR.  This keeps memory proportional to one SR rather than
            # to the sum of every previously selected SR.
            del mc_events_dict, data_events, data_mask, sample_values
            gc.collect()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description=(
            "Apply ordered SR cuts from JSON to one merged MC+Data parquet "
            "and make Data/MC plots"
        )
    )
    parser.add_argument(
        "--input-parquet",
        required=True,
        help="Merged parquet containing sample, era, weight_tot, and all cut columns",
    )
    parser.add_argument(
        "--sr-cuts-json",
        required=True,
        help="Ordered JSON object mapping SR category names to cut expressions",
    )
    parser.add_argument("--out-dir", default="SR_plots/16-24/")
    parser.add_argument(
        "--blind",
        action="store_true",
        default=True,
        help="Blind data in diphoton mass window across all samples and variables",
    )


    args = parser.parse_args()

    variables = [
        "mass",
        "dijet_mass",
        "HHbbggCandidate_mass",
        "Dn",
        "Dt",
        "DsH",
        "nonRes_score",
        "ttH_score",
        "singleH_score",
        "ggHH_score",
        "boosted_score",
        "lead_mvaID",
        "sublead_mvaID",
    ]

    plotter = SRPlotter(
        input_parquet=args.input_parquet,
        sr_cuts_path=args.sr_cuts_json,
    )

    plotter.plot_SR(
        variables=variables,
        out_dir=f"{args.out_dir}",
        plot_func=plot_stacked_histogram_from_events,
        signal_scale=1000,
        only_MC=False,
        blind=args.blind,
    )
