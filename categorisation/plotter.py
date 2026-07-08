import matplotlib.pyplot as plt
import mplhep as hep
from mplhep.error_estimation import poisson_interval
from matplotlib.patches import Patch
from itertools import combinations

plt.style.use([hep.style.CMS])

import os
import numpy as np
import pandas as pd
from hist import Hist
from copy import deepcopy
from sklearn.metrics import log_loss
try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None

try:
    from .train_helper import create_histogram, jensen_shannon_divergence
except Exception:  # pragma: no cover - optional dependency
    create_histogram = None
    jensen_shannon_divergence = None


# following CAT recommendation for colors
PETROFF_COLORS_6 = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
]

PETROFF_COLORS_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

TAB_COLORS = [
    "tab:olive",
    "tab:cyan",
    "tab:green",
    "tab:pink",
    "tab:brown",
    "tab:purple",
    "tab:red",
    "tab:blue",
    "black",
    "white",
]

# Thesis-oriented sizing helpers (inches for Matplotlib).
CM_PER_INCH = 2.54
TEXTWIDTH_CM = 16.2
GUTTER_CM = 0.4
TEXTWIDTH_IN = TEXTWIDTH_CM / CM_PER_INCH
DEFAULT_FIGURE_WIDTH_FRACTION = 1
DEFAULT_FIGURE_LAYOUT = DEFAULT_FIGURE_WIDTH_FRACTION
DEFAULT_ASPECT = 0.7
RATIO_HEIGHT_SCALE = 1.25
BAR_PLOT_ASPECT = 0.55
LINE_PLOT_ASPECT = 0.75

FIGURE_WIDTHS_FRACTION = {
    # Legacy convenience names; prefer explicit width fractions (e.g. 0.45, 0.7).
    "full": 1.0,
    "half": 0.5,
    "third": 1 / 3,
}

FIGURE_WIDTHS_IN = {
    key: value * TEXTWIDTH_IN for key, value in FIGURE_WIDTHS_FRACTION.items()
}

# Base sizes for fractional-width plots (kept consistent by default).
# Units: fonts in points, markers in points, line widths in points.
# Tick/spine/pad values are relative to matplotlib defaults and scaled by width.
BASE_FONT_SIZES = {
    "label": 14, #9
    "ratio_label": 14, #9
    "cms": 12, #10
    "annotation": 10, #7.5
    "legend": 12, #9
    "tick": 12, #7.5
    "ratio_tick": 12, #7.5
    "title": 7.5,
}
BASE_MARKER_SIZES = {"line": 2, "errorbar": 6}
BASE_LINE_WIDTHS = {"main": 1.8, "secondary": 0.9}
BASE_TICK_SCALE = 0.35
BASE_SPINE_SCALE = 0.35
BASE_TICK_PAD_SCALE = 0.6
BASE_LABEL_PAD_SCALE = 0.6


def get_width_fraction(figure_layout):
    if figure_layout is None:
        return DEFAULT_FIGURE_WIDTH_FRACTION
    if isinstance(figure_layout, (int, float)) and not isinstance(figure_layout, bool):
        fraction = float(figure_layout)
        if fraction <= 0:
            raise ValueError("Width fraction must be positive.")
        return fraction
    layout = str(figure_layout).strip().lower()
    if layout in FIGURE_WIDTHS_FRACTION:
        return FIGURE_WIDTHS_FRACTION[layout]
    if layout.endswith("%"):
        try:
            fraction = float(layout.rstrip("%")) / 100.0
            if fraction <= 0:
                raise ValueError("Width fraction must be positive.")
            return fraction
        except ValueError as exc:
            raise ValueError(f"Invalid width fraction '{figure_layout}'.") from exc
    try:
        fraction = float(layout)
        if fraction <= 0:
            raise ValueError("Width fraction must be positive.")
        return fraction
    except ValueError as exc:
        raise ValueError(
            f"Unknown figure layout '{figure_layout}'. Provide a fraction (e.g. 0.45) or one of "
            f"{sorted(FIGURE_WIDTHS_FRACTION)}."
        ) from exc


def get_width_in(figure_layout):
    return get_width_fraction(figure_layout) * TEXTWIDTH_IN


def scale_style(style, *, scale=1.0, font_scale=1.0, marker_scale=1.0, line_scale=1.0):
    """Return a scaled copy of a style dict.

    scale applies to all numeric sizes (fonts/markers/lines/ticks/pads/spines).
    Use font_scale/marker_scale/line_scale to tweak those groups independently.
    """
    new_style = deepcopy(style)
    font_factor = scale * font_scale
    marker_factor = scale * marker_scale
    line_factor = scale * line_scale
    new_style["fonts"] = {k: v * font_factor for k, v in new_style["fonts"].items()}
    new_style["markers"] = {k: v * marker_factor for k, v in new_style["markers"].items()}
    new_style["lines"] = {k: v * line_factor for k, v in new_style["lines"].items()}
    new_style["tick_scale"] = new_style["tick_scale"] * scale
    new_style["spine_scale"] = new_style["spine_scale"] * scale
    if new_style.get("tick_pad") is not None:
        new_style["tick_pad"] = new_style["tick_pad"] * scale
    if new_style.get("label_pad") is not None:
        new_style["label_pad"] = new_style["label_pad"] * scale
    return new_style

def _get_plot_style(figure_layout=DEFAULT_FIGURE_LAYOUT, ratio=False, aspect=None):
    width_fraction = get_width_fraction(figure_layout)
    width = width_fraction * TEXTWIDTH_IN
    width_scale = width_fraction / DEFAULT_FIGURE_WIDTH_FRACTION
    base_aspect = DEFAULT_ASPECT if aspect is None else aspect
    height = width * base_aspect
    if ratio:
        height *= RATIO_HEIGHT_SCALE
    label_pad = plt.rcParams["axes.labelpad"] * BASE_LABEL_PAD_SCALE * width_scale
    tick_pad = plt.rcParams["xtick.major.pad"] * BASE_TICK_PAD_SCALE * width_scale
    return {
        "figsize": (width, height),
        "fonts": deepcopy(BASE_FONT_SIZES),
        "markers": deepcopy(BASE_MARKER_SIZES),
        "lines": deepcopy(BASE_LINE_WIDTHS),
        "tick_scale": BASE_TICK_SCALE * width_scale,
        "spine_scale": BASE_SPINE_SCALE * width_scale,
        "tick_pad": tick_pad,
        "label_pad": label_pad,
        "width_fraction": width_fraction,
    }


def _apply_tick_style(ax, style, labelsize):
    tick_scale = style["tick_scale"]
    spine_scale = style["spine_scale"]
    tick_pad = style.get("tick_pad")
    label_pad = style.get("label_pad")

    if tick_scale == 1.0:
        if tick_pad is None:
            ax.tick_params(labelsize=labelsize)
        else:
            ax.tick_params(labelsize=labelsize, pad=tick_pad)
    else:
        major_len = plt.rcParams["xtick.major.size"] * tick_scale
        minor_len = plt.rcParams["xtick.minor.size"] * tick_scale
        major_w = plt.rcParams["xtick.major.width"] * tick_scale
        minor_w = plt.rcParams["xtick.minor.width"] * tick_scale
        major_kwargs = {
            "which": "major",
            "length": major_len,
            "width": major_w,
            "labelsize": labelsize,
        }
        if tick_pad is not None:
            major_kwargs["pad"] = tick_pad
        ax.tick_params(**major_kwargs)
        ax.tick_params(
            which="minor",
            length=minor_len,
            width=minor_w,
        )

    if label_pad is not None:
        ax.xaxis.labelpad = label_pad
        ax.yaxis.labelpad = label_pad

    if spine_scale != 1.0:
        base_width = plt.rcParams["axes.linewidth"]
        for spine in ax.spines.values():
            spine.set_linewidth(base_width * spine_scale)


def _set_axis_label(ax, text, fontsize, style, axis):
    label_pad = style.get("label_pad")
    kwargs = {"fontsize": fontsize}
    if label_pad is not None:
        kwargs["labelpad"] = label_pad
    if axis == "x":
        ax.set_xlabel(text, **kwargs)
    else:
        ax.set_ylabel(text, **kwargs)

def finalize_layout(fig, figure_layout, *, pad=None, w_pad=None, h_pad=None, rect=None):
    def _tight_layout_call(pad_value, w_pad_value, h_pad_value, rect_value):
        kwargs = {}
        if pad_value is not None:
            kwargs["pad"] = pad_value
        if w_pad_value is not None:
            kwargs["w_pad"] = w_pad_value
        if h_pad_value is not None:
            kwargs["h_pad"] = h_pad_value
        if rect_value is not None:
            kwargs["rect"] = rect_value
        fig.tight_layout(**kwargs)

    try:
        get_width_fraction(figure_layout)
        _tight_layout_call(
            0.1 if pad is None else pad,
            0.0 if w_pad is None else w_pad,
            0.0 if h_pad is None else h_pad,
            (0, 0, 1, 1) if rect is None else rect,
        )
        return
    except ValueError:
        pass
    print("Warning: Unknown figure_layout '{figure_layout}'. Using default tight_layout.")
    _tight_layout_call(pad, w_pad, h_pad, rect)


def get_categorical_colors(n: int) -> list[str]:
    """Return Petroff palette with at least ``n`` colors."""

    if n <= len(PETROFF_COLORS_6):
        return PETROFF_COLORS_6[:n]
    if n <= len(PETROFF_COLORS_10):
        return PETROFF_COLORS_10[:n]

    palette = PETROFF_COLORS_10 + TAB_COLORS
    if n <= len(palette):
        return palette[:n]

    repeats = (n + len(palette) - 1) // len(palette)
    return (palette * repeats)[:n]


def _poisson_errors(counts, zero_lower_mask=None):
    counts = np.asarray(counts, dtype=float)
    lo, hi = poisson_interval(counts, counts, coverage=0.682689492)
    err_low = counts - lo
    err_high = hi - counts
    err_low = np.nan_to_num(err_low, nan=0.0, posinf=0.0, neginf=0.0)
    err_high = np.nan_to_num(err_high, nan=0.0, posinf=0.0, neginf=0.0)
    if zero_lower_mask is not None:
        err_low = np.where(zero_lower_mask, 0.0, err_low)
    return err_low, err_high


def _fmt_lumi(lumi):
    """Format lumi to one decimal (string) for mplhep label.

    Accepts float/int/np.floating or a preformatted string. Returns None
    unchanged if lumi is None.
    """
    if lumi is None:
        return None
    try:
        val = float(lumi)
        return f"{val:.1f}"
    except Exception:
        return lumi


def _cms_loc_params(loc):
    if loc == 1:
        return 0.98, 0.98, "right", "top"
    if loc == 2:
        return 0.02, 0.02, "left", "bottom"
    if loc == 3:
        return 0.98, 0.02, "right", "bottom"
    return 0.001, 1.015, "left", "baseline"


def add_cms_label(
    ax,
    *,
    label="Private Work",
    data=False,
    com=False,
    lumi=None,
    loc=0,
    fontsize=None,
    style="mplhep",
    custom_text=None,
    rlabel=None,
    llabel=None,
    data_label=None,
    y_offset=0.0,
):
    """Add a CMS label using mplhep or a plain italic text alternative."""
    if style == "mplhep":
        com_value = None if com is False else com
        lumi_value = _fmt_lumi(lumi)
        rlabel_value = rlabel
        if com is False:
            # Suppress the COM label while still allowing a lumi-only right label.
            com_value = None
            if rlabel_value is None and lumi_value is not None:
                rlabel_value = f"{lumi_value} fb$^{{-1}}$"
            elif rlabel_value is None and lumi_value is None:
                rlabel_value = ""
            lumi_value = None
        before = len(ax.texts)
        hep.cms.label(
            data=data,
            ax=ax,
            loc=loc,
            label=label,
            com=com_value,
            lumi=lumi_value,
            fontsize=fontsize,
            rlabel=rlabel_value,
            llabel=llabel,
        )
        if y_offset:
            for text in ax.texts[before:]:
                x, y = text.get_position()
                text.set_position((x, y + y_offset))
        return
    if style != "plain":
        raise ValueError(f"Unknown CMS label style '{style}'. Use 'mplhep' or 'plain'.")

    def _format_com(com_value):
        if com_value is None or com_value is False:
            return None
        if isinstance(com_value, bool):
            return None
        if isinstance(com_value, str):
            return com_value
        try:
            return f"{float(com_value):g}"
        except Exception:
            return str(com_value)

    text = custom_text if custom_text is not None else (llabel if llabel is not None else label)
    use_custom_text = False
    if custom_text is None and llabel is None:
        if data:
            if data_label is not None:
                text = data_label
            else:
                text = "Private Work (CMS Data/Simulation)"
            use_custom_text = True
        else:
            text = "Private Work (CMS Simulation)"
    right_text = None
    if rlabel is not None:
        right_text = rlabel
    else:
        parts = []
        lumi_text = _fmt_lumi(lumi)
        if lumi_text is not None:
            parts.append(f"{lumi_text} fb$^{{-1}}$")
        com_text = _format_com(com)
        if com_text is not None:
            parts.append(f"({com_text} TeV)")
        if parts:
            right_text = " ".join(parts)
    if right_text is None:
        right_text = ""

    plain_fontsize = fontsize
    if use_custom_text and fontsize is not None:
        plain_fontsize = fontsize * 0.85 if "Data/Simulation" in text else fontsize * 0.9
    if plain_fontsize is not None:
        plain_fontsize = plain_fontsize - 1.5

    x, y, ha, va = _cms_loc_params(loc)
    y += y_offset
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=plain_fontsize,
        ha=ha,
        va=va,
    )
    if right_text:
        ax.text(
            1.0,
            y,
            right_text,
            transform=ax.transAxes,
            fontsize=plain_fontsize,
            ha="right",
            va=va,
        )

    return


def plot_histograms(
    hists,
    process_labels,
    output_filename = "./plot.pdf", 
    axis_labels=("x-axis","Events"), 
    normalize=False,
    linestyle="solid",
    log=False,
    include_flow=False,
    CMSlabel="Private Work",
    data=False,
    lumi=None,
    return_figure=False,
    ax = None,
    colors=None,
    alpha=1.0,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
):

    style = _get_plot_style(figure_layout, ratio=False)
    fonts = style["fonts"]

    if include_flow:
        hists = [include_overflow_underflow(hist) for hist in hists]
    if normalize:
        integrals = [_hist.sum().value for _hist in hists]
        hists = [_hist / integral for _hist, integral in zip(hists, integrals)]
    binning = hists[0].to_numpy()[1]
    values = [_hist.values() for _hist in hists]
    uncertainties = [np.sqrt(_hist.variances()) for _hist in hists]

    n_processes = len(process_labels)
    if colors is None:
        colors_used = get_categorical_colors(n_processes)
    else:
        colors_used = list(colors)
        if len(colors_used) < n_processes:
            repeats = (n_processes + len(colors_used) - 1) // len(colors_used)
            colors_used = (colors_used * repeats)[:n_processes]

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"])
    else:
        # meant to be used with fig created outside of this function
        fig = None
    hep.histplot(
        values,
        label=process_labels,
        bins=binning,
        linewidth=1.5,
        yerr=uncertainties,
        ax=ax,
        linestyle=linestyle,
        color=colors_used,
        alpha=alpha,
    )

    ax.margins(y=0.15)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.15*ax.get_ylim()[1])
    _set_axis_label(ax, axis_labels[0], fonts["label"], style, axis="x")
    _set_axis_label(ax, axis_labels[1], fonts["label"], style, axis="y")
    _apply_tick_style(ax, style, fonts["tick"])

    # adjust legend line width
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
        handle.get_children()[0].set_linewidth(3)  # Line
        handle.get_children()[1].set_linewidth(3)  # Caplines
        # handle.get_children()[2].set_linewidth(3)  # Bars
        new_handles.append(handle)
    ncols = 1 if len(hists) < 4 else 2
    ax.legend(handles=new_handles, labels=labels, loc="upper right", fontsize=fonts["legend"], ncols=ncols)#, handlelength=3)

    if CMSlabel is not None:
        hep.cms.label(
            data=data,
            ax=ax,
            loc=0,
            label=CMSlabel,
            com=13.6,
            lumi=_fmt_lumi(lumi),
            fontsize=fonts["cms"],
        )

    # in case the figure should be modified later on, use return_figure=True
    if not return_figure:
        if not os.path.exists(output_filename.replace(output_filename.split("/")[-1], "")):
            os.makedirs(output_filename.replace(output_filename.split("/")[-1], ""))
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close()
        return
    else:
        return fig, ax


def plot_histograms_with_ratio(
    hists_num,
    hists_den,
    labels_num,
    labels_den,
    output_filename="./plot_with_ratio.pdf",
    axis_labels=("x-axis", "Events"),
    ratio_label="Numerator / Denominator",
    normalize=False,
    linestyles=("solid", "dashed"),
    log=False,
    include_flow=False,
    CMSlabel="Private Work",
    data=False,
    lumi=None,
    return_figure=False,
    colors=None,
    ax=None,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
):
    """
    Plots multiple pairs of histograms (numerator vs denominator) with their ratios in a lower panel.
    """
    style = _get_plot_style(figure_layout, ratio=True)
    fonts = style["fonts"]

    if include_flow:
        hists_num = [include_overflow_underflow(hist) for hist in hists_num]
        hists_den = [include_overflow_underflow(hist) for hist in hists_den]

    if normalize:
        integrals_num = [_hist.sum().value for _hist in hists_num]
        hists_num = [
            _hist / integral if integral != 0 else _hist
            for _hist, integral in zip(hists_num, integrals_num)
        ]
        integrals_den = [_hist.sum().value for _hist in hists_den]
        hists_den = [
            _hist / integral if integral != 0 else _hist
            for _hist, integral in zip(hists_den, integrals_den)
        ]

    bin_edges = hists_num[0].axes[0].edges

    if ax is None:
        fig, (ax, ax_ratio) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=style["figsize"],
            gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
        )
    else:
        fig = None
        ax_ratio = None  # In case ax is provided, ax_ratio should be handled separately

    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    # Initialize lists to store lines and labels for the legend
    lines = []
    labels = []

    # Plot histograms
    for idx, (hist_num, hist_den, label_num, label_den) in enumerate(
        zip(hists_num, hists_den, labels_num, labels_den)
    ):
        color = colors[idx % len(colors)]

        # Plot numerator histogram
        lines_num = hep.histplot(
            [hist_num.values()],
            bins=bin_edges,
            yerr=[np.sqrt(hist_num.variances())],
            label=label_num,
            ax=ax,
            linestyle=linestyles[0],
            linewidth=3,
            color=color,
        )
        # Collect the line and label for the legend
        lines.append(lines_num[0])
        labels.append(label_num)

        # Plot denominator histogram
        lines_den = hep.histplot(
            [hist_den.values()],
            bins=bin_edges,
            yerr=[np.sqrt(hist_den.variances())],
            label=label_den,
            ax=ax,
            linestyle=linestyles[1],
            linewidth=3,
            color=color,
        )
        # Collect the line and label for the legend
        lines.append(lines_den[0])
        labels.append(label_den)

    ax.margins(y=0.15)
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.13 * ax.get_ylim()[1])

    if log:
        ax.set_yscale("log")
    _set_axis_label(ax, axis_labels[1], fonts["label"], style, axis="y")
    _apply_tick_style(ax, style, fonts["tick"])

    # Adjust legend line width
    handles, labels_legend = ax.get_legend_handles_labels()
    new_handles = []
    for handle in handles:
        handle.get_children()[0].set_linewidth(3)  # Line
        handle.get_children()[1].set_linewidth(3)  # Caplines
        new_handles.append(handle)
    ncols = 1 if len(hists_num) == 1 else 2
    ax.legend(
        handles=new_handles,
        labels=labels_legend,
        loc="upper right",
        fontsize=fonts["legend"],
        ncols=ncols,
    )

    hep.cms.label(
        data=data,
        ax=ax,
        loc=0,
        label=CMSlabel,
        com=13.6,
        lumi=_fmt_lumi(lumi),
        fontsize=fonts["cms"],
    )

    # Calculate bin centers and bin widths
    bin_edges = hists_num[0].axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]  # Assuming equal bin widths

    # Determine offsets for ratios
    n_ratios = len(hists_num)
    if n_ratios == 1:
        offsets = [0.0]
    elif n_ratios == 2:
        offsets = [-0.1 * bin_width, 0.1 * bin_width]
    else:
        offsets = np.linspace(-0.1 * bin_width, 0.1 * bin_width, n_ratios)

    # Plot ratios
    for idx, (hist_num, hist_den) in enumerate(zip(hists_num, hists_den)):
        values_num = hist_num.values()
        values_den = hist_den.values()
        variances_num = hist_num.variances()
        variances_den = hist_den.variances()

        # Avoid division by zero
        mask = (values_den > 0) & (values_num > 0)
        ratio = np.ones_like(values_num)
        ratio_unc = np.zeros_like(values_num)

        ratio[mask] = values_num[mask] / values_den[mask]
        ratio_unc[mask] = ratio[mask] * np.sqrt(
            (np.sqrt(variances_num[mask]) / values_num[mask]) ** 2
            + (np.sqrt(variances_den[mask]) / values_den[mask]) ** 2
        )

        color = colors[idx % len(colors)]

        # Apply offset
        offset = offsets[idx]
        # Adjust bin edges for the ratio plot
        ratio_bin_edges = bin_edges + offset

        # Ensure ratio_bin_edges has the correct length
        if len(ratio_bin_edges) != len(values_num) + 1:
            ratio_bin_edges = ratio_bin_edges[: len(values_num) + 1]

        # Plot ratio using hep.histplot
        hep.histplot(
            ratio,
            bins=ratio_bin_edges,
            yerr=ratio_unc,
            ax=ax_ratio,
            histtype="errorbar",
            elinewidth=style["lines"]["secondary"],
            markersize=style["markers"]["errorbar"],
            color=color,
        )

    _set_axis_label(ax_ratio, axis_labels[0], fonts["label"], style, axis="x")
    ax_ratio.set_xlim(ax.get_xlim())
    _set_axis_label(ax_ratio, ratio_label, fonts["ratio_label"], style, axis="y")
    _apply_tick_style(ax_ratio, style, fonts["ratio_tick"])
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1, color="gray", linestyle="--", linewidth=1)

    plt.subplots_adjust(hspace=0.08)

    if not return_figure:
        fig.tight_layout()
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))
        fig.savefig(output_filename)
        plt.close(fig)
    else:
        return fig, (ax, ax_ratio)


def plot_contours_2d(
    contours,
    output_filename,
    xlabel,
    ylabel,
    title=None,
    levels=(2.30, 5.99),
    colors=None,
    linewidth=3,
    show_heatmap=False,
    heatmap_levels=60,
    heatmap_cmap="viridis",
    CMSlabel="Private Work",
    data=False,
    lumi=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    legend=True,
    legend_loc="best",
    legend_fontsize=None,
    legend_kwargs=None,
    bestfit_marker="+",
    bestfit_size=12,
    return_figure=False,
    ax=None,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
):
    """Plot 2D likelihood contours from precomputed scan points.

    Each contour dict must provide: x, y, dnll, label, best_x, best_y.
    If show_heatmap=True and multiple contours are provided, only the first
    contour is used for the heatmap to avoid visual clutter.
    """
    style = _get_plot_style(figure_layout, ratio=False)
    fonts = style["fonts"]

    if colors is None:
        colors = get_categorical_colors(len(contours))

    if ax is None:
        fig, ax = plt.subplots(figsize=style["figsize"], constrained_layout=True)
    else:
        fig = None

    heatmap_done = False
    marker_entries = []
    for idx, contour in enumerate(contours):
        x = np.asarray(contour["x"], dtype=float)
        y = np.asarray(contour["y"], dtype=float)
        dnll = np.asarray(contour["dnll"], dtype=float)

        if show_heatmap and not heatmap_done:
            ax.tricontourf(
                x,
                y,
                dnll,
                levels=heatmap_levels,
                cmap=heatmap_cmap,
                alpha=0.6,
            )
            heatmap_done = True

        color = colors[idx % len(colors)]
        contour_lw = linewidth[idx] if isinstance(linewidth, (list, tuple, np.ndarray)) else linewidth
        is_sm = "(SM)" in str(contour.get("label", ""))
        cs = ax.tricontour(
            x,
            y,
            dnll,
            levels=list(levels),
            colors=[color],
            linewidths=contour_lw,
            alpha=0.8,
        )
        if hasattr(cs, "collections"):
            if cs.collections:
                cs.collections[0].set_linestyle("-")
            if len(cs.collections) > 1:
                cs.collections[1].set_linestyle("--")
        else:
            try:
                cs.set_linestyles(["-", "--"])
            except Exception:
                pass
        # Keep legend/data order untouched but draw SM contours on top.
        try:
            for coll in cs.collections:
                coll.set_zorder(4 if is_sm else 3)
        except Exception:
            pass

        marker_entries.append(
            {
                "label": contour.get("label", ""),
                "x": contour["best_x"],
                "y": contour["best_y"],
                "color": color,
                "is_sm": is_sm,
            }
        )

    # Draw best-fit markers after contour lines.
    for entry in marker_entries:
        ax.plot(
            entry["x"],
            entry["y"],
            bestfit_marker,
            color=entry["color"],
            markersize=bestfit_size,
            markeredgewidth=2,
            alpha=0.8,
            zorder=6 if entry["is_sm"] else 5,
        )

    if title:
        ax.set_title(title, fontsize=fonts["title"])
    _set_axis_label(ax, xlabel, fonts["label"], style, axis="x")
    _set_axis_label(ax, ylabel, fonts["label"], style, axis="y")
    _apply_tick_style(ax, style, fonts["tick"])

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if legend:
        handles = []
        labels = []
        for idx, contour in enumerate(contours):
            color = colors[idx % len(colors)]
            contour_lw = linewidth[idx] if isinstance(linewidth, (list, tuple, np.ndarray)) else linewidth
            handles.append(plt.Line2D([0], [0], color=color, lw=contour_lw, alpha=0.8))
            labels.append(contour["label"])
        resolved_legend_fontsize = legend_fontsize if legend_fontsize is not None else fonts["legend"]
        legend_args = {"fontsize": resolved_legend_fontsize, "loc": legend_loc}
        if legend_kwargs:
            legend_args.update(legend_kwargs)
        ax.legend(handles=handles, labels=labels, **legend_args)

    if CMSlabel is not None:
        hep.cms.label(
            data=data,
            ax=ax,
            loc=0,
            label=CMSlabel,
            com=13.6,
            lumi=_fmt_lumi(lumi),
            fontsize=fonts["cms"],
        )

    if not return_figure:
        out_dir = os.path.dirname(output_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_filename)
        plt.close()
        return
    return fig, ax


def plot_stacked_histograms(
    stacked_hists,
    process_labels,
    output_filename="./plot.pdf",
    axis_labels=("x-axis", "Events"),
    signal_hists=None,
    signal_labels=None,
    data_hist=None,
    data_label="Data",
    normalize=False,
    log=False,
    log_min=None,
    include_flow=False,
    CMSlabel="Private Work",
    lumi=None,
    ratio=False,
    ratio_limits=(0.5, 1.5),
    colors=None,
    signal_colors=None,
    return_figure=False,
    ax=None,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
    cms_label_style="mplhep",
    cms_label_com=13.6,
    cms_label_rlabel=None,
    cms_label_llabel=None,
    cms_label_y_offset=0.0,
):
    """
    Plots stacked histograms for backgrounds and optionally overlays signals and data.
    If 'ratio=True', also draws a lower ratio panel of (Data / stacked backgrounds).
    """
    style = _get_plot_style(figure_layout, ratio=ratio)
    fonts = style["fonts"]

    # -------------------------------------------------------------------------
    # 1) Include overflow if requested
    # -------------------------------------------------------------------------
    if include_flow:
        stacked_hists = [include_overflow_underflow(h) for h in stacked_hists]
        if signal_hists:
            signal_hists = [include_overflow_underflow(h) for h in signal_hists]
        if data_hist:
            data_hist = include_overflow_underflow(data_hist)

    # -------------------------------------------------------------------------
    # 2) Normalization if requested
    # -------------------------------------------------------------------------
    if normalize:
        stack_integral = sum([_hist.sum().value for _hist in stacked_hists])
        stacked_hists = [_hist / stack_integral for _hist in stacked_hists]
        if signal_hists:
            for i, sig in enumerate(signal_hists):
                integral_ = sig.sum().value
                if integral_ > 0:
                    signal_hists[i] = sig / integral_
        if data_hist:
            data_integral = data_hist.sum().value
            if data_integral > 0:
                data_hist = data_hist / data_integral

    # Prepare binning from the first background
    bin_edges = stacked_hists[0].to_numpy()[1]
    # We'll gather the MC values + uncertainties
    mc_values_list = [_hist.values() for _hist in stacked_hists]
    mc_errors_list = [np.sqrt(_hist.variances()) for _hist in stacked_hists]

    # -------------------------------------------------------------------------
    # 3) Setup figure/axes
    #    - If ratio=False, we just do a single-axis plot
    #    - If ratio=True, do a top pad (main) + bottom pad (ratio)
    # -------------------------------------------------------------------------
    if not ratio:
        if ax is None:
            fig, ax_main = plt.subplots(figsize=style["figsize"])
        else:
            # user provided an Axes
            fig = None
            ax_main = ax
        ax_ratio = None
    else:
        # ratio panel
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            gridspec_kw={"height_ratios": [4, 1]},
            sharex=True,
            figsize=style["figsize"]
        )

    # -------------------------------------------------------------------------
    # 4) Draw the stacked backgrounds on the top axis
    # -------------------------------------------------------------------------
    hep.histplot(
        mc_values_list,
        label=process_labels,
        bins=bin_edges,
        stack=True,
        histtype="fill",
        edgecolor="black",
        linewidth=style["lines"]["main"]/3,
        yerr=mc_errors_list,
        ax=ax_main,
        color=colors,
        alpha=0.8,
    )

    # Also add a band for the total MC uncertainty
    mc_total = np.sum(mc_values_list, axis=0)
    mc_total_var = np.sum([err**2 for err in mc_errors_list], axis=0)
    mc_total_err = np.sqrt(mc_total_var)
    hep.histplot(
        mc_total,
        bins=bin_edges,
        histtype="band",
        yerr=mc_total_err,
        ax=ax_main,
        alpha=0.5,
        label=None,  # or "Stat. unc." if you want a legend entry
        hatch="///////"
    )

    # -------------------------------------------------------------------------
    # 5) Overlay signal histograms (if any)
    # -------------------------------------------------------------------------
    if signal_hists:
        for idx, (sig_hist, label) in enumerate(zip(signal_hists, signal_labels)):
            sig_values = sig_hist.values()
            sig_errors = np.sqrt(sig_hist.variances())
            color_kwargs = {}
            if signal_colors and idx < len(signal_colors):
                color_kwargs["color"] = signal_colors[idx]
            hep.histplot(
                [sig_values],
                label=[label],
                bins=bin_edges,
                linewidth=style["lines"]["main"],
                linestyle="--",
                yerr=sig_errors,
                ax=ax_main,
                alpha=1.0,
                **color_kwargs,
            )

    # -------------------------------------------------------------------------
    # 6) Plot data (if provided)
    # -------------------------------------------------------------------------
    if data_hist:
        data_values = data_hist.values()
        hep.histplot(
            data_values,
            bins=bin_edges,
            yerr=True,
            color="black",
            label=data_label,
            histtype="errorbar",
            markersize=style["markers"]["errorbar"],
            elinewidth=style["lines"]["secondary"],
            marker=".",
            ax=ax_main,
        )

    # -------------------------------------------------------------------------
    # 7) Final styling of top axis
    # -------------------------------------------------------------------------
    ax_main.margins(y=0.15)
    if log:
        ax_main.set_yscale("log")
        # Expand upper limit in log scale
        ax_main.set_ylim(ax_main.get_ylim()[0], 30 * ax_main.get_ylim()[1])
        if log_min is not None:
            ax_main.set_ylim(log_min, ax_main.get_ylim()[1])
    else:
        ax_main.set_ylim(0, 1.28 * ax_main.get_ylim()[1])

    _set_axis_label(ax_main, axis_labels[1], fonts["label"], style, axis="y")
    _apply_tick_style(ax_main, style, fonts["tick"])
    ax_main.set_xlim(bin_edges[0], bin_edges[-1])
    # Title or legend
    handles, labels = ax_main.get_legend_handles_labels()
    ncols = 2 if len(labels) < 6 else 3
    # if len(labels) > 9: ncols = 4
    # ax_main.legend(loc="upper right", fontsize=fonts["legend"], ncols=ncols, labelspacing=0.4, columnspacing=1.5)
    ax_main.legend(
        loc="upper right",
        fontsize=fonts["legend"],
        ncols=ncols,
        labelspacing=0.05,
        columnspacing=0.45,
        handlelength=1.3,
        handletextpad=0.3,
        borderpad=0.,
        markerscale=1,
        handleheight=0.35
    )


    # Place the CMS label
    add_cms_label(
        ax_main,
        label=CMSlabel,
        data=(data_hist is not None),
        com=cms_label_com,
        lumi=lumi,
        loc=0,
        fontsize=fonts["cms"],
        style=cms_label_style,
        rlabel=cms_label_rlabel,
        llabel=cms_label_llabel,
        y_offset=cms_label_y_offset,
    )

    # -------------------------------------------------------------------------
    # 8) If ratio=True, make the ratio plot
    # -------------------------------------------------------------------------
    if ratio and data_hist:
        # Sum up the backgrounds
        # (We've already done mc_total, mc_total_err up above.)
        data_vals = data_hist.values()
        data_errs = np.sqrt(data_hist.variances())
        eps = 1e-12

        # ratio = data / total MC
        ratio_vals = np.divide(data_vals, mc_total, out=np.zeros_like(data_vals), where=(mc_total > eps))

        # data error in ratio => (sqrt(N_data)) / N_MC
        # but we must do it carefully: ratio_err_data = data_err / MC_value
        ratio_err_data = np.divide(data_errs, mc_total, out=np.zeros_like(data_errs), where=(mc_total > eps))

        # MC relative errors => mc_total_err / mc_total
        rel_mc_errors = np.divide(mc_total_err, mc_total, out=np.zeros_like(mc_total_err), where=(mc_total > eps))
        # Then the ratio band is 1 +/- rel_mc_errors
        lower = 1.0 - rel_mc_errors
        upper = 1.0 + rel_mc_errors

        # x-axis
        ax_ratio.axhline(1.0, linestyle="--", color="black", linewidth=1)
        # plot the band for MC
        ax_ratio.fill_between(
            bin_edges,
            np.concatenate([lower, [lower[-1]]]),   # "step" fill approach
            np.concatenate([upper, [upper[-1]]]),
            step="post",
            facecolor="none",
            edgecolor="tab:gray",
            hatch="XXX",
            alpha=0.3,
            linewidth=0,
            label="MC unc.",
        )

        # Plot the data ratio
        hep.histplot(
            ratio_vals,
            bins=bin_edges,
            histtype="errorbar",
            yerr=ratio_err_data,
            color="black",
            markersize=style["markers"]["errorbar"],
            elinewidth=style["lines"]["secondary"],
            marker=".",
            alpha=1,
            ax=ax_ratio,
        )
        _set_axis_label(ax_ratio, "Data / bkg.", fonts["ratio_label"], style, axis="y")
        ax_ratio.set_ylim(ratio_limits)
        ax_ratio.set_xlim(bin_edges[0], bin_edges[-1])
        _apply_tick_style(ax_ratio, style, fonts["ratio_tick"])
        _set_axis_label(ax_ratio, axis_labels[0], fonts["label"], style, axis="x")

    elif ratio:
        # If ratio=True but we have no data, just leave the ratio panel blank or hide it
        # You could also skip creating the ratio panel entirely if data is None.
        ax_ratio.set_visible(False)
        _set_axis_label(ax_main, axis_labels[0], fonts["label"], style, axis="x")
        ax_main.set_xlim(bin_edges[0], bin_edges[-1])

    else:
        # No ratio => set X label on the main axis
        _set_axis_label(ax_main, axis_labels[0], fonts["label"], style, axis="x")
        ax_main.set_xlim(bin_edges[0], bin_edges[-1])

    # -------------------------------------------------------------------------
    # 9) Save or return figure
    # -------------------------------------------------------------------------
    if not return_figure:
        out_dir = os.path.dirname(output_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig_to_save = fig if fig is not None else ax_main.figure
        finalize_layout(fig_to_save, figure_layout, pad=0.5, rect=(0.03, 0.03, 0.99, 0.99))
        fig_to_save.savefig(output_filename)
        plt.close(fig_to_save)
    else:
        return fig, (ax_main if not ratio else (ax_main, ax_ratio))


def plot_splusb_model(
    bin_edges,
    data_vals,
    bkg_vals,
    sig_vals=None,
    sig_split=None,
    ratio_sig_split=None,
    bands=None,
    data_err=None,
    blinded_region=None,
    unblind=False,
    output_filename="./SplusBModels/plot.pdf",
    split_labels=("ttH+tHW", "tH", "Resonant"),
    cms_label="Work in Progress",
    lumi=61.9,
    title_right=None,
    title_left=None,
    xlim=(100, 180),
    hide_zero_points=False,
    y_label="Events / GeV",
    figure_layout=DEFAULT_FIGURE_LAYOUT,
    bkg_includes_resonant=False,
    show_resonant_legend=False,
    resonant_legend_label="Resonant bkg.",
    cms_label_style="mplhep",
):
    """
    Make an S+B style model plot with a lower panel showing background-subtracted data
    versus the signal model, using mplhep.

    Parameters:
      bin_edges: 1D array-like of bin edges (length N+1)
      data_vals: 1D array-like of data counts per bin (length N)
      bkg_vals : 1D array-like of background expectation per bin (length N)
      sig_vals : 1D array-like of signal expectation per bin (length N). Ignored if sig_split is provided.
      sig_split: Optional tuple/list of arrays (top, thq, resonant), each length N. "resonant" may be None.
      bands    : Optional dict with keys ["median","up1","down1","up2","down2"], each length N, for total model (S+B)
      data_err : Optional 1D array-like of symmetric data errors per bin. If None, Poisson (Garwood) errors are used.
      blinded_region: Optional tuple (xmin, xmax). If provided and unblind=False, only show signal between these bounds on the top panel.
      unblind  : If True show S+B and B lines; otherwise show B line and filled signal in blinded region only.
      output_filename: Path to save the figure (pdf/png)
      split_labels: Legend labels to use when sig_split is provided: (top, thq, resonant)
      cms_label: Text for CMS label (e.g. "Preliminary" or "Private Work")
      lumi     : Luminosity text (fb^-1)
      title_right: Optional right-hand title text (e.g. category label)
    """

    style = _get_plot_style(figure_layout, ratio=True)
    fonts = style["fonts"]

    # Prepare arrays
    data_vals = np.asarray(data_vals, dtype=float)
    bkg_vals = np.asarray(bkg_vals, dtype=float)
    if sig_split is not None:
        sig_top, sig_thq, sig_res = sig_split
        sig_top = None if sig_top is None else np.asarray(sig_top, dtype=float)
        sig_thq = None if sig_thq is None else np.asarray(sig_thq, dtype=float)
        sig_res = None if sig_res is None else np.asarray(sig_res, dtype=float)
    else:
        sig_vals = np.zeros_like(data_vals) if sig_vals is None else np.asarray(sig_vals, dtype=float)

    # Build figure: main + ratio-like panel (background-subtracted)
    fig, (ax, ax_ratio) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=style["figsize"],
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )

    # 1) Optional toy bands (total model S+B)
    if bands is not None:
        # Expect arrays of length N
        for k in ["median", "up1", "down1", "up2", "down2"]:
            if k not in bands:
                raise ValueError(f"bands dict missing key '{k}'")
        med = np.asarray(bands["median"])
        up1 = np.asarray(bands["up1"])
        dn1 = np.asarray(bands["down1"])
        up2 = np.asarray(bands["up2"])
        dn2 = np.asarray(bands["down2"])
        target_len = len(bin_edges) - 1
        for arr, name in [(med, "median"), (up1, "up1"), (dn1, "down1"), (up2, "up2"), (dn2, "down2")]:
            if len(arr) != target_len:
                raise ValueError(f"bands array '{name}' has length {len(arr)}, expected {target_len}")

        # Colors: 1σ = green (#607641), 2σ = yellow (#F5BB54)
        col_1sigma = "#607641"
        col_2sigma = "#F5BB54"

        # Draw as filled bands using step='post'
        ax.fill_between(
            bin_edges,
            np.r_[dn2, dn2[-1]],
            np.r_[up2, up2[-1]],
            step="post",
            color=col_2sigma,
            alpha=0.8,
            linewidth=0,
            label=r"Bkg. $\pm 2\,\sigma$",
        )
        ax.fill_between(
            bin_edges,
            np.r_[dn1, dn1[-1]],
            np.r_[up1, up1[-1]],
            step="post",
            color=col_1sigma,
            alpha=0.8,
            linewidth=0,
            label=r"Bkg. $\pm 1\,\sigma$",
        )

    # 2) Draw background and signal on main axis
    # Background line (continuum + resonant if available)
    res_for_bkg = None
    if not bkg_includes_resonant:
        if sig_split is not None and len(sig_split) == 3:
            res_for_bkg = sig_split[2]
        if ratio_sig_split is not None and len(ratio_sig_split) == 3:
            res_for_bkg = ratio_sig_split[2]
    res_for_bkg = np.asarray(res_for_bkg, dtype=float) if res_for_bkg is not None else 0.0
    bkg_line = bkg_vals + res_for_bkg
    if not unblind:
        hep.histplot(
            bkg_line,
            bins=bin_edges,
            ax=ax,
            linewidth=style["lines"]["secondary"],
            color="tab:red",
            linestyle=":",
            label=r"Bkg. (cont. + $H\rightarrow\gamma\gamma$)",
        )

    # Signal drawing
    if unblind:
        # Draw dotted total background (continuum + resonant) and solid S+B (unscaled signal)
        if ratio_sig_split is not None:
            top_line, thq_line, res_line = ratio_sig_split
        else:
            top_line, thq_line, res_line = sig_top, sig_thq, sig_res
        if bkg_includes_resonant:
            res_line = None
        bkg_line = bkg_vals + (res_line if res_line is not None else 0)
        splusb_line = bkg_line.copy()
        if sig_split is not None:
            if top_line is not None:
                splusb_line = splusb_line + top_line
            if thq_line is not None:
                splusb_line = splusb_line + thq_line
        else:
            splusb_line = splusb_line + sig_vals
        hep.histplot(
            bkg_line,
            bins=bin_edges,
            ax=ax,
            linewidth=style["lines"]["secondary"],
            color="tab:red",
            linestyle=":",
            label=r"Bkg. (cont. + $H\rightarrow\gamma\gamma$)",
        )
        hep.histplot(
            splusb_line,
            bins=bin_edges,
            ax=ax,
            linewidth=style["lines"]["secondary"],
            color="tab:red",
            linestyle="-",
            label="S+B fit",
        )
        # Overlay scaled signal components as filled shapes (like blinded view)
        if sig_split is not None:
            arrays = []
            labels = []
            colors = []
            if sig_top is not None:
                arrays.append(sig_top)
                labels.append(split_labels[0] if len(split_labels) > 0 else "ttH+tHW")
                colors.append("C0")
            if sig_thq is not None:
                arrays.append(sig_thq)
                labels.append(split_labels[1] if len(split_labels) > 1 else "tH")
                colors.append("C3")
            if arrays:
                hep.histplot(
                    arrays,
                    bins=bin_edges,
                    ax=ax,
                    histtype="fill",
                    stack=True,
                    linewidth=0,
                    color=colors,
                    label=labels,
                    alpha=0.8,
                )
    else:
        # draw filled signal only in blinded region (if provided). Use stacking order: resonant (gray), top (blue), thq (green)
        # Also draw total S+B as a solid red line (continuum+resonant+signal).
        splusb_line = bkg_line.copy()
        if sig_split is not None:
            if ratio_sig_split is not None:
                top_line, thq_line, _ = ratio_sig_split
            else:
                top_line, thq_line = sig_top, sig_thq
            if top_line is not None:
                splusb_line = splusb_line + top_line
            if thq_line is not None:
                splusb_line = splusb_line + thq_line
        else:
            splusb_line = splusb_line + sig_vals
        hep.histplot(
            splusb_line,
            bins=bin_edges,
            ax=ax,
            linewidth=style["lines"]["secondary"],
            color="tab:red",
            linestyle="-",
            label="S+B fit",
        )
        if sig_split is not None:
            arrays = []
            labels = []
            colors = []
            # Stack only ttH+tHW and tHq
            if sig_top is not None:
                arrays.append(sig_top)
                labels.append(split_labels[0] if len(split_labels) > 0 else "ttH+tHW")
                colors.append("C0")
            # tHq on top
            if sig_thq is not None:
                arrays.append(sig_thq)
                labels.append(split_labels[1] if len(split_labels) > 1 else "tH")
                colors.append("C3")

            if arrays:
                hep.histplot(
                    arrays,
                    bins=bin_edges,
                    ax=ax,
                    histtype="fill",
                    stack=True,
                    linewidth=0,
                    color=colors,
                    label=labels,
                    alpha=0.8,
                )
            # (removed separate resonant line to keep old convention)
        else:
            hep.histplot(
                [sig_vals],
                bins=bin_edges,
                ax=ax,
                linewidth=0,
                histtype="fill",
                color="tab:blue",
                label="S model",
            )

    # Data points with errors (mask blinded region if needed)
    def _mask_blind(arr):
        if unblind or blinded_region is None:
            mask = np.ones_like(arr, dtype=bool)
            return np.array(arr, dtype=float), mask
        xcent = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = (xcent < blinded_region[0]) | (xcent > blinded_region[1])
        out = np.array(arr, dtype=float)
        out[~mask] = np.nan
        return out, mask

    data_main, visible_mask = _mask_blind(data_vals)
    zero_mask_main = data_vals == 0
    data_main[zero_mask_main] = np.nan
    visible_mask = visible_mask & (~zero_mask_main)
    if data_err is not None:
        err_low_main = np.asarray(data_err, dtype=float)
        err_high_main = np.asarray(data_err, dtype=float)
    else:
        err_low_main, err_high_main = _poisson_errors(data_vals)
        err_low_main = err_low_main.astype(float)
        err_high_main = err_high_main.astype(float)
    err_low_main[~visible_mask] = np.nan
    err_high_main[~visible_mask] = np.nan
    hep.histplot(
        data_main,
        bins=bin_edges,
        ax=ax,
        histtype="errorbar",
        yerr=(err_low_main, err_high_main),
        color="black",
        label="Data",
        markersize=style["markers"]["errorbar"],
        elinewidth=style["lines"]["secondary"],
        marker=".",
    )

    # Style main
    ax.margins(y=0.15)
    ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
    _set_axis_label(ax, y_label, fonts["label"], style, axis="y")
    ax.set_xlabel("")
    ax.set_xticklabels([])
    _apply_tick_style(ax, style, fonts["tick"])
    # Reorder legend into requested columns:
    # left: bkg, bkg ±1σ, bkg ±2σ
    # middle: S+B fit, Data
    # right: ttH+tHW, tH
    handles, labels_display = ax.get_legend_handles_labels()

    def _pop_match(pred):
        for i, lbl in enumerate(labels_display):
            if pred(lbl):
                return handles.pop(i), labels_display.pop(i)
        return None, None

    def _is_bkg(lbl: str) -> bool:
        return lbl.startswith("Bkg.") and "cont." in lbl

    def _is_pm1(lbl: str) -> bool:
        return "$\\pm 1" in lbl

    def _is_pm2(lbl: str) -> bool:
        return "$\\pm 2" in lbl

    def _is_splusb(lbl: str) -> bool:
        return lbl == "S+B fit"

    def _is_data(lbl: str) -> bool:
        return lbl == "Data"

    def _is_tth(lbl: str) -> bool:
        return ("tHW" in lbl) or ("t\\bar{t}H" in lbl) or ("ttH" in lbl)

    def _is_th(lbl: str) -> bool:
        return ("tH" in lbl) and ("tHW" not in lbl) and ("ttH" not in lbl) and ("\\bar{t}" not in lbl)

    col1 = []
    col2 = []
    col3 = []
    for pred, target in [
        (_is_bkg, col1),
        (_is_pm1, col1),
        (_is_pm2, col1),
        (_is_splusb, col2),
        (_is_data, col2),
        (_is_tth, col3),
        (_is_th, col3),
    ]:
        h, l = _pop_match(pred)
        if h is not None:
            target.append((h, l))

    # Append any remaining handles to the right-most column (keep order)
    leftovers = list(zip(handles, labels_display))
    if leftovers:
        col3.extend(leftovers)

    # Matplotlib fills legend entries column-wise. Pad columns to equal length
    # so entries align as desired.
    from matplotlib.lines import Line2D

    def _pad_column(col, rows):
        while len(col) < rows:
            col.append((Line2D([], [], linestyle="none", marker="", alpha=0), ""))
        return col

    max_rows = max(len(col1), len(col2), len(col3), 0)
    col1 = _pad_column(col1, max_rows)
    col2 = _pad_column(col2, max_rows)
    col3 = _pad_column(col3, max_rows)

    handles_ordered = [h for h, _ in (col1 + col2 + col3)]
    labels_ordered = [l for _, l in (col1 + col2 + col3)]

    if show_resonant_legend and resonant_legend_label and resonant_legend_label not in labels_ordered:
        from matplotlib.lines import Line2D

        handles_ordered.append(
            Line2D([0], [0], color="gray", linestyle="--", linewidth=style["lines"]["secondary"])
        )
        labels_ordered.append(resonant_legend_label)

    ncols = 3
    ax.legend(
        handles_ordered,
        labels_ordered,
        loc="upper right",
        fontsize=fonts["legend"] - 1.0,
        ncols=ncols,
        labelspacing=0.2,
        columnspacing=0.7,
        handletextpad=0.4,
        borderpad=0.2,
    )

    add_cms_label(
        ax,
        label=cms_label,
        data=True,
        com=13.6,
        lumi=lumi,
        fontsize=fonts["cms"],
        style=cms_label_style,
    )
    if title_right:
        ax.text(0.94, 0.73, title_right, ha="right", va="center", transform=ax.transAxes, fontsize=fonts["legend"] - 1.0)
    if title_left:
        ax.text(0.14, 0.86, title_left, ha="left", va="center", transform=ax.transAxes, fontsize=fonts["legend"] - 1.0)

    # 3) Ratio-like panel: (Data - B) vs Signal (subtract resonant if present)
    res_for_ratio = None
    if not bkg_includes_resonant:
        if ratio_sig_split is not None and len(ratio_sig_split) == 3:
            res_for_ratio = ratio_sig_split[2]
        elif sig_split is not None and len(sig_split) == 3:
            res_for_ratio = sig_split[2]
    res_for_ratio = np.asarray(res_for_ratio, dtype=float) if res_for_ratio is not None else 0.0
    data_minus_b = data_vals - bkg_vals - res_for_ratio
    # Build "band" for ratio panel if provided
    if bands is not None:
        med = np.asarray(bands["median"]) - bkg_vals
        up1 = np.asarray(bands["up1"]) - bkg_vals
        dn1 = np.asarray(bands["down1"]) - bkg_vals
        up2 = np.asarray(bands["up2"]) - bkg_vals
        dn2 = np.asarray(bands["down2"]) - bkg_vals
        # Also subtract resonant from the band to avoid a bump
        res_unscaled = None
        if ratio_sig_split is not None and len(ratio_sig_split) == 3:
            res_unscaled = ratio_sig_split[2]
        elif sig_split is not None and len(sig_split) == 3:
            res_unscaled = sig_split[2]
        # Subtract resonant component so the bands are centered around 0 in the ratio panel,
        # unless the background already includes resonant.
        if (not bkg_includes_resonant) and res_unscaled is not None:
            med = med - res_unscaled
            up1 = up1 - res_unscaled
            dn1 = dn1 - res_unscaled
            up2 = up2 - res_unscaled
            dn2 = dn2 - res_unscaled
        col_1sigma = "#607641"
        col_2sigma = "#F5BB54"
        ax_ratio.fill_between(
            bin_edges,
            np.r_[dn2, dn2[-1]],
            np.r_[up2, up2[-1]],
            step="post",
            color=col_2sigma,
            alpha=0.8,
            linewidth=0,
        )
        ax_ratio.fill_between(
            bin_edges,
            np.r_[dn1, dn1[-1]],
            np.r_[up1, up1[-1]],
            step="post",
            color=col_1sigma,
            alpha=0.8,
            linewidth=0,
        )

    # Draw signal expectation on ratio panel (prefer unscaled if provided); do not draw resonant
    if ratio_sig_split is not None:
        top_u, thq_u, res_u = ratio_sig_split
        arrays = []
        colors = []
        if top_u is not None:
            arrays.append(top_u)
            colors.append("C0")
        if thq_u is not None:
            arrays.append(thq_u)
            colors.append("C3")
        if arrays:
            hep.histplot(
                arrays,
                bins=bin_edges,
                ax=ax_ratio,
                histtype="fill",
                stack=True,
                linewidth=0,
                color=colors,
                alpha=0.8,
            )
    elif sig_split is not None:
        arrays = []
        colors = []
        if sig_top is not None:
            arrays.append(sig_top)
            colors.append("C0")
        if sig_thq is not None:
            arrays.append(sig_thq)
            colors.append("C3")
        if arrays:
            hep.histplot(
                arrays,
                bins=bin_edges,
                ax=ax_ratio,
                histtype="fill",
                stack=True,
                linewidth=0,
                color=colors,
                alpha=0.8,
            )
    else:
        hep.histplot(sig_vals, bins=bin_edges, ax=ax_ratio, linewidth=3, color="tab:blue")

    # Data minus background with errors
    data_ratio, mask_ratio = _mask_blind(data_minus_b)
    zero_mask_sub = data_vals == 0
    if hide_zero_points:
        data_ratio[zero_mask_sub] = np.nan
        mask_ratio = mask_ratio & (~zero_mask_sub)
    zero_mask_sub = (data_vals == 0)
    if data_err is not None:
        err_low_ratio = np.asarray(data_err, dtype=float)
        err_high_ratio = np.asarray(data_err, dtype=float)
    else:
        err_low_ratio, err_high_ratio = _poisson_errors(data_vals, zero_lower_mask=zero_mask_sub)
        err_low_ratio = err_low_ratio.astype(float)
        err_high_ratio = err_high_ratio.astype(float)
    err_low_ratio[~mask_ratio] = np.nan
    err_high_ratio[~mask_ratio] = np.nan
    hep.histplot(
        data_ratio,
        bins=bin_edges,
        ax=ax_ratio,
        histtype="errorbar",
        yerr=(err_low_ratio, err_high_ratio),
        color="black",
        markersize=style["markers"]["errorbar"],
        elinewidth=style["lines"]["secondary"],
        marker=".",
    )
    _set_axis_label(ax_ratio, "Data - bkg.", fonts["ratio_label"], style, axis="y")
    _set_axis_label(ax_ratio, r"Diphoton mass [GeV]", fonts["label"], style, axis="x")
    _apply_tick_style(ax_ratio, style, fonts["ratio_tick"])
    if xlim is not None:
        ax.set_xlim(*xlim)
        ax_ratio.set_xlim(*xlim)
    # Autoscale y from data only (mask blinded region), for stable limits
    valid = ~np.isnan(data_ratio)
    if np.any(valid):
        ymin = np.nanmin(data_ratio[valid])
        ymax = np.nanmax(data_ratio[valid])
    else:
        ymin, ymax = 0.0, 1.0
    if hide_zero_points:
        spread = np.nanmax(np.abs(data_ratio[valid])) if np.any(valid) else 1.0
        spread = max(spread, 1.5)
        ax_ratio.set_ylim(-spread, spread)
    else:
        span = ymax - ymin
        ax_ratio.set_ylim(ymin - 0.15 * span, ymax + 0.25 * span)

    # Finalize
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.07, left=0.16, bottom=0.12)
    fig.savefig(output_filename)
    plt.close(fig)


def include_overflow_underflow(hist):

    # do not change in place:
    new_hist = deepcopy(hist)
    bin_contents = new_hist.view(flow=True)
    overflow, underflow = bin_contents[-1], bin_contents[0]
    n_bins = len(bin_contents) - 2 
    # print(new_hist.view(flow=True))
    new_hist[0] += underflow
    new_hist[n_bins-1] += overflow
    # print(new_hist.view(flow=True))
    return new_hist


def get_dict_hists(Zee=False):

    dict_histograms = {
        "mass": [Hist.new.Reg(50, 100, 150).Weight(), (r"Diphoton mass [GeV]", "Fraction of events")],
        "massH": [Hist.new.Reg(32, 100, 180).Weight(), (r"Diphoton mass [GeV]", "Fraction of events")] if not Zee else [Hist.new.Reg(40, 60, 180).Weight(), (r"Diphoton mass [GeV]", "Fraction of events")],
        "topMt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Top quark $m_T$ [GeV]", "Fraction of events")],
        "ptH": [Hist.new.Reg(20, 0, 350).Weight(), (r"Diphoton $p_T$ [GeV]", "Fraction of events")],
        "ptOverM_H": [Hist.new.Reg(25, 0, 4).Weight(), (r"Diphoton $p_T / m$", "Fraction of events")],
        "yH": [Hist.new.Reg(20, -3, 3).Weight(), (r"Diphoton $y$", "Fraction of events")],

        "nJets": [Hist.new.Reg(13, -0.5, 12.5).Weight(), (r"N(jets)", "Fraction of events")],
        "nJetsCentral": [Hist.new.Reg(11, -0.5, 10.5).Weight(), (r"N(central jets)", "Fraction of events")],
        "nJetsForward": [Hist.new.Reg(7, -0.5, 6.5).Weight(), (r"N(forward jets)", "Fraction of events")],
        "nBjetsCentral": [Hist.new.Reg(5, -0.5, 4.5).Weight(), (r"N(b-tags)", "Fraction of events")],
        "nBtags": [Hist.new.Reg(5, -0.5, 4.5).Weight(), (r"N(b-tags)", "Fraction of events")],

        "bJetsEta": [Hist.new.Reg(20, 0., 4.7).Weight(), (r"b-jets $\eta$", "Fraction of events")],
        "leading_bJetPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"b-jet $p_T$ [GeV]", "Fraction of events")],
        "leading_bJetEta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"b-jet $\eta$", "Fraction of events")],
        "bJetPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"b-jet $p_T$ [GeV]", "Fraction of events")] if not Zee else [Hist.new.Reg(60, 0, 300).Weight(), (r"b-jet $p_T$ [GeV]", "Fraction of events")],
        "bJetEta": [Hist.new.Reg(50, -2.5, 2.5).Weight(), (r"b-jet $\eta$", "Fraction of events")],
        "bJetMass": [Hist.new.Reg(20, 0, 50).Weight(), (r"b-jet mass", "Fraction of events")],
        "bJetBTag": [Hist.new.Reg(25, 0., 1).Weight(), (r"b-jet b-tag", "Fraction of events")],

        "jetLeadPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Leading jet $p_T$ [GeV]", "Fraction of events")] if not Zee else [Hist.new.Reg(60, 0, 300).Weight(), (r"Leading jet $p_T$ [GeV]", "Fraction of events")],
        "jetLeadEta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Leading jet $\eta$", "Fraction of events")],
        "jetLeadMass": [Hist.new.Reg(20, 0, 70).Weight(), (r"Leading jet mass", "Fraction of events")],
        "jetLeadBTag": [Hist.new.Reg(25, -0.2, 1).Weight(), (r"Leading jet b-tag", "Fraction of events")],

        "jet3Pt": [Hist.new.Reg(20, 0, 250).Weight(), (r"Jet 3 $p_T$ [GeV]", "Fraction of events")] if not Zee else [Hist.new.Reg(50, 0, 250).Weight(), (r"Jet 3 $p_T$ [GeV]", "Fraction of events")],
        "jet3Eta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Jet 3 $\eta$", "Fraction of events")],
        "jet3Mass": [Hist.new.Reg(20, 0, 50).Weight(), (r"Jet 3 mass", "Fraction of events")],
        "jet3BTag": [Hist.new.Reg(25, -0.2, 1).Weight(), (r"Jet 3 b-tag", "Fraction of events")],
        "jet4Pt": [Hist.new.Reg(20, 0, 150).Weight(), (r"Jet 4 $p_T$ [GeV]", "Fraction of events")],
        "jet4Eta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Jet 4 $\eta$", "Fraction of events")],
        "jet4Mass": [Hist.new.Reg(20, 0, 50).Weight(), (r"Jet 4 mass", "Fraction of events")],
        "jet4BTag": [Hist.new.Reg(25, -0.2, 1).Weight(), (r"Jet 4 b-tag", "Fraction of events")],
        "jet5Pt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Jet 5 $p_T$ [GeV]", "Fraction of events")],
        "jet5Eta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Jet 5 $\eta$", "Fraction of events")],
        "jet5Mass": [Hist.new.Reg(20, 0, 50).Weight(), (r"Jet 5 mass", "Fraction of events")],
        "jet5BTag": [Hist.new.Reg(25, -0.2, 1).Weight(), (r"Jet 5 b-tag", "Fraction of events")],
        "jet6Pt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Jet 6 $p_T$ [GeV]", "Fraction of events")],
        "jet6Eta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Jet 6 $\eta$", "Fraction of events")],
        "jet6Mass": [Hist.new.Reg(20, 0, 50).Weight(), (r"Jet 6 mass", "Fraction of events")],
        "jet6BTag": [Hist.new.Reg(25, -0.2, 1).Weight(), (r"Jet 6 b-tag", "Fraction of events")],

        "leadingJetPt": [Hist.new.Reg(20, 0, 300).Weight(), (r"Leading jet $p_T$ [GeV]", "Fraction of events")],
        "leadingJetEta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Leading jet $\eta$", "Fraction of events")],

        "subleadingJetPt": [Hist.new.Reg(20, 0, 200).Weight(), (r"Subleading jet $p_T$ [GeV]", "Fraction of events")],
        "subleadingJetEta": [Hist.new.Reg(50, -4.7, 4.7).Weight(), (r"Subleading jet $\eta$", "Fraction of events")], 

        "nLeptons": [Hist.new.Reg(4, -0.5, 3.5).Weight(), (r"N(leptons)", "Fraction of events")],
        "leptonPt": [Hist.new.Reg(20, 0, 200).Weight(), (r"Lepton $p_T$ [GeV]", "Fraction of events")],
        "leptonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Lepton $\eta$", "Fraction of events")],
        "leptonCharge": [Hist.new.Reg(3, -1.5, 1.5).Weight(), (r"Lepton charge", "Fraction of events")],
        "leptonMvaTTH": [Hist.new.Reg(25, -1, 1).Weight(), (r"Lepton MVA-$(t\bar tH)$", "Fraction of events")],
        "leptonGenPartFlav": [Hist.new.Reg(44, 0, 25).Weight(), (r"Lepton genPartFlav", "Fraction of events")],
        "mass_lgg": [Hist.new.Reg(30, 100, 400).Weight(), (r"$m(\ell,\gamma\gamma)$ [GeV]", "Fraction of events")],
        "mass_l_phoLead": [Hist.new.Reg(30, 10, 190).Weight(), (r"$m(\ell, \gamma 1)$ [GeV]", "Fraction of events")],
        "mass_l_phoSublead": [Hist.new.Reg(30, 10, 160).Weight(), (r"$m(\ell, \gamma 2)$ [GeV]", "Fraction of events")],

        "leadingPhotonPt": [Hist.new.Reg(28, 25, 305).Weight(), (r"Leading photon $p_T$ [GeV]", "Fraction of events")],
        "leadingPhotonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Leading photon $\eta$", "Fraction of events")],
        "leadingPhotonPhi": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Leading photon $\phi$", "Fraction of events")],
        "subleadingPhotonPt": [Hist.new.Reg(26, 20, 150).Weight(), (r"Subleading photon $p_T$ [GeV]", "Fraction of events")],
        "subleadingPhotonEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Subleading photon $\eta$", "Fraction of events")],
        "subleadingPhotonPhi": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Subleading photon $\phi$", "Fraction of events")],
        "phoLeadPt": [Hist.new.Reg(27, 35, 305).Weight(), (r"Leading photon $p_T$ [GeV]", "Fraction of events")],
        "phoLeadPtOverM_H": [Hist.new.Reg(25, 0.35, 2.5).Weight(), (r"Subleading photon $p_T / m(\gamma\gamma)$", "Fraction of events")],
        "phoLeadEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Leading photon $\eta$", "Fraction of events")],
        "phoLeadGenPartFlav": [Hist.new.Reg(20, 0, 11).Weight(), (r"Leading photon genPartFlav", "Fraction of events")],
        "phoLeadMVAID": [Hist.new.Reg(20, -1, 1).Weight(), (r"Leading photon MVA ID", "Fraction of events")],
        "phoLeadPixelSeed": [Hist.new.Reg(2, -0.5, 1.5).Weight(), (r"Leading photon pixel seed", "Fraction of events")],
        "phoSubleadPt": [Hist.new.Reg(26, 25, 150).Weight(), (r"Subleading photon $p_T$ [GeV]", "Fraction of events")],
        "phoSubleadPtOverM_H": [Hist.new.Reg(25, 0.25, 1.25).Weight(), (r"Subleading photon $p_T / m(\gamma\gamma)$", "Fraction of events")],
        "phoSubleadEta": [Hist.new.Reg(20, -2.5, 2.5).Weight(), (r"Subleading photon $\eta$", "Fraction of events")],
        "phoSubleadGenPartFlav": [Hist.new.Reg(20, 0, 11).Weight(), (r"Subleading photon genPartFlav", "Fraction of events")],
        "phoSubleadMVAID": [Hist.new.Reg(20, -1, 1).Weight(), (r"Subleading photon MVA ID", "Fraction of events")],
        "phoMaxMVAID": [Hist.new.Reg(20, -1, 1).Weight(), (r"Maximum photon MVA ID", "Fraction of events")],
        "phoMinMVAID": [Hist.new.Reg(20, -1, 1).Weight(), (r"Minimum photon MVA ID", "Fraction of events")],
        "phoSubleadPixelSeed": [Hist.new.Reg(2, -0.5, 1.5).Weight(), (r"Subleading photon pixel seed", "Fraction of events")],
        "cosThetaStar": [Hist.new.Reg(20, -1, 1).Weight(), (r"$\cos(\theta^{*})$", "Fraction of events")],

        "met": [Hist.new.Reg(20, 0, 200).Weight(), (r"MET $p_T$ [GeV]", "Fraction of events")],
        "metSignificance": [Hist.new.Reg(20, 0, 250).Weight(), (r"MET significance", "Fraction of events")],

        "rapiditySumJH": [Hist.new.Reg(20, 0, 6).Weight(), (r"$y(jet 1) \oplus y(\gamma\gamma)$", "Fraction of events")],
        "rapidityDiffJH": [Hist.new.Reg(20, -6, 6).Weight(), (r"$\Delta y(j, \gamma\gamma)$", "Fraction of events")],
        "DeltaY_Hb": [Hist.new.Reg(20, -5, 5).Weight(), (r"$\Delta y(\gamma\gamma, b)$", "Fraction of events")],
        "DeltaY_Hl": [Hist.new.Reg(20, -5, 5).Weight(), (r"$\Delta y(\gamma\gamma, l)$", "Fraction of events")],
        "delta_eta_b_j": [Hist.new.Reg(20, -7, 7).Weight(), (r"$\Delta \eta(b,j)$", "Fraction of events")],
        "delta_r_jet_pho": [Hist.new.Reg(60, 0, 8).Weight(), (r"$\Delta R(j, \gamma)$", "Fraction of events")],

        "delta_r_b_j": [Hist.new.Reg(20, 0, 7).Weight(), (r"$\Delta R(b,j)$", "Fraction of events")],
        "delta_r_j_h": [Hist.new.Reg(20, 0, 7).Weight(), (r"$\Delta R(j,\gamma\gamma)$", "Fraction of events")],
        "delta_r_h_b": [Hist.new.Reg(20, 0, 7).Weight(), (r"$\Delta R(\gamma\gamma, b)$", "Fraction of events")],
        "delta_r_h_l": [Hist.new.Reg(20, 0, 7).Weight(), (r"$\Delta R(\gamma\gamma, \ell)$", "Fraction of events")],
        "delta_r_G_G": [Hist.new.Reg(20, 0, 4).Weight(), (r"$\Delta R(\gamma\gamma)$", "Fraction of events")],

        "NN_output": [Hist.new.Reg(20, 0, 1).Weight(), ("Network output", "Fraction of events")],
        "ttH_vs_tH_NN": [Hist.new.Reg(20, 0, 1).Weight(), (r"$t\bar{t}H$ vs. $tH$ NN output", "Fraction of events")],
        "sig_vs_bkg_NN": [Hist.new.Reg(20, 0, 1).Weight(), ("Signal vs. background NN output", "Fraction of events")],
        "sig_vs_bkg_NN_ttH_had": [Hist.new.Reg(20, 0, 1).Weight(), (r"Sig. vs. bkg. NN output, $t\bar t H$ (had)", "Fraction of events")],
        "sig_vs_bkg_NN_ttH_lep": [Hist.new.Reg(20, 0, 1).Weight(), (r"Sig. vs. bkg. NN output, $t\bar t H$ (lep)", "Fraction of events")],
        "sig_vs_bkg_NN_tH_had": [Hist.new.Reg(20, 0, 1).Weight(), (r"Sig. vs. bkg. NN output, $tH$ (had)", "Fraction of events")],
        "sig_vs_bkg_NN_tH_lep": [Hist.new.Reg(20, 0, 1).Weight(), (r"Sig. vs. bkg. NN output, $tH$ (lep)", "Fraction of events")],
    }

    def _update_hist_label_local(feature_name, x_label):
        if feature_name in dict_histograms:
            hist_cfg = dict_histograms[feature_name]
            y_label = hist_cfg[1][1]
            dict_histograms[feature_name][1] = (x_label, y_label)

    for prefix, jet_idx in JET_NUMBER_MAP.items():
        _update_hist_label_local(f"{prefix}Pt", rf"$p_T(jet {jet_idx})$ [GeV]")
        _update_hist_label_local(f"{prefix}Eta", rf"$\eta(jet {jet_idx})$")
        _update_hist_label_local(f"{prefix}Phi", rf"$\phi(jet {jet_idx})$")
        _update_hist_label_local(f"{prefix}Mass", rf"$m(jet {jet_idx})$ [GeV]")
        _update_hist_label_local(f"{prefix}BTag", rf"$\mathrm{{b\!-\!tag}}(jet {jet_idx})$")

        if prefix != "jetLead":
            original_idx = prefix[3:]
            _update_hist_label_local(
                f"rapiditySumJet{original_idx}H",
                rf"$y^{{jet 1{jet_idx}}} \oplus y^{{\gamma\gamma}}$",
            )
            _update_hist_label_local(
                f"delta_r_h_jet{original_idx}",
                rf"$\Delta R(\gamma\gamma, jet {jet_idx})$",
            )

    _update_hist_label_local("rapiditySumJH", "$y^{j_1} \\oplus y^{\\gamma\\gamma}$")
    _update_hist_label_local("rapidityDiffJH", "$$\\Delta y(j_1, \\gamma\\gamma)$")
    _update_hist_label_local("delta_eta_b_j", "$\\Delta \\eta(b, j_1)$")
    _update_hist_label_local("delta_r_j_h", "$\\Delta R(j_1, \\gamma\\gamma)$")
    _update_hist_label_local("delta_r_b_j", "$\\Delta R(b, j_1)$")

    return dict_histograms

tex_feature_names = {
    'ptH': r'$\mathit{p}_T(\mathit{\gamma\gamma})$', # r'$p_{T}(\gamma\gamma)$', 
    'ptOverM_H': r'$\mathit{p}_T(\mathit{\gamma\gamma})\,/\, \mathit{m(\gamma\gamma)}$',
    'yH': '$\mathit{y}(\mathit{\gamma\gamma})$',
    'phiH': '$\mathit{\phi}(\mathit{\gamma\gamma})$',
    'nJets': '$\mathit{N}$(jets)',
    'nLeptons': "$\mathit{N}$(leptons)",
    'nJetsCentral': "$\mathit{N}$(central jets)",
    'nBjetsCentral': "$\mathit{N}$($\mathit{b}$-tags)",
    'nJetsForward': "$\mathit{N}$(forward jets)",
    'metPt': "$\mathit{E}_T^{miss}$",
    'metPhi': r"$\mathit{\phi}\left(\mathit{E}_T^{miss}\right)$",
    'metSignificance': "$\mathit{E}_T^{miss}$ significance",
    'topMt': "$\mathit{m}_T(\mathit{t})$",
    'delta_eta_b_j': "$\Delta \eta(b-jet, jet 1)$",
    'rapiditySumJH': "$y^{jet 1} \oplus y^{\gamma\gamma}$",
    'rapidityDiffJH': "$\Delta y(jet 1, \gamma\gamma)$",
    'DeltaY_Hb': "$\Delta \mathit{y}(\mathit{\gamma\gamma, b})$",
    'DeltaY_Hl': "$\Delta \mathit{y}(\mathit{\gamma\gamma, \ell})$",
    "jetLeadPt": "$\mathit{p}_T(jet 1)$",
    "jetLeadEta": "$\mathit{\eta}(jet 1)$",
    "jetLeadPhi": "$\mathit{\phi}(jet 1)$",
    "bJetPt": "$\mathit{p}_T(\mathit{b})$",
    "bJetEta": "$\mathit{\eta}(\mathit{b})$",
    "bJetPhi": "$\mathit{\phi}(\mathit{b})$",
    "leptonPt": "$\mathit{p}_T(\mathit{\ell})$",
    "leptonEta": "$\mathit{\eta}(\mathit{\ell})$",
    "leptonPhi": "$\mathit{\phi}(\mathit{\ell})$",
    "leptonGeneration": "Lepton flavor",
    "delta_r_b_j": r"$\Delta R(b, jet 1)$",
    "delta_r_j_h": r"$\Delta R(jet 1,\gamma\gamma)$",
    "delta_r_h_b": r"$\Delta R(\gamma\gamma, b)$",
    "delta_r_h_l": r"$\Delta R(\gamma\gamma, \ell)$",
}

JET_LATEX_LABELS = [
    "b-jet",
    "jet 1",
    "jet 2",
    "jet 3",
    "jet 4",
    "jet 5",
]

for idx, (i, j) in enumerate(combinations(range(len(JET_LATEX_LABELS)), 2), start=1):
    jet_label = ", ".join([JET_LATEX_LABELS[i], JET_LATEX_LABELS[j]])
    tex_feature_names.setdefault(f"mass_pair_{idx}", rf"$m({jet_label})$")
    tex_feature_names.setdefault(f"deltaR_pair_{idx}", rf"$\Delta R({jet_label})$")

for idx, combo in enumerate(combinations(range(len(JET_LATEX_LABELS)), 3), start=1):
    jet_label = ", ".join(JET_LATEX_LABELS[k] for k in combo)
    tex_feature_names.setdefault(f"mass_triplet_{idx}", rf"$m({jet_label})$")

# legend for pairs of DR and mass:
# deltaR_pair_1 -> DR(b-jet, jet 1)
# deltaR_pair_2 -> DR(b-jet, jet 2)
# deltaR_pair_3 -> DR(b-jet, jet 3)
# deltaR_pair_4 -> DR(b-jet, jet 4)
# deltaR_pair_5 -> DR(b-jet, jet 5)
# deltaR_pair_6 -> DR(jet 1, jet 2)
# deltaR_pair_7 -> DR(jet 1, jet 3)
# deltaR_pair_8 -> DR(jet 1, jet 4)
# deltaR_pair_9 -> DR(jet 1, jet 5)
# deltaR_pair_10 -> DR(jet 2, jet 3)
# deltaR_pair_11 -> DR(jet 2, jet 4)
# deltaR_pair_12 -> DR(jet 2, jet 5)
# deltaR_pair_13 -> DR(jet 3, jet 4)
# deltaR_pair_14 -> DR(jet 3, jet 5)
# deltaR_pair_15 -> DR(jet 4, jet 5)

FEATURE_GROUP_COLORS = {
    "event": PETROFF_COLORS_10[0],
    "higgs": PETROFF_COLORS_10[1],
    "photon": PETROFF_COLORS_10[2],
    "lepton": PETROFF_COLORS_10[3],
    "bjet": PETROFF_COLORS_10[4],
    "jet": PETROFF_COLORS_10[5],
    "jet_pair_mass": PETROFF_COLORS_10[6],
    "jet_pair_dr": PETROFF_COLORS_10[7],
    "jet_triplet_mass": PETROFF_COLORS_10[8],
    "other": PETROFF_COLORS_10[9],
}

FEATURE_GROUP_LABELS = {
    "event": "Event-level",
    "higgs": "Diphoton system",
    "photon": "Photons",
    "lepton": "Lepton",
    "bjet": "b-jet",
    "jet": "Other jets",
    "jet_pair_mass": "Jet-pair masses",
    "jet_pair_dr": "Jet-pair $\\Delta R$",
    "jet_triplet_mass": "Jet-triplet masses",
    "other": "Other high-level",
}

FEATURE_GROUP_OVERRIDES = {
    "delta_eta_b_j": "other",
    "rapiditysumjh": "other",
    "rapiditydiffjh": "other",
    "deltay_hb": "other",
    "deltay_hl": "other",
    "delta_r_h_l": "other",
    "delta_r_h_b": "other",
    "delta_r_b_j": "other",
    "delta_r_g_g": "photon",
    "mass_lgg": "higgs",
    "mass_l_pholead": "photon",
    "mass_l_phosublead": "photon",
    "leptoncharge": "lepton",
    "leptongeneration": "lepton",
    "leptonmvatth": "lepton",
    "leptonmvatthhad": "lepton",
    "leptonmvatthlep": "lepton",
}

EVENT_FEATURE_PREFIXES = ("n", "met", "nbjet", "tt", "sig_vs_bkg")
EVENT_FEATURE_NAMES = {"topmt", "rapiditysumjh", "rapiditydiffjh"}
HIGGS_FEATURE_NAMES = {"pth", "ptoverm_h", "yh", "phih", "massh"}

def _determine_feature_group(feature_name: str) -> str:
    lname = feature_name.lower()
    if lname in FEATURE_GROUP_OVERRIDES:
        return FEATURE_GROUP_OVERRIDES[lname]
    if lname.startswith("mass_pair_"):
        return "jet_pair_mass"
    if lname.startswith("deltar_pair_"):
        return "jet_pair_dr"
    if lname.startswith("mass_triplet_"):
        return "jet_triplet_mass"
    if any(lname.startswith(prefix) for prefix in EVENT_FEATURE_PREFIXES) or lname in EVENT_FEATURE_NAMES:
        return "event"
    if lname in HIGGS_FEATURE_NAMES:
        return "higgs"
    if "pho" in lname or "gamma" in lname:
        return "photon"
    if "lepton" in lname:
        return "lepton"
    if lname.startswith("bjet") or "bjet" in lname:
        return "bjet"
    if lname.startswith("jet") or lname.startswith("rapiditysumjet") or lname.startswith("delta_r_h_jet"):
        return "jet"
    return "other"

# --- Canonical jet labels for individual features ------------------------------
JET_NUMBER_MAP = {
    "jetLead": 1,
    "jet3": 2,
    "jet4": 3,
    "jet5": 4,
    "jet6": 5,
}

for prefix, jet_idx in JET_NUMBER_MAP.items():
    tex_feature_names[f"{prefix}Pt"] = rf"$\mathit{{p}}_T(jet {jet_idx})$"
    tex_feature_names[f"{prefix}Eta"] = rf"$\mathit{{\eta}}(jet {jet_idx})$"
    tex_feature_names[f"{prefix}Phi"] = rf"$\mathit{{\phi}}(jet {jet_idx})$"
    tex_feature_names[f"{prefix}Mass"] = rf"$m(jet {jet_idx})$"
    tex_feature_names[f"{prefix}BTag"] = rf"$\mathrm{{b\!-\!tag}}(jet {jet_idx})$"

    if prefix != "jetLead":
        original_idx = prefix[3:]  # '3', '4', ...
        tex_feature_names[f"rapiditySumJet{original_idx}H"] = rf"$y^{{jet {jet_idx}}} \oplus y^{{\gamma\gamma}}$"
        tex_feature_names[f"delta_r_h_jet{original_idx}"] = rf"$\Delta R(\gamma\gamma, jet {jet_idx})$"
def get_process_tex_names_dict():

    label_mapping = {
            'GJet': r'$\gamma+$jet',
            'WG': r'$W\gamma$',
            'ZG': r'$Z\gamma$',
            'VG': r'$V\gamma$',
            'DYG': r'$DY(\rightarrow \ell\ell)\gamma$',
            'DY': r'$DY(\rightarrow \ell\ell)$',
            'DYto2L': r'$DY(\rightarrow \ell\ell)$',
            'Diboson': r"$VV$",
            'SingleTop': r"$tq$",
            'TTbar': r"$t\bar{t}$",
            'TTbar_noPresel': r"$t\bar{t}\ (jj\rightarrow\gamma\gamma)$",
            'TTGamma': r"$t\bar{t}\gamma$",
            'TTLL': r"$t\bar{t}\ell\ell$",
            'TTG_TQG': r"$t\bar{t}\gamma + tq\gamma$",
            'TTGG_TJGG': r"$t\bar{t}\gamma\gamma + tq\gamma\gamma$",
            'TTW_TW': r"$t\bar{t}W + tW$",
            'TZQB': 'tZq',
            'QCD': 'Multijet',
            'TTGG': r"$t\bar{t}\gamma\gamma$",
            'GG-Box': r'$\gamma\gamma$',
            'TJGG': r'$tq\gamma\gamma$',
            'TGQB': r'$t\gamma$',
            "HGG": r"Other $H\to\gamma\gamma$",
            "tHq": fr"$tHq$",
            "tHqCPEven": fr"$tHq$ (SM)",
            "tHqCPOdd": fr"$tHq$ (CP-odd)",
            "tHq_kt_m1": fr"$tHq\ (\kappa_t=-1)$",
            "ttH": r"$t\bar{t}H$",
            "tHW": fr"$tHW$",
            "ttHCPEven": r"$t\bar{t}H$ SM",
            "ttHCPOdd": r"$t\bar{t}H$ CP-odd",
        }

    return label_mapping


tHq_sample_names = {
    "tHq": "$tHq$ SM",
    "tHq_CPodd": "$tHq$ CP-odd",
    "tHq_k-m1_ktilde-0": r"$tHq\ \mathit{\kappa_t}=-1,\ \mathit{\tilde{\kappa}_t}=0$",
    "tHq_k-0_ktilde-m1": r"$tHq\ \mathit{\kappa_t}=0,\ \mathit{\tilde{\kappa}_t}=-1$",
    "tHq_k-0_ktilde-0": r"$tHq\ \mathit{\kappa_t}=0,\ \mathit{\tilde{\kappa}_t}=0$",
    "tHq_k-0p7_ktilde-0p7": r"$tHq\ \mathit{\kappa_t}=\frac{1}{\sqrt{2}},\ \mathit{\tilde{\kappa}_t}=\frac{1}{\sqrt{2}}$",
    "tHq_k-0p7_ktilde-m0p7": r"$tHq\ \mathit{\kappa_t}=\frac{1}{\sqrt{2}},\ \mathit{\tilde{\kappa}_t}=-\frac{1}{\sqrt{2}}$",
}


def plotDataFrame(
    data_dict,
    weights_dict,
    features,
    process_names,
    output_dir,
    linestyles=None,
    colors=None,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
):
    """
    Plots histograms for each feature in the given dataframes for all classes.

    Parameters:
    - data_dict: Dictionary mapping process names to DataFrames containing data.
    - weights_dict: Dictionary mapping process names to numpy arrays of weights.
    - features: List of features to plot.
    - process_names: List of process names corresponding to classes.
    - output_dir: Directory where plots will be saved.
    - linestyles: (optional) List of linestyles for each process, or single string.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    dict_histograms = get_dict_hists()  # Function defined in the script

    # No linestyles: all solid
    if linestyles is None:
        linestyles = ["solid"] * len(process_names)
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * len(process_names)

    # For each feature
    for feature in features:
        print(f"Plotting feature: {feature}")
        # Get the axis labels from tex_feature_names
        x_label = tex_feature_names.get(feature, feature)
        y_label = "Events"

        # Get histogram configuration from dict_histograms if available
        if feature in dict_histograms:
            hist_config, _ = dict_histograms[feature]
            bins = hist_config.axes[0].edges
            lower = bins[0]
            upper = bins[-1]
            n_bins = len(bins) - 1
        else:
            # Determine bins based on quantiles across all data
            all_data = pd.concat([df[feature] for df in data_dict.values()])
            n_bins = 25
            if 'Eta' in feature or 'eta' in feature:
                lower = -5
                upper = 5
                if "jet" in feature.lower():
                    n_bins = 100
            elif 'Phi' in feature or 'phi' in feature:
                lower = -np.pi
                upper = np.pi
            else:
                lower = np.percentile(all_data, 0)
                upper = np.percentile(all_data, 98)

        hists = []
        for process_name in process_names:
            df = data_dict[process_name]
            weights = weights_dict[process_name]
            data = df[feature]
            hist = Hist.new.Reg(n_bins, lower, upper).Weight()
            hist.fill(data.to_numpy(), weight=weights)
            hists.append(hist)

        output_filename = os.path.join(output_dir, f"{feature}.pdf")
        plot_histograms(
            hists=hists,
            process_labels=process_names,
            output_filename=output_filename,
            axis_labels=(x_label, y_label),
            normalize=True,
            include_flow=True,
            linestyle=linestyles if linestyles is not None else "solid",
            log=False,
            CMSlabel="Private Work",
            data=False,
            lumi=None,
            return_figure=False,
            ax=None,
            colors=colors,
            figure_layout=figure_layout,
        )


# how background processes should appear in stack plots
processes_order = [
        'HGG',
        'TTGG',
        'TJGG',
        'TTG_TQG',
        'TTbar',
        'TTbar_noPresel',
        'GJet',
        'QCD',
        # 'WG',
        # 'ZG',
        'WG',
        'DYG',
        'SingleTop',
        'GG-Box',
        'TGQB',
        'TTGamma',
]

processes_order_Zee = [
    'TZQB',
    'TTLL',
    'TTW_TW',
    'Diboson',
    'TTbar',
    'DYto2L',
]

def get_process_colors_new(process):
    cms_colors = PETROFF_COLORS_10
    tab_colors = ["tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:brown", "black", "white"]

    # Combine the palettes
    all_colors = cms_colors + tab_colors

    colours = {
        'QCD': all_colors[0],
        'GJet': all_colors[1],
        'GG-Box': all_colors[2],
        'TTbar': all_colors[3],
        "VG": all_colors[6],


        'HGG': all_colors[11],
        'TTG_TQG': all_colors[7],
        'TTGG_TJGG': all_colors[4],
        'SingleTop': all_colors[9],

        # old scheme before summarizing TTbar and VG
        'TTbar_noPresel': all_colors[4],
        'DYG': all_colors[6],
        'WG': all_colors[5],
        'TGQB': all_colors[12],
        'TJGG': all_colors[4],
        'TTGG': all_colors[7],

        # Zee validation samples:
        'DYto2L': all_colors[0],
        'TTLL': all_colors[1],
        'Diboson': all_colors[2],
        'TZQB': all_colors[4],
        'TTW_TW': all_colors[5],
    }

    return colours.get(process, "k")


def get_ordered_indices(process_names, Zee=False):
    """
    Returns indices to reorder process_names according to processes_order.

    Any process not in processes_order will be placed at the end in their original order.

    Args:
        process_names (list): List of process names to be ordered.

    Returns:
        list: Indices that can be used to reorder process_names.
    """
    # Build a mapping from process name to its order
    order_dict = {name: i for i, name in enumerate(processes_order if not Zee else processes_order_Zee)}
    # Assign a high order value for processes not in processes_order
    max_order = len(processes_order if not Zee else processes_order_Zee)
    # Pair each process with its order and original index
    process_with_order = [
        (i, order_dict.get(name, max_order), name) for i, name in enumerate(process_names)
    ]
    # Now sort based on the order
    sorted_processes = sorted(process_with_order, key=lambda x: x[1])
    # Extract the indices
    ordered_indices = [x[0] for x in sorted_processes]
    return ordered_indices


def get_process_colors(process_names):

    # Get CMS color palette
    cms_colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    tab_colors = ["tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:brown", "black", "white"]

    # Combine the palettes
    all_colors = cms_colors + tab_colors

    # Assign colors to processes
    color_mapping = {}
    for i, process in enumerate(processes_order):
        color_mapping[process] = all_colors[i % len(all_colors)]

    # For any remaining processes, continue assigning colors
    remaining_processes = [proc for proc in process_names if proc not in processes_order]
    start_index = len(processes_order)
    for i, process in enumerate(remaining_processes, start=start_index):
        color_mapping[process] = all_colors[i % len(all_colors)]
    # Now, create the list of colors corresponding to process_names
    colors = [color_mapping.get(proc, 'grey') for proc in process_names]

    return colors


def plot_feature_importance(
    df_class0,
    w_class0,
    df_class1,
    w_class1,
    feats,
    scaler,
    model,
    batch_size=4096,
    n_max=50_000,
    n_repeats=1,
    random_seed=42,
    decorrelation_groups=None,
    jsd_bins=20,
    compute_jsd=False,
    compute_disco=False,
    disco_batch_size=4096,
    decor_sample_limit=None,
    figure_layout=DEFAULT_FIGURE_LAYOUT,
):
    """
    Evaluate permutation-based feature importance as the increase in weighted binary cross-entropy loss.
    
    Parameters:
      df_class0      : pd.DataFrame with data for the first class
      w_class0       : 1D array-like, weights for the first class
      df_class1      : pd.DataFrame with data for the second class
      w_class1       : 1D array-like, weights for the second class
      feats          : list of features to use (must be present in both DataFrames)
      scaler         : fitted scaler (e.g. StandardScaler) used to transform the features
      model          : trained Keras model (expects the scaled features)
      batch_size     : batch size for model.predict (default: 4096)
      n_max          : maximum number of events to use (default: 8192)
      decorrelation_groups: optional dict with keys {'class', 'labels'} describing which class to decorrelate and the per-event labels
      compute_jsd    : if True, evaluate JSD-based permutation importance for the selected class
      compute_disco  : if True, evaluate DisCo-based permutation importance for the selected class
      disco_batch_size: chunk size to use when computing DisCo (keeps memory bounded)
      decor_sample_limit: optional cap on the number of events used for decorrelation metrics (None keeps all)
    
    Returns:
      ((fig_loss, ax_loss),
       (fig_jsd, ax_jsd),
       (fig_disco, ax_disco),
       dict_loss,
       dict_jsd,
       dict_disco)

      JSD / DisCo outputs are None when no decorrelation labels are provided.
    """
    if tf is None or create_histogram is None or jensen_shannon_divergence is None:
        raise ImportError("TensorFlow and train_helper are required for plot_feature_importance.")

    style = _get_plot_style(figure_layout, ratio=False, aspect=BAR_PLOT_ASPECT)
    fonts = style["fonts"]

    print("\nINFO: calculating feature importance.\n")

    # Ensure weights are positive
    w_class0 = np.abs(np.array(w_class0))
    w_class1 = np.abs(np.array(w_class1))

    # --- Apply n_max if specified ---
    decor_class = None
    decor_labels = None
    compute_jsd = bool(compute_jsd) and decorrelation_groups is not None
    compute_disco = bool(compute_disco) and decorrelation_groups is not None

    if n_max is not None:
        # Shuffle class0 data while maintaining correspondence with weights
        class0_indices = np.random.permutation(len(df_class0))
        df_class0 = df_class0.iloc[class0_indices]
        w_class0 = w_class0[class0_indices]

        # Shuffle class1 data while maintaining correspondence with weights
        class1_indices = np.random.permutation(len(df_class1))
        df_class1 = df_class1.iloc[class1_indices]
        w_class1 = w_class1[class1_indices]

        # Handle decorrelation labels if provided
        if decorrelation_groups:
            decor_class = decorrelation_groups.get("class", 1)
            decor_labels = np.asarray(decorrelation_groups.get("labels"))
            if decor_class not in (0, 1):
                raise ValueError("decorrelation_groups['class'] must be 0 or 1.")
            expected = len(df_class0) if decor_class == 0 else len(df_class1)
            if len(decor_labels) != expected:
                raise ValueError("Length of decorrelation labels must match the chosen class.")
            if decor_class == 0:
                decor_labels = decor_labels[class0_indices]
            else:
                decor_labels = decor_labels[class1_indices]

        # Select only the first n_max events (if more than available, take all)
        df_class0 = df_class0.iloc[:n_max]
        w_class0 = w_class0[:n_max]
        df_class1 = df_class1.iloc[:n_max]
        w_class1 = w_class1[:n_max]
        if decorrelation_groups:
            if decor_class == 0:
                decor_labels = decor_labels[:len(df_class0)]
            else:
                decor_labels = decor_labels[:len(df_class1)]
    else:
        if decorrelation_groups:
            decor_class = decorrelation_groups.get("class", 1)
            decor_labels = np.asarray(decorrelation_groups.get("labels"))
        if decor_class not in (0, 1):
            raise ValueError("decorrelation_groups['class'] must be 0 or 1.")
        expected = len(df_class0) if decor_class == 0 else len(df_class1)
        if len(decor_labels) != expected:
            raise ValueError("Length of decorrelation labels must match the chosen class.")

    # Optionally downsample the class used for decorrelation metrics to avoid OOM
    decor_batches = None
    if (compute_jsd or compute_disco) and decor_labels is not None:
        target_df = df_class0 if decor_class == 0 else df_class1
        target_weights = w_class0 if decor_class == 0 else w_class1
        target_weights = np.asarray(target_weights, dtype=np.float32)
        orig_sum = float(np.sum(target_weights))
        if decor_sample_limit is not None and len(target_df) > decor_sample_limit:
            rng_dec = np.random.default_rng(random_seed + 12345)
            sel_indices = np.sort(rng_dec.choice(len(target_df), size=decor_sample_limit, replace=False))
            target_df = target_df.iloc[sel_indices].reset_index(drop=True)
            target_weights = target_weights[sel_indices]
            decor_labels = np.asarray(decor_labels[sel_indices])
            new_sum = np.sum(target_weights)
            if new_sum > 0 and orig_sum > 0:
                target_weights = target_weights * (orig_sum / new_sum)
        else:
            target_df = target_df.reset_index(drop=True)
            target_weights = target_weights.copy()
        if decor_class == 0:
            df_class0 = target_df
            w_class0 = target_weights
        else:
            df_class1 = target_df
            w_class1 = target_weights
        decor_labels = np.asarray(decor_labels)
        if compute_disco:
            total_len = len(decor_labels)
            if total_len == 0:
                compute_disco = False
            elif disco_batch_size is None or disco_batch_size <= 0 or total_len <= disco_batch_size:
                decor_batches = [np.arange(total_len)]
            else:
                indices = np.arange(total_len)
                decor_batches = [indices[i : i + disco_batch_size] for i in range(0, total_len, disco_batch_size)]

    # --- Normalize class1 weights only once ---
    sum_class0 = np.sum(w_class0)
    sum_class1 = np.sum(w_class1)
    if sum_class1 != 0:  # Avoid division by zero
        w_class1 = w_class1 * (sum_class0 / sum_class1)

    # --- Combine class0 and class1 into one dataset ---
    df_all = pd.concat([df_class0, df_class1], ignore_index=True)
    # Create true labels: 1 for class1, 0 for class0
    y_class0 = np.zeros(len(df_class0))
    y_class1 = np.ones(len(df_class1))
    y_all = np.concatenate([y_class0, y_class1])
    # Combine weights (order must match the concatenation order)
    w_all = np.concatenate([w_class0, w_class1])

    # Extract the features (ensure the order of columns matches feats)
    X_all = df_all[feats]

    # --- Compute the baseline loss ---
    X_all_scaled = scaler.transform(X_all)
    preds = model.predict(X_all_scaled, batch_size=batch_size).flatten()
    baseline_loss = log_loss(y_all, preds, sample_weight=w_all)

    def _compute_jsd_metric(predictions, weights, labels, bins=20):
        unique = np.unique(labels)
        if len(unique) < 2:
            return None
        hists = []
        for lab in unique:
            mask = labels == lab
            if not np.any(mask):
                continue
            hist = create_histogram(predictions[mask], weights[mask], bins=bins)
            hists.append(hist.values())
        if len(hists) < 2:
            return None
        jsd_vals = []
        for i in range(len(hists)):
            for j in range(i + 1, len(hists)):
                p = np.array(hists[i], dtype=np.float32)
                q = np.array(hists[j], dtype=np.float32)
                if not np.any(p) or not np.any(q):
                    continue
                # convert to probability vectors
                p = p / np.sum(p)
                q = q / np.sum(q)
                jsd = jensen_shannon_divergence(
                    tf.convert_to_tensor(p, dtype=tf.float32),
                    tf.convert_to_tensor(q, dtype=tf.float32),
                )
                jsd_vals.append(float(jsd.numpy()))
        return float(np.mean(jsd_vals)) if jsd_vals else None

    def _compute_disco_metric(predictions, weights, labels):
        unique = np.unique(labels)
        if len(unique) < 2:
            return None
        if not hasattr(_compute_disco_metric, "_distance_corr"):
            try:
                from .train_helper import distance_corr  # local import to avoid circular dependency
            except ImportError:
                return None
            _compute_disco_metric._distance_corr = distance_corr
        distance_corr = _compute_disco_metric._distance_corr

        batches = decor_batches if decor_batches is not None else [np.arange(len(labels))]
        disco_vals = []
        disco_weights = []
        for batch_indices in batches:
            batch_labels = labels[batch_indices]
            batch_weights = weights[batch_indices]
            batch_preds = predictions[batch_indices]
            unique_batch = np.unique(batch_labels)
            if len(unique_batch) < 2:
                continue
            for i in range(len(unique_batch)):
                for j in range(i + 1, len(unique_batch)):
                    mask_i = batch_labels == unique_batch[i]
                    mask_j = batch_labels == unique_batch[j]
                    if not np.any(mask_i) or not np.any(mask_j):
                        continue
                    preds_i = batch_preds[mask_i]
                    preds_j = batch_preds[mask_j]
                    weights_i = batch_weights[mask_i]
                    weights_j = batch_weights[mask_j]
                    preds_pair = np.concatenate([preds_i, preds_j])
                    labels_pair = np.concatenate(
                        [np.zeros_like(preds_i, dtype=np.float32), np.ones_like(preds_j, dtype=np.float32)]
                    )
                    weights_pair = np.concatenate([weights_i, weights_j]).astype(np.float32)
                    weight_sum = np.sum(weights_pair)
                    if weight_sum <= 0.0:
                        continue
                    weights_tf = tf.convert_to_tensor(weights_pair, dtype=tf.float32)
                    norm = tf.reduce_sum(weights_tf)
                    if norm <= 0:
                        continue
                    weights_tf = weights_tf / norm * tf.cast(tf.size(weights_tf), tf.float32)
                    disco_val = distance_corr(
                        tf.convert_to_tensor(labels_pair, dtype=tf.float32),
                        tf.convert_to_tensor(preds_pair, dtype=tf.float32),
                        weights_tf,
                    )
                    disco_vals.append(float(disco_val.numpy()))
                    disco_weights.append(weight_sum)
        if not disco_vals:
            return None
        disco_vals = np.array(disco_vals)
        disco_weights = np.array(disco_weights)
        if np.sum(disco_weights) <= 0:
            return float(np.mean(disco_vals))
        return float(np.average(disco_vals, weights=disco_weights))

    baseline_jsd = None
    target_slice = None
    target_weights = None
    baseline_disco = None
    if decor_labels is not None and (compute_jsd or compute_disco):
        n0 = len(df_class0)
        n1 = len(df_class1)
        if decor_class == 0:
            target_slice = slice(0, n0)
            target_weights = w_all[:n0]
        else:
            target_slice = slice(n0, n0 + n1)
            target_weights = w_all[n0:]
        if compute_jsd:
            baseline_jsd = _compute_jsd_metric(preds[target_slice], target_weights, decor_labels, bins=jsd_bins)
            if baseline_jsd is None:
                compute_jsd = False
        if compute_disco:
            baseline_disco = _compute_disco_metric(preds[target_slice], target_weights, decor_labels)
            if baseline_disco is None:
                compute_disco = False
        if not compute_jsd and not compute_disco:
            decor_labels = None
            target_slice = None
            target_weights = None

    # --- Permutation importance ---
    rng = np.random.default_rng(random_seed)

    importance_dict = {}
    importance_jsd = {} if compute_jsd else None
    importance_disco = {} if compute_disco else None
    if decor_labels is None:
        importance_jsd = None
        importance_disco = None
    for feat in feats:
        X_perm_scaled = X_all_scaled.copy()
        feat_idx = feats.index(feat)  # Get the column index of the feature
        original_column = X_all_scaled[:, feat_idx].copy()
        perm_losses = []
        perm_jsds = [] if importance_jsd is not None else None
        perm_discos = [] if importance_disco is not None else None
        for _ in range(n_repeats):
            X_perm_scaled[:, feat_idx] = rng.permutation(original_column)
            preds_perm = model.predict(X_perm_scaled, batch_size=batch_size).flatten()
            loss_perm = log_loss(y_all, preds_perm, sample_weight=w_all)
            perm_losses.append(loss_perm)
            if target_slice is not None and (importance_jsd is not None or importance_disco is not None):
                jsd_perm = _compute_jsd_metric(
                    preds_perm[target_slice],
                    target_weights,
                    decor_labels,
                    bins=jsd_bins,
                )
                if importance_jsd is not None:
                    perm_jsds.append(jsd_perm if jsd_perm is not None else baseline_jsd)
                if importance_disco is not None:
                    disco_perm = _compute_disco_metric(
                        preds_perm[target_slice],
                        target_weights,
                        decor_labels,
                    )
                    perm_discos.append(disco_perm if disco_perm is not None else baseline_disco)
        X_perm_scaled[:, feat_idx] = original_column
        importance = np.mean(perm_losses) - baseline_loss
        importance_dict[feat] = importance
        if importance_jsd is not None:
            if perm_jsds:
                jsd_diff = np.mean(perm_jsds) - (baseline_jsd if baseline_jsd is not None else 0.0)
            else:
                jsd_diff = 0.0
            importance_jsd[feat] = jsd_diff
        if importance_disco is not None:
            if perm_discos:
                disco_diff = np.mean(perm_discos) - (baseline_disco if baseline_disco is not None else 0.0)
            else:
                disco_diff = 0.0
            importance_disco[feat] = disco_diff

    # --- Normalize and plot ---
    sorted_features = sorted(importance_dict, key=importance_dict.get, reverse=True)
    importance_values = np.array([importance_dict[feat] for feat in sorted_features])
    # Normalize so that the most important feature has a value of 1
    max_val = np.max(importance_values)
    if max_val == 0:
        importance_values_norm = importance_values
    else:
        importance_values_norm = importance_values / max_val

    # Get feature labels from the plotter dictionary if available
    variable_labels = []
    dict_hists_ = get_dict_hists()
    for feature_name in sorted_features:
        if feature_name in dict_hists_:
            label = dict_hists_[feature_name][1][0].replace("[GeV]", "")
        else:
            label = tex_feature_names.get(feature_name, feature_name)
        variable_labels.append(label)
    print("Sorted features and labels:", sorted_features, variable_labels)
    
    feature_groups = [_determine_feature_group(feat) for feat in sorted_features]
    bar_colors = [
        FEATURE_GROUP_COLORS.get(group, FEATURE_GROUP_COLORS["other"])
        for group in feature_groups
    ]

    # Plot using explicit positions for even spacing and proper alignment
    positions = np.arange(len(variable_labels))
    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.bar(
        positions,
        importance_values_norm,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_xticks(positions)
    base_tick = fonts["tick"]
    if len(variable_labels) > 70:
        fontsize = max(6, base_tick - 4)
    elif len(variable_labels) > 50:
        fontsize = max(7, base_tick - 3)
    elif len(variable_labels) > 40:
        fontsize = max(8, base_tick - 2)
    elif len(variable_labels) > 30:
        fontsize = max(9, base_tick - 1)
    else:
        fontsize = base_tick
    ax.set_xticklabels(variable_labels, rotation=45, ha="right", fontsize=fontsize)
    _set_axis_label(ax, "Permutation feature importance", fonts["label"], style, axis="y")
    ax.set_ylim(ax.get_ylim()[0], 1.1)
    _apply_tick_style(ax, style, fonts["tick"])
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="x", which="minor", bottom=False)
    legend_handles = []
    seen_groups = set()
    for group, color in zip(feature_groups, bar_colors):
        if group not in seen_groups:
            label = FEATURE_GROUP_LABELS.get(group, group.title())
            legend_handles.append(Patch(facecolor=color, edgecolor="black", label=label))
            seen_groups.add(group)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            # title="Feature groups",
            frameon=False,
            loc="upper right",
            ncol=len(legend_handles) if len(legend_handles) < 7 else 4,
            columnspacing=0.8,
            handlelength=1.0,
            fontsize=fonts["legend"],
            title_fontsize=fonts["title"],
        )
    plt.tight_layout()

    fig_jsd = None
    ax_jsd = None
    if importance_jsd is not None:
        jsd_values = np.array([importance_jsd.get(feat, 0.0) for feat in sorted_features])
        max_abs = np.max(np.abs(jsd_values))
        jsd_norm = jsd_values / max_abs if max_abs > 0 else jsd_values
        fig_jsd, ax_jsd = plt.subplots(figsize=style["figsize"])
        ax_jsd.bar(
            positions,
            jsd_norm,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax_jsd.set_xticks(positions)
        ax_jsd.set_xticklabels(variable_labels, rotation=45, ha="right", fontsize=fontsize)
        _set_axis_label(ax_jsd, "Permutation feature importance (JSD)", fonts["label"], style, axis="y")
        ax_jsd.set_ylim(ax_jsd.get_ylim()[0], 1.1)
        _apply_tick_style(ax_jsd, style, fonts["tick"])
        ax_jsd.tick_params(axis="x", which="both", top=False)
        ax_jsd.tick_params(axis="x", which="minor", bottom=False)
        if legend_handles:
            ax_jsd.legend(
                handles=legend_handles,
                frameon=False,
                loc="upper right",
                ncol=len(legend_handles) if len(legend_handles) < 7 else 4,
                columnspacing=0.8,
                handlelength=1.0,
                fontsize=fonts["legend"],
                title_fontsize=fonts["title"],
            )
        plt.tight_layout()

    fig_disco = None
    ax_disco = None
    if importance_disco is not None:
        disco_values = np.array([importance_disco.get(feat, 0.0) for feat in sorted_features])
        max_abs = np.max(np.abs(disco_values))
        disco_norm = disco_values / max_abs if max_abs > 0 else disco_values
        fig_disco, ax_disco = plt.subplots(figsize=style["figsize"])
        ax_disco.bar(
            positions,
            disco_norm,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax_disco.set_xticks(positions)
        ax_disco.set_xticklabels(variable_labels, rotation=45, ha="right", fontsize=fontsize)
        _set_axis_label(ax_disco, "Permutation feature importance (DisCo)", fonts["label"], style, axis="y")
        ax_disco.set_ylim(ax_disco.get_ylim()[0], 1.1)
        _apply_tick_style(ax_disco, style, fonts["tick"])
        ax_disco.tick_params(axis="x", which="both", top=False)
        ax_disco.tick_params(axis="x", which="minor", bottom=False)
        if legend_handles:
            ax_disco.legend(
                handles=legend_handles,
                frameon=False,
                loc="upper right",
                ncol=len(legend_handles) if len(legend_handles) < 7 else 4,
                columnspacing=0.8,
                handlelength=1.0,
                fontsize=fonts["legend"],
                title_fontsize=fonts["title"],
            )
        plt.tight_layout()

    return (
        (fig, ax),
        (fig_jsd, ax_jsd),
        (fig_disco, ax_disco),
        importance_dict,
        importance_jsd,
        importance_disco,
    )


def jsd_plot(csv_path, has_header=True, forced_epoch=None, figure_layout=DEFAULT_FIGURE_LAYOUT):
    style = _get_plot_style(figure_layout, ratio=False, aspect=LINE_PLOT_ASPECT)
    fonts = style["fonts"]

    df = pd.read_csv(csv_path, index_col=0, header=0 if has_header else None)
    cms_colors = hep.styles.CMS["axes.prop_cycle"].by_key()["color"]

    if not has_header:
        df.columns = ["val_jsd"]

    val_jsd = df["val_jsd"].to_numpy() if "val_jsd" in df.columns else df.iloc[:, 0].to_numpy()
    val_clf = df["clf_val"].to_numpy() if "clf_val" in df.columns else df.iloc[:, 0].to_numpy()
    epochs = np.arange(1, len(val_jsd) + 1)

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(epochs, val_jsd, marker=".", linewidth=2, color=cms_colors[0], label="Validation set")
    _set_axis_label(ax, "Epoch", fonts["label"], style, axis="x")
    _set_axis_label(ax, "JSD CP-even vs. CP-odd", fonts["label"], style, axis="y")
    _apply_tick_style(ax, style, fonts["tick"])

    metric = np.array(val_jsd) * np.array(val_clf)
    if forced_epoch is not None:
        idx = int(forced_epoch) - 1
        if idx < 0 or idx >= len(metric):
            idx = int(metric.argmin())
    else:
        idx = int(metric.argmin())
    best_epoch = epochs[idx]
    ax.scatter(best_epoch, val_jsd[idx],
               color=cms_colors[2], zorder=5,
               label=f"Saved model (epoch {best_epoch}, JSD = {val_jsd[idx]:.5f})")

    ax.legend(fontsize=fonts["legend"], loc="best", frameon=True)
    fig.tight_layout()
    out_pdf = os.path.splitext(csv_path)[0] + "_jsd.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Plot saved at: {os.path.abspath(out_pdf)}")

