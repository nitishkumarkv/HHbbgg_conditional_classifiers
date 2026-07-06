import os
import sys
import multiprocessing
import numpy as np
import hist
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append("./..")
sys.path.append("./../ttH_tH_classification")
sys.path.append("./../signal_background_classification")
# We reuse your logic from the ML prep script:
from collections import defaultdict
#from prepareML_datasets import gather_filenames, process_single_file, reweight_ttbar_noPresel_in_DFs
#from extraStudies import QCD_MC_helper
from scipy.optimize import curve_fit
from joblib import Memory
#import plotBackgrounds
memory = Memory(location="./cache_dir2", verbose=0)
import plotter as plotter

# use gato-hep installed package utilities
from gatohep.utils import create_hist


background_processes = [
        "VBFHToGG_M_125",
        "VHtoGG_M_125",
        "ttHtoGG_M_125",
        "BBHto2G_M_125",
        "GluGluHToGG_M_125",
        #"GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
        #"GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
        #"VBFHH_CV_1p000_C2V_1p000_C3_1p000",
        "TTGG",
        "GGJets",
        "DDQCDGJET",
        "TTG_100_200",
        "TTG_200",
      ]
ignore_processes_lep = [
    # "GJet_PT-20to40","GJet_PT-40",
    "QCD_PT-30to40","QCD_PT-40toInf",
    "QCD_noPresel_PT-30to40", "QCD_noPresel_PT-40toInf",
    # "TTto4Q",
    # "TTto4Q_noPresel",
    # "TGQB",
    "TBbarQ","TbarBQ",
]
ignore_processes_had = [
    "TBbarQ","TbarBQ"
]


def infer_year_from_strings(process_name, filepath):
    if filepath and "2023" in filepath:
        return "2023"
    if "2023" in str(process_name):
        return "2023"
    return "2022"


@memory.cache(ignore=["use_multiprocessing", "n_workers"])
def load_dataframes_for_diff_cats(base_path, channel="had", entry_stop=None, n_workers=1, use_multiprocessing=None):
    """
    Reads in the background & signal processest, reweights QCD & TTbar,
    then returns a dictionary { "ttH": pd.DataFrame(...), "tHqLep":..., "GG-Box":..., ... }.
    """

    if use_multiprocessing is None:
        use_multiprocessing = n_workers > 1

    # signals
    signal_processes = [
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"
    ]

    # We gather the tasks, run them, do merges
    # define paths (Run-3 production samples)
    path_backgrounds = "/net/data_cms3a-2/nobackup/group_JE_CMS/Hgg_ttH_tH_Run3/ntuples/data_and_bkg_24Sep2025/"
    path_HGG = "/net/data_cms3a-2/nobackup/group_JE_CMS/Hgg_ttH_tH_Run3/ntuples/HGG_with_systs_18Aug2025/"
    path_data = path_backgrounds

    verbose_default = not use_multiprocessing

    background_tasks = []
    for proc in background_processes:
        if (proc in ignore_processes_lep and channel in ["lep","electron","muon"]) or (proc in ignore_processes_had and channel=="had"):
            continue
        base_path = path_HGG if proc in physicsHelper.list_HGG_processes else path_backgrounds
        pattern = os.path.join(base_path, f"{proc}_202*/merged/nominal/NOTAG_merged.parquet")
        flist = gather_filenames(pattern)
        if not flist:
            print(f"WARNING: found no files for process {proc}, {pattern=}")
        cross_section = physicsHelper.dict_cross_sections.get(proc, 0)
        for f_ in flist:
            era = physicsHelper.get_era(f_)
            lumi= physicsHelper.lumi_dict.get(era, 0)
            year = infer_year_from_strings(proc, f_)
            background_tasks.append( (f_, proc, channel, lumi, cross_section, entry_stop, verbose_default, False, year) )

    signal_tasks = []
    for sig_name in signal_processes:
        pattern = os.path.join(path_HGG, f"{sig_name}_202*/merged/nominal/NOTAG_merged.parquet")
        flist = gather_filenames(pattern)
        if not flist:
            print(f"WARNING: found no files for signal {sig_name}, {pattern=}")
        for f_ in flist:
            if "tHq" in f_:
                cross_sec_name = "tHqLep" if "lep" in f_.lower() else "tHqHad"
            elif "tHW" in f_:
                cross_sec_name = "tHW"
            elif "ttH" in f_:
                cross_sec_name = "ttH"
            else:
                cross_sec_name = sig_name
            xsec = physicsHelper.dict_cross_sections.get(cross_sec_name, 0)
            era = physicsHelper.get_era(f_)
            lumi = physicsHelper.lumi_dict.get(era, 0)
            year = infer_year_from_strings(sig_name, f_)
            signal_tasks.append( (f_, sig_name, channel, lumi, xsec, entry_stop, verbose_default, False, year) )

    all_tasks_MC = background_tasks + signal_tasks

    # data eras
    data_eras = [
        'Data_2022C','Data_2022D','Data_2022E','Data_2022F','Data_2022G',
        'Data_2023C_v123', 'Data_2023C_v4','Data_2023D'
    ]

    data_tasks = []
    for era in data_eras:
        pattern = os.path.join(path_data, era, "merged/nominal/NOTAG_merged.parquet")
        flist = gather_filenames(pattern)
        cross_section = 1
        for f_ in flist:
            lumi = physicsHelper.lumi_dict.get(era, 0)
            year = "2023" if "2023" in era else "2022"
            data_tasks.append( (f_, era, channel, lumi, cross_section, None, verbose_default, False, year) )

    def process_all_tasks(tasks, parallel=use_multiprocessing):
        if parallel:
            procs = max(1, min(int(n_workers), multiprocessing.cpu_count()))
            with multiprocessing.Pool(processes=procs) as pool:
                results = pool.starmap(process_single_file, tasks)
        else:
            results = []
            for t_ in tasks:
                out = process_single_file(*t_)
                results.append(out)
        return results

    results_MC = process_all_tasks(all_tasks_MC, parallel=use_multiprocessing)
    results_data = process_all_tasks(data_tasks, parallel=use_multiprocessing)

    # sort them into background_dfs, signal_dfs
    background_by_proc_year = defaultdict(lambda: defaultdict(list))
    signal_dfs = defaultdict(pd.DataFrame)

    for item in results_MC:
        if len(item) == 3:
            process_name, df_, _year = item
        else:
            process_name, df_ = item
            _year = None
        if df_.empty:
            continue
        if process_name in background_processes:
            tag_year = _year or "combined"
            background_by_proc_year[process_name][tag_year].append(df_)
        else:
            # treat everything else as signals
            signal_dfs[process_name] = pd.concat([signal_dfs[process_name], df_], ignore_index=True)

    # inclusive reweighting of dominant backgrounds
    optimized_scales = QCD_MC_helper.optimize_background_scaling(
        "signal_background_classification/Plots/had/phoID_reweighting_hists_had.pkl",
        variables=["phoLeadMVAID", "phoSubleadMVAID"],
        min_x=-0.7
    )

    combined_scale_map = {
        "GG-Box": float(optimized_scales.get("GG-Box", 1.0)) if isinstance(optimized_scales, dict) else 1.0,
        "GJet": float(optimized_scales.get("GJet", 1.0)) if isinstance(optimized_scales, dict) else 1.0,
        "QCD": float(optimized_scales.get("QCD", 1.0)) if isinstance(optimized_scales, dict) else 1.0,
    }
    per_year_scales = optimized_scales.get("per_year", {}) if isinstance(optimized_scales, dict) else {}

    background_dfs = defaultdict(pd.DataFrame)
    used_scales_per_year = defaultdict(dict)
    used_scales_combined = {}

    for proc, year_map in background_by_proc_year.items():
        frames = []
        for year_label, df_list in year_map.items():
            if not df_list:
                continue
            df_year = pd.concat(df_list, ignore_index=True)
            scale_map = per_year_scales.get(year_label, {}) if isinstance(per_year_scales, dict) else {}

            scale = 1.0
            scale_key = None
            if proc == "GG-Box":
                scale = float(scale_map.get("GG-Box", combined_scale_map["GG-Box"]))
                scale_key = "GG-Box"
            elif proc.startswith("GJet"):
                scale = float(scale_map.get("GJet", combined_scale_map["GJet"]))
                scale_key = "GJet"
            elif proc.startswith("QCD"):
                scale = float(scale_map.get("QCD", combined_scale_map["QCD"]))
                scale_key = "QCD"

            if scale_key:
                if year_label in per_year_scales and scale_key in per_year_scales[year_label]:
                    used_scales_per_year[year_label][scale_key] = scale
                else:
                    used_scales_combined[scale_key] = scale

            if scale != 1.0 and not df_year.empty:
                df_year = df_year.copy()
                df_year["weights"] *= scale
            frames.append(df_year)

        if frames:
            background_dfs[proc] = pd.concat(frames, ignore_index=True)

    def _format_scale_string(scale_map):
        ordered = [(key, scale_map[key]) for key in sorted(scale_map.keys())]
        return ", ".join(f"{key}={value:.3f}" for key, value in ordered)

    for yr in sorted(used_scales_per_year.keys()):
        if used_scales_per_year[yr]:
            print(f"[load_dataframes] Photon-ID scaling {yr}: {_format_scale_string(used_scales_per_year[yr])}")
    if used_scales_combined:
        print(f"[load_dataframes] Photon-ID scaling (inclusive): {_format_scale_string(used_scales_combined)}")

    # TTbarNoPresel and DYG (lep) reweight
    background_dfs = reweight_ttbar_noPresel_in_DFs(background_dfs, channel=channel, min_x=0.05)

    # Merge them into one final dictionary
    data_dict_MC = {}
    relevant_columns = ["massH", "weights", "ttH_vs_tH_NN", "sig_vs_bkg_NN_ttH_lep", "sig_vs_bkg_NN_tH_lep", "sig_vs_bkg_NN_ttH_had", "sig_vs_bkg_NN_tH_had"]
    for k, df_ in background_dfs.items():
        if df_.empty:
            continue
        # in lep channel, we have a small overall yield and a few MC events with large weights (mainly from GJet). Take them out.
        if channel == "lep":
            df_ = df_[np.abs(df_.weights) < 3]
        data_dict_MC[k] = df_[relevant_columns]
    for k, df_ in signal_dfs.items():
        if df_.empty:
            continue
        data_dict_MC[k] = df_[relevant_columns]

    df_data_frames = []
    for item in results_data:
        if len(item) == 3:
            _, df_, _ = item
        else:
            _, df_ = item
        df_data_frames.append(df_)

    df_data_obs = pd.concat(df_data_frames)
    df_data_obs = df_data_obs[relevant_columns]
    # blinding:
    df_data_obs = df_data_obs[
        (df_data_obs.massH < 120) | (df_data_obs.massH > 130)
    ]

    return data_dict_MC, df_data_obs


def make_schedule(lr, steps=50, decay=0.5):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=steps,
        decay_rate=decay,
        staircase=True,
    )

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
    figure_layout=0.48,
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



def convert_data_to_tensors(data_dict):
    """
    Convert the dictionary of pandas DataFrames (each containing the columns
    'massH', 'weights', 'ttH_vs_tH_NN', 'sig_vs_bkg_NN_ttH_lep', 'sig_vs_bkg_NN_tH_lep',
    'sig_vs_bkg_NN_ttH_had', 'sig_vs_bkg_NN_tH_had')
    into a dictionary of dictionaries with corresponding TF tensors.
    """
    # List of keys we want to convert.
    keys_to_convert = [
        "massH",
        "weights",  # this column is assumed to contain the event weights
        "ttH_vs_tH_NN",
        "sig_vs_bkg_NN_ttH_lep",
        "sig_vs_bkg_NN_tH_lep",
        "sig_vs_bkg_NN_ttH_had",
        "sig_vs_bkg_NN_tH_had"
    ]
    tensor_data = {}
    for proc, df in data_dict.items():
        # Create a new dictionary to store the converted tensors for each process.
        proc_tensors = {}
        for key in keys_to_convert:
            if key in df.columns:
                # Convert the column values to a numpy array, then to a TF tensor of type float32.
                proc_tensors[key] = tf.constant(np.array(df[key].values), dtype=tf.float32)
        tensor_data[proc] = proc_tensors
    return tensor_data


def sum_in_quad(z1, z2):
    return tf.sqrt(z1**2 + z2**2)

def arithmetic_mean(z1, z2):
    return (z1 + z2) / 2

def geometric_mean(z1, z2):
    return tf.sqrt(z1 * z2)

def harmonic_mean(z1, z2, eps=1e-9):
    return 2 * z1 * z2 / (z1 + z2 + eps)


def compute_nonres_reweight_factors(model, data_dict, channel, mass_sb_low=100.0, mass_sb_high=180.0, mass_sig_low=120.0, mass_sig_high=130.0, nbins=40, return_params=False):
    """
    1) Obtain current category boundaries from the model (hard).
    2) Build one hist.Hist mass histogram per (ttH / tH) sub-category,
       but only for non-resonant backgrounds, in the range [mass_sb_low, mass_sb_high].
    3) Fit that histogram to an exponential => predict integral in [mass_sig_low, mass_sig_high].
    4) Compare the predicted yield to the raw MC yield in [mass_sig_low, mass_sig_high]
       => define a reweight factor (pred / raw).
    5) Return arrays of reweight_factors: (factors_ttH, factors_tH).
    """
    bdict = model.get_effective_boundaries()  
    #  → bdict['tth_vs_th']  is length 1  
    #  → bdict['ttH_vs_bkg'] is length (ncat_ttH−1)  
    #  → bdict['tH_vs_bkg']  is length (ncat_tH−1)  
    boundary_region = float(bdict['tth_vs_th'][0])
    boundaries_ttH  = bdict['ttH_vs_bkg']
    boundaries_tH   = bdict['tH_vs_bkg']
    ncat_ttH = boundaries_ttH.size + 1
    ncat_tH  = boundaries_tH.size  + 1

    hists_ttH = [hist.Hist.new.Reg(nbins, mass_sb_low, mass_sb_high, name="mass").Weight() for _ in range(ncat_ttH)]
    hists_tH = [hist.Hist.new.Reg(nbins, mass_sb_low, mass_sb_high, name="mass").Weight()for _ in range(ncat_tH)]

    # We also want to keep track of the "raw" total yield in signal window
    # so that we can do: reweight factor = (fitted integral) / (raw in that window).
    raw_sigwin_ttH = np.zeros(ncat_ttH, dtype=float)
    raw_sigwin_tH  = np.zeros(ncat_tH,  dtype=float)

    # We'll define which processes are considered resonant vs. non-res.
    resonant = [
        "GluGluH","VBFH","VH","WplusH_Wto2Q","WplusH_WtoLNu",
        "WminusH_Wto2Q","WminusH_WtoLNu","ZH_Zto2Q","ZH_Zto2L","ZH_Zto2Nu",
        "ttH","tHqLep","tHqHad"
    ]

    # Helper function to get sub-category index from a score & boundaries
    def get_subcat_index(score, boundaries):
        # subcat=0 if score < boundaries[0]
        # subcat=i if boundaries[i-1] <= score < boundaries[i], etc.
        return np.searchsorted(boundaries, score, side='left')

    # ---------------------
    # 3) Fill the histograms
    # ---------------------
    for proc_name, df_ in data_dict.items():
        if proc_name in resonant or df_.empty or "cpodd" in proc_name.lower():
            continue

        mass_vals   = df_["massH"].values
        weights     = df_["weights"].values
        tthvsth_vals= df_["ttH_vs_tH_NN"].values

        # pick the correct columns for the sig-vs-bkg NN
        if channel in ["lep", "electron", "muon"]:
            sb_ttH_vals = df_["sig_vs_bkg_NN_ttH_lep"].values
            sb_tH_vals  = df_["sig_vs_bkg_NN_tH_lep"].values
        else:
            sb_ttH_vals = df_["sig_vs_bkg_NN_ttH_had"].values
            sb_tH_vals  = df_["sig_vs_bkg_NN_tH_had"].values

        # "Hard" region assignment:
        mask_ttH = (tthvsth_vals < boundary_region)
        mask_tH  = ~mask_ttH

        # -- Fill ttH region histograms
        if np.any(mask_ttH):
            mass_ttH    = mass_vals[mask_ttH]
            weight_ttH  = weights[mask_ttH]
            sb_ttH_score= sb_ttH_vals[mask_ttH]

            # sub-category index
            cat_idx = get_subcat_index(sb_ttH_score, boundaries_ttH)

            for icat in range(ncat_ttH):
                # fill histogram for [100..180]
                # but let's not forcibly cut here; the hist is already limited to [mass_sb_low,mass_sb_high]
                # so out-of-range entries just won't fill. 
                m_ = mass_ttH
                w_ = weight_ttH
                mask_cat = (cat_idx == icat)
                if not np.any(mask_cat):
                    continue
                hists_ttH[icat].fill(m_[mask_cat], weight=w_[mask_cat])

                # also sum raw yield in signal window
                in_sigwin = (m_>=mass_sig_low) & (m_<mass_sig_high) & mask_cat
                raw_sigwin_ttH[icat] += np.sum(w_[in_sigwin])

        # -- Fill tH region histograms
        if np.any(mask_tH):
            mass_tH     = mass_vals[mask_tH]
            weight_tH   = weights[mask_tH]
            sb_tH_score = sb_tH_vals[mask_tH]

            cat_idx = get_subcat_index(sb_tH_score, boundaries_tH)

            for icat in range(ncat_tH):
                m_ = mass_tH
                w_ = weight_tH
                mask_cat = (cat_idx == icat)
                if not np.any(mask_cat):
                    continue
                hists_tH[icat].fill(m_[mask_cat], weight=w_[mask_cat])

                in_sigwin = (m_>=mass_sig_low) & (m_<mass_sig_high) & mask_cat
                raw_sigwin_tH[icat] += np.sum(w_[in_sigwin])

    # ---------------------
    # 4) For each histogram, do an exponential fit => predicted yield in signal window
    #    Then the reweight factor = (predicted) / (raw_sigwin).
    # ---------------------

    def exp_func(x, A, B):
        """Exponential function A * exp(B*x)."""
        return A * np.exp(B*x)

    def integral_exp(A, B, xlo, xhi):
        """Integral of A exp(B*x) dx from xlo..xhi."""
        if abs(B) < 1e-10:
            # ~ constant => A*(xhi-xlo)
            return A*(xhi - xlo)
        return (A/B) * (np.exp(B*xhi) - np.exp(B*xlo))

    def fit_exponential(histogram, xlow, xhigh):
        """
        Fit A*exp(B*x) to the histogram histogram in [xlow,xhigh].
        Returns the tuple (A,B) or None if it fails or histogram is empty.
        """
        # retrieve bin edges, bin contents, errors
        edges = histogram.axes[0].edges  # array of length nbins+1
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        values = histogram.values()
        variances = histogram.variances()
        # ensure positivity in errors
        errors = np.sqrt(np.maximum(variances, 1e-12))

        # check if there's any content
        total = np.sum(values)
        if total < 1e-10:
            return None

        # initial guess: 
        #   A ~ first bin content, B ~ small negative slope
        A0 = max(1e-6, values[0])
        B0 = -0.01

        try:
            popt, pcov = curve_fit(
                exp_func,
                bin_centers,
                values,
                p0=[A0,B0],
                # sigma=errors, absolute_sigma=True,
                maxfev=2000
            )
            return popt  # (A, B)
        except RuntimeError:
            return None

    # We'll do: predicted integral in [mass_sig_low, mass_sig_high] from the fit
    # then ratio = predicted / raw_sigwin
    bin_width = float(mass_sb_high - mass_sb_low) / float(nbins)

    factors_ttH = []
    popts_ttH = []
    for icat in range(ncat_ttH):
        h = hists_ttH[icat]
        raw_in_sig = raw_sigwin_ttH[icat]
        popt = fit_exponential(h, mass_sb_low, mass_sb_high)
        if popt is None:
            factors_ttH.append(0.062)
            popts_ttH.append(None)
        else:
            A_fit, B_fit = popt
            # predicted integral in signal window (convert from area to counts by dividing by bin width)
            pred_in_sig = integral_exp(A_fit, B_fit, mass_sig_low, mass_sig_high) / bin_width
            pred_total = integral_exp(A_fit, B_fit, mass_sb_low, mass_sb_high) / bin_width
            if raw_in_sig>0:
                factors_ttH.append(pred_in_sig / pred_total)
            else:
                factors_ttH.append(0.062)
            popts_ttH.append(popt)

    # same for tH
    factors_tH = []
    popts_tH = []
    for icat in range(ncat_tH):
        h = hists_tH[icat]
        raw_in_sig = raw_sigwin_tH[icat]
        popt = fit_exponential(h, mass_sb_low, mass_sb_high)
        if popt is None:
            factors_tH.append(0.062)
            popts_tH.append(None)
        else:
            A_fit, B_fit = popt
            pred_in_sig = integral_exp(A_fit, B_fit, mass_sig_low, mass_sig_high) / bin_width
            pred_total = integral_exp(A_fit, B_fit, mass_sb_low, mass_sb_high) / bin_width
            if raw_in_sig>0:
                factors_tH.append(pred_in_sig / pred_total)
            else:
                factors_tH.append(0.062)
            popts_tH.append(popt)
    if return_params:
        return (np.array(factors_ttH, dtype=float), np.array(factors_tH, dtype=float)), (popts_ttH, popts_tH), (hists_ttH, hists_tH), bin_width
    else:
        return np.array(factors_ttH, dtype=float), np.array(factors_tH, dtype=float)


def compute_nonres_reweight_factors_3d(
        model,
        data_dict,               # raw pandas-dataframe dict
        channel,
        mass_sb_low  = 100.0,
        mass_sb_high = 180.0,
        mass_sig_low = 123.0,
        mass_sig_high= 127.0,
        reduce=False,
        nbins=40):
    """
    Returns 1D numpy array  factors[k]  (len = model.n_cats).

    factor[k] =  (continuum yield predicted in 123-127 GeV by
                  exp fit to side-bands)  /
                 (raw MC yield in that window)
    *hard* arg-max assignment per GMM component is used.
    """

    n_cats = model.n_cats

    # ------------------------------------------------------------------
    # 1) Hard-assign every MC event to one GMM component
    # ------------------------------------------------------------------
    assignments = {}                       # proc -> np.ndarray of ints
    for proc, df in data_dict.items():
        if not "NN_output" in df.columns:
            NN_output = build_pseudo_softmax(df, channel)             # (N,3)
            if reduce:
                NN_output = NN_output[:, :-1]
        else:
            NN_output = np.vstack(df["NN_output"].values)  # shape (N,3)
        # Use gato-hep API for assignments (hard bins)
        #bins = model.get_bin(NN_output)                   # tf.Tensor (N,)
        bins = model.get_bin_indices(NN_output)
        assignments[proc] = bins.numpy()

    # ------------------------------------------------------------------
    # 2) Build one mass histogram per category (continuum processes only)
    # ------------------------------------------------------------------
    hists = [hist.Hist.new.Reg(nbins, mass_sb_low, mass_sb_high,name="mass").Weight() for _ in range(n_cats)]
    raw_sigwin = np.zeros(n_cats)
    bin_width = (mass_sb_high - mass_sb_low) / nbins

    #resonant = {
    #    "GluGluH","VBFH","VH","WplusH_Wto2Q","WplusH_WtoLNu","WminusH_Wto2Q",
    #    "WminusH_WtoLNu","ZH_Zto2Q","ZH_Zto2L","ZH_Zto2Nu",
    #    "ttH","tHqLep","tHqHad"
    #}
    resonant = {
        "VBFHToGG_M_125", 
        "VHtoGG_M_125", 
        "ttHtoGG_M_125", 
        "BBHto2G_M_125", 
        "GluGluHToGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        "VBFHH_CV_1p000_C2V_1p000_C3_1p000"
    }


    for proc, df in data_dict.items():
        if proc in resonant or df.empty:
            continue

        m  = df.massH.values
        w  = df.weights.values
        cat = assignments[proc]

        for k in range(n_cats):
            mask = (cat == k)
            if not np.any(mask):
                continue
            hists[k].fill(m[mask], weight=w[mask])

            in_sig = mask & (m>=mass_sig_low) & (m<mass_sig_high)
            raw_sigwin[k] += np.sum(w[in_sig])
    # ------------------------------------------------------------------
    # 3) Fit A·exp(Bx) per hist  → prediction in signal window
    # ------------------------------------------------------------------
    def expf(x, A, B):
        return A*np.exp(B*x)
    def integral_exp(A, B, x1, x2):
        return (A/B)*(np.exp(B*x2)-np.exp(B*x1)) if abs(B)>1e-10 else A*(x2-x1)

    # 0.062 is what tipically comes out for 4GeV signal window
    factors = 0.062 * np.ones(n_cats)

    for k in range(n_cats):
        h = hists[k]
        if h.values().sum()==0:
            continue                                   # leave factor = 1

        edges = h.axes[0].edges
        centers = 0.5*(edges[:-1] + edges[1:])
        vals  = h.values()
        errs  = np.sqrt(np.maximum(h.variances(), 1e-12))
        if vals.sum()<1e-10:
            continue

        A0,B0 = max(vals[0],1e-6), -0.03
        try:
            (A,B),_ = curve_fit(expf, centers, vals, p0=[A0,B0], sigma=errs, absolute_sigma=True, maxfev=2000)
            pred_sig  = integral_exp(A,B, mass_sig_low, mass_sig_high) / bin_width
            pred_tot  = integral_exp(A,B, mass_sb_low, mass_sb_high) / bin_width
            if pred_tot>0:
                factors[k] = pred_sig / pred_tot
        except RuntimeError:
            pass                                        # keep factor = 1

        # if MC has zero in sig window, default to naive window fraction
        if raw_sigwin[k] <= 0.0:
            factors[k] = (mass_sig_high - mass_sig_low) / (mass_sb_high - mass_sb_low)
    print("factors", factors)
    return factors


# A helper function to create a diphoton mass histogram.
def create_mass_hist():
    # 80 bins in [100, 180] GeV; adjust as needed.
    return hist.Hist.new.Reg(40, 100, 180, name="massH", label="Diphoton Mass (GeV)").Weight()


def group_histograms(hist_dict, groupings):
    """
    Groups histograms by the specified groupings. 
    E.g. 'TTbar' => sum of TTto4Q, TTto2L2Nu, etc.
    """
    grouped_hist_dict = {}
    for group_name, processes in groupings.items():
        processes_in_hist_dict = [p for p in processes if p in hist_dict]
        if not processes_in_hist_dict:
            continue
        grouped_hist_dict[group_name] = {}

        grouped_hist_list = None
        for proc in processes_in_hist_dict:
            # if variable in hist_dict[proc]:
            if grouped_hist_list is None:
                grouped_hist_list = hist_dict[proc].copy()
            else:
                for i in range(len(grouped_hist_list)):
                    grouped_hist_list[i] += hist_dict[proc][i]
        if grouped_hist_list is not None:
            grouped_hist_dict[group_name] = grouped_hist_list
    # now add leftover processes not in groupings
    grouped_procs = [p for g in groupings.values() for p in g]
    remaining = [p for p in hist_dict if p not in grouped_procs]
    for proc in remaining:
        grouped_hist_dict[proc] = hist_dict[proc]
    return grouped_hist_dict


def compute_combined_totals(ttH_grouped, tH_grouped):
    """
    Compute the combined total yield for each process across both regions.
    Returns a dictionary mapping process names to the total yield (sum over all categories in ttH and tH).
    """
    combined_totals = {}
    # First, add yields from the ttH region.
    for proc, hist_list in ttH_grouped.items():
        total = np.sum([np.sum(h.values()) for h in hist_list])
        combined_totals[proc] = total

    # Then add yields from the tH region.
    for proc, hist_list in tH_grouped.items():
        yield_sum = np.sum([np.sum(h.values()) for h in hist_list])
        if proc in combined_totals:
            combined_totals[proc] += yield_sum
        else:
            combined_totals[proc] = yield_sum
    return combined_totals

def compute_metrics_for_region(grouped_hists, combined_totals):
    """
    For a dictionary of grouped histograms for a single region (keys: process names, values: list of hist.Hist objects),
    compute per-process and per-category metrics:
    - yield: sum of histogram bin contents in that category,
    - uncertainty: sqrt(sum of variances) in that category,
    - efficiency: yield in that category divided by the combined total yield for that process (across both regions).
    
    Returns:
    A dictionary with keys = category index (0,1,...) and values = dict mapping process name -> metrics dict.
    """
    cat_metrics = {}
    # Determine the number of categories for this region (use the maximum length of histogram lists).
    ncat = max(len(hist_list) for hist_list in grouped_hists.values())
    for cat in range(ncat):
        cat_metrics[cat] = {}
        for proc, hist_list in grouped_hists.items():
            if cat >= len(hist_list):
                continue
            hist_obj = hist_list[cat]
            # Yield: sum over bin contents.
            yield_cat = np.sum(hist_obj.values())
            # Uncertainty: sqrt(sum of variances).
            unc_cat = np.sqrt(np.sum(hist_obj.variances()))
            # Efficiency: yield in this category divided by combined total for this process.
            tot = combined_totals.get(proc, 0.0)
            eff = yield_cat / tot if tot > 0 else 0.0
            cat_metrics[cat][proc] = {
                "yield": yield_cat,
                "uncertainty": unc_cat,
                "efficiency": eff
            }
    return cat_metrics

def save_metrics_to_txt(ttH_metrics, tH_metrics, out_filename):
    """
    Save the per-category metrics for both regions into a text file.
    The file will contain separate sections for the ttH and tH regions.
    """
    with open(out_filename, "w") as f:
        f.write("=== ttH Region Metrics ===\n\n")
        for cat, proc_dict in ttH_metrics.items():
            f.write(f"--- ttH Category {cat+1} ---\n")
            f.write(f"{'Process':30s} {'Yield':>12s} {'Uncertainty':>15s} {'Efficiency':>12s}\n")
            for proc, metrics in proc_dict.items():
                f.write(f"{proc:30s} {metrics['yield']:12.2f} {metrics['uncertainty']:15.2f} {metrics['efficiency']:12.3f}\n")
            f.write("\n")
        f.write("\n=== tH Region Metrics ===\n\n")
        for cat, proc_dict in tH_metrics.items():
            f.write(f"--- tH Category {cat+1} ---\n")
            f.write(f"{'Process':30s} {'Yield':>12s} {'Uncertainty':>15s} {'Efficiency':>12s}\n")
            for proc, metrics in proc_dict.items():
                f.write(f"{proc:30s} {metrics['yield']:12.2f} {metrics['uncertainty']:15.2f} {metrics['efficiency']:12.3f}\n")
            f.write("\n")

def get_ordered_indices(process_names, Zee=False):
    """
    Returns indices to reorder process_names according to processes_order.

    Any process not in processes_order will be placed at the end in their original order.

    Args:
        process_names (list): List of process names to be ordered.

    Returns:
        list: Indices that can be used to reorder process_names.
    """
    processes_order = []
    #processes_order = [
    #    "VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "VBFHH_CV_1p000_C2V_1p000_C3_1p000"
    #]

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


# --- Plot the stacked histograms per sub-category in each region ---
def plot_mass_spectrum_by_subcat(region, channel, grouped_hists, subcat_idx, out_fname, *, return_figure=False):
    """
    region: string, e.g. "ttH" or "tH"
    grouped_hists: dict mapping group names to a list of histograms per sub-category.
    subcat_idx: integer index (0 or 1 for two sub-categories)
    out_fname: output PDF filename.
    """

    get_process_tex_names_dict = {
        "VBFHToGG_M_125": "VBFHToGG_M_125",
        "VHtoGG_M_125": "VHtoGG_M_125",
        "ttHtoGG_M_125": "ttHtoGG_M_125",
        "BBHto2G_M_125": "BBHto2G_M_125",
        "GluGluHToGG_M_125": "GluGluHToGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        "VBFHH_CV_1p000_C2V_1p000_C3_1p000": "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
        "TTGG": "TTGG",
        "GGJets": "GGJets",
        "DDQCDGJET": "DDQCDGJET",
        "TTG_100_200": "TTG_100_200",
        "TTG_200": "TTG_200",
    }

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
    cms_colors = PETROFF_COLORS_10
    tab_colors = ["tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:brown", "black", "white"]
    all_colors = cms_colors + tab_colors
    get_process_colors_new = {
        "VBFHToGG_M_125": all_colors[0],
        "VHtoGG_M_125": all_colors[1],
        "ttHtoGG_M_125": all_colors[2],
        "BBHto2G_M_125": all_colors[3],
        "GluGluHToGG_M_125": all_colors[4],
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": all_colors[5],
        "VBFHH_CV_1p000_C2V_1p000_C3_1p000": all_colors[6],
        "TTGG": all_colors[7],
        "GGJets": all_colors[8],
        "DDQCDGJET": all_colors[9],
        "TTG_100_200": all_colors[10],
        "TTG_200": all_colors[11],

        "VBFHToGG": all_colors[0],
        "VHToGG": all_colors[1],
        "ttHToGG": all_colors[2],
        "BBHToGG": all_colors[3],
        "GluGluHToGG": all_colors[4],
        "GluGluToHH_kl-1p00_kt-1p00_c2-0p00": all_colors[5],
        "VBFToHH_CV-1p000_C2V-1p000_C3-1p000": all_colors[6],
        "TTGG": all_colors[7],
        "GGJets": all_colors[8],
        "DDQCDGJets": all_colors[9]
    }


    label_mapping = get_process_tex_names_dict

    merge_map = {
        #"TTGG_TJGG": ["TTGG", "TJGG"],
        #"VG": ["VG", "WG", "DYG"],
    }

    def combine_members(members):
        combined = None
        for name in members:
            if name not in grouped_hists:
                continue
            hlist = grouped_hists[name]
            if combined is None:
                combined = [h.copy() for h in hlist]
            else:
                for idx in range(len(combined)):
                    combined[idx] += hlist[idx]
        return combined

    slimmed = {}
    consumed = set()
    for slim, members in merge_map.items():
        combined = combine_members(members)
        if combined is not None:
            slimmed[slim] = combined
            consumed.update([m for m in members if m in grouped_hists])

    for proc, hist_list in grouped_hists.items():
        if proc in consumed and proc not in slimmed:
            continue
        if proc in slimmed:
            continue
        slimmed[proc] = hist_list

    if "Data" not in slimmed:
        raise KeyError("Data histogram missing for mass spectrum plot")

    data_hist = slimmed["Data"][subcat_idx]

    # Drop CP-odd samples from plots
    for key in [name for name in list(slimmed.keys()) if "CPodd" in name]:
        if key != "Data":
            slimmed.pop(key, None)

    #ttH_list = slimmed.pop("ttH", None)
    #tHW_list = slimmed.pop("tHW", None)
    #tHq_list = slimmed.pop("tHq", None)

    background_hists = []
    background_labels = []
    background_colors = []
    background_names = []
    for proc, hist_list in slimmed.items():
        if proc == "Data":
            continue
        if subcat_idx >= len(hist_list):
            continue
        background_hists.append(hist_list[subcat_idx])
        background_labels.append(label_mapping.get(proc, proc))
        background_colors.append(get_process_colors_new[proc])
        background_names.append(proc)

    if background_names:
        ordered_indices = get_ordered_indices(background_names, False)
        background_hists = [background_hists[i] for i in ordered_indices]
        background_labels = [background_labels[i] for i in ordered_indices]
        background_colors = [background_colors[i] for i in ordered_indices]
        background_names = [background_names[i] for i in ordered_indices]

    if "HGG" in background_names:
        idx = background_names.index("HGG")
        # move HGG to the end so it stacks last
        background_hists.append(background_hists.pop(idx))
        background_labels.append(background_labels.pop(idx))
        background_colors.append(background_colors.pop(idx))
        background_names.append(background_names.pop(idx))

    if channel in ("lep", "electron", "muon"):
        scaling_ttH = 2
        scaling_tHq = 20
    else:
        scaling_ttH = 5
        scaling_tHq = 50

    signal_hists = []
    signal_labels = []
    signal_colors = []

    combined_tth = None
#    if ttH_list is not None and subcat_idx < len(ttH_list):
#        combined_tth = ttH_list[subcat_idx].copy()
#    if tHW_list is not None and subcat_idx < len(tHW_list):
#        combined_tth = tHW_list[subcat_idx].copy() if combined_tth is None else combined_tth + tHW_list[subcat_idx]
#    if combined_tth is not None:
#        combined_tth_scaled = combined_tth * scaling_ttH
#        signal_hists.append(combined_tth_scaled)
#        signal_labels.append(rf'$(t\bar{{t}}H + tHW)\times {scaling_ttH}$')
#        signal_colors.append('tab:red')
#
#    if tHq_list is not None and subcat_idx < len(tHq_list):
#        tHq_scaled = tHq_list[subcat_idx] * scaling_tHq
#        signal_hists.append(tHq_scaled)
#        signal_labels.append(rf'$tHq\times {scaling_tHq}$')
#        signal_colors.append('tab:orange')

    lumi_dict = {
    "2016preVFP": 19.5,  # Integrated luminosity for preEE in fb^-1
    "2016postVFP": 16.8,  # Integrated luminosity for preEE in fb^-1
    "2017": 42.07,  # Integrated luminosity for preEE in fb^-1
    "2018": 59.56, # Integrated luminosity in fb^-1
    "2022preEE": 7.98,
    "2022postEE": 26.68,
    "2023preBPix": 17.96,
    "2023postBPix": 9.68,
    "2024": 108.95,  # Integrated luminosity in fb^-1
    #"2024": 109.95  
    }


    lumi = sum(lumi_dict.values())
    result = plotter.plot_stacked_histograms(
        stacked_hists=background_hists,
        process_labels=background_labels,
        signal_hists=signal_hists,
        signal_labels=signal_labels,
        data_hist=data_hist,
        output_filename=out_fname,
        include_flow=True,
        axis_labels=("Diphoton Mass (GeV)", "Events"),
        lumi=lumi,
        ratio=False,
        ratio_limits=None,
        colors=background_colors,
        signal_colors=signal_colors,
        log=False,
        return_figure=return_figure,
    )
    if return_figure:
        return result
    print(f"[INFO] {region} region, sub-category {subcat_idx} plot saved as {out_fname}")


# ======== History plotting helpers (used by optimization runners) ========
def plot_significance_and_loss_histories(out_dir, z_hist, loss_hist, reg_hist):
    epochs = np.arange(len(z_hist))

    # Significance history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, z_hist, marker='o', label="ttH")
    #ax.plot(epochs, z_tH_hist, marker='o', label="tH")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Significance", fontsize=22)
    ax.legend(fontsize=20, loc="upper right")
    ax.set_ylim(0, 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory.pdf"))
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 3*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory_log.pdf"))
    plt.close(fig)

    # Loss history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss_hist, marker='o', label="Loss")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel(r"Neg. geom. mean ($z_{ttH}$)", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Loss.pdf"))
    plt.close(fig)

    # Regularisation history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, reg_hist, marker='o', label="Regularisation")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Low-background penalty", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation.pdf"))
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation_log.pdf"))
    plt.close(fig)

def plot_significance_and_loss_histories_(out_dir, z_hist, loss_hist, reg_hist):
    epochs = np.arange(len(z_hist))

    # Significance history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, z_hist, marker='o', label="ggHH")
    #ax.plot(epochs, z_tH_hist, marker='o', label="tH")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Significance", fontsize=22)
    ax.legend(fontsize=20, loc="upper right")
    ax.set_ylim(0, 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory.pdf"))
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 3*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory_log.pdf"))
    plt.close(fig)

    # Loss history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss_hist, marker='o', label="Loss")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel(r"Neg. geom. mean ($z_{ggHH}$)", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Loss.pdf"))
    plt.close(fig)

    # Regularisation history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, reg_hist, marker='o', label="Regularisation")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Low-background penalty", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation.pdf"))
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation_log.pdf"))
    plt.close(fig)

def plot_significance_and_loss_histories(out_dir, z_ttH_hist, z_tH_hist, loss_hist, reg_hist):
    epochs = np.arange(len(z_ttH_hist))

    # Significance history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, z_ttH_hist, marker='o', label="ttH")
    ax.plot(epochs, z_tH_hist, marker='o', label="tH")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Significance", fontsize=22)
    ax.legend(fontsize=20, loc="upper right")
    ax.set_ylim(0, 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory.pdf"))
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 3*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "significanceHistory_log.pdf"))
    plt.close(fig)

    # Loss history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, loss_hist, marker='o', label="Loss")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel(r"Neg. geom. mean ($z_{ttH}, z_{tH}$)", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Loss.pdf"))
    plt.close(fig)

    # Regularisation history
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, reg_hist, marker='o', label="Regularisation")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Low-background penalty", fontsize=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation.pdf"))
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Regularisation_log.pdf"))
    plt.close(fig)


def plot_bias_history_with_temp(out_dir, bias_epochs, bias_history, temp_points, ylabel="Temperature"):
    if not bias_history:
        return
    bias_arr = np.stack(bias_history)
    bias_x = np.array(bias_epochs)
    mean_bias = np.mean(np.abs(bias_arr), axis=1)

    # Linear
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bias_x, mean_bias, marker='o', color='C0', label="Mean bias")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Mean soft–hard bias", fontsize=22, color='C0')
    ax.tick_params(axis='y', colors='C0')
    ax.spines['left'].set_color('C0')
    ax2 = ax.twinx()
    ax2.plot(bias_x, np.array(temp_points), color='C1', linestyle='--', marker='s', label=ylabel)
    ax2.set_ylabel("Temperature", fontsize=22, color='C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.spines['right'].set_color('C1')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=2, fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "biasHistory.pdf"))
    plt.close(fig)

    # Log for bias
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = mean_bias[mean_bias > 0]
    min_pos = float(np.min(pos)) if pos.size else None
    ax.plot(bias_x, mean_bias, marker='o', color='C0', label="Mean bias")
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Mean soft–hard bias", fontsize=22, color='C0')
    if min_pos is not None:
        ax.set_yscale('log')
        ax.set_ylim(max(min_pos * 0.7, 1e-6), None)
    ax.tick_params(axis='y', colors='C0')
    ax.spines['left'].set_color('C0')
    ax2 = ax.twinx()
    ax2.plot(bias_x, np.array(temp_points), color='C1', linestyle='--', marker='s', label=ylabel)
    ax2.set_ylabel(ylabel, fontsize=22, color='C1')
    ax2.tick_params(axis='y', colors='C1')
    ax2.spines['right'].set_color('C1')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=2, fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "biasHistory_log.pdf"))
    plt.close(fig)


def plot_yield_histories_(out_dir, epochs_arr, S_ttH_history, B_nonres_history, channel, n_cats):

    # ttH-like region yields (S vs B)
    fig, ax = plt.subplots(figsize=(8, 6))
    S_ttH_arr = np.stack(S_ttH_history)
    ncat_ttH = S_ttH_arr.shape[1]
    for cat in range(ncat_ttH):
        ax.plot(epochs_arr, S_ttH_arr[:, cat], label=fr"S, $ggHH$ cat. {cat}", linewidth=3)
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Yields", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_ggHH_linear.pdf"))
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 10*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_ggHH_log.pdf"))
    plt.close(fig)

    

    # Integrated continuum background
    fig, ax = plt.subplots(figsize=(8, 6))
    B_arr = np.stack(B_nonres_history)
    for cat in range(n_cats):
        ax.plot(epochs_arr, B_arr[:, cat], label=fr"Category {cat}", linewidth=3)
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel(r"Continuum bkg. ($100-180\,$GeV)", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_cont_bkg_linear.pdf"))
    ax.set_yscale('log')
    ax.set_ylim(1, 50*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_cont_bkg_log.pdf"))
    plt.close(fig)

def plot_yield_histories(out_dir, epochs_arr, S_ttH_history, S_tH_history, B_nonres_history, channel, n_cats):

    # ttH-like region yields (S vs B)
    fig, ax = plt.subplots(figsize=(8, 6))
    S_ttH_arr = np.stack(S_ttH_history)
    ncat_ttH = S_ttH_arr.shape[1]
    for cat in range(ncat_ttH):
        ax.plot(epochs_arr, S_ttH_arr[:, cat], label=fr"S, $t\bar{{t}}H$ cat. {cat}", linewidth=3)
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Yields", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_ttH_linear.pdf"))
    ax.set_yscale('log')
    ax.set_ylim(1e-1, 10*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_ttH_log.pdf"))
    plt.close(fig)

    # tH-like region yields (S vs B)
    fig, ax = plt.subplots(figsize=(8, 6))
    S_tH_arr = np.stack(S_tH_history)
    ncat_tH = S_tH_arr.shape[1]
    for cat in range(ncat_tH):
        ax.plot(epochs_arr, S_tH_arr[:, cat], label=fr"S, $tH$ cat. {cat}", linewidth=3)
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Yields", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_tH_linear.pdf"))
    ax.set_yscale('log')
    ax.set_ylim(1e-2 if channel=="had" else 2e-3, 50*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_tH_log.pdf"))
    plt.close(fig)

    # Integrated continuum background
    fig, ax = plt.subplots(figsize=(8, 6))
    B_arr = np.stack(B_nonres_history)
    for cat in range(n_cats):
        ax.plot(epochs_arr, B_arr[:, cat], label=fr"Category {cat}", linewidth=3)
    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel(r"Continuum bkg. ($100-180\,$GeV)", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)
    ax.set_ylim(ax.get_ylim()[0], 1.2*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_cont_bkg_linear.pdf"))
    ax.set_yscale('log')
    ax.set_ylim(1, 50*ax.get_ylim()[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "yields_cont_bkg_log.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------
# Build a 3-component pseudo-softmax vector from your three classifiers
# ------------------------------------------------------------------
def build_pseudo_softmax(df, channel):
    """
    Returns array of shape (N_events, 3) with columns:
        [P_ttH, P_tH, P_bkg]    (sums to 1 per row)
    """
    s_ttH_tH  = df["ttH_vs_tH_NN"].values * (1/0.95)

    if channel in ("lep", "electron", "muon"):
        s_ttH_bkg = df["sig_vs_bkg_NN_ttH_lep"].values
        s_tH_bkg  = df["sig_vs_bkg_NN_tH_lep"].values
    else:
        s_ttH_bkg = df["sig_vs_bkg_NN_ttH_had"].values
        s_tH_bkg  = df["sig_vs_bkg_NN_tH_had"].values

    P_ttH = (1-s_ttH_tH) * s_ttH_bkg
    P_tH  = s_ttH_tH * s_tH_bkg
    P_bkg = (1.0 - s_ttH_bkg) * (1.0 - s_tH_bkg)

    norm = P_ttH + P_tH + P_bkg
    P_ttH /= norm
    P_tH  /= norm
    P_bkg /= norm

    return np.stack([P_ttH, P_tH, P_bkg], axis=1)   # (N,3)



def plot_pseudo_softmax_marginals(data_dict, out_dir, channel):
    """
    Creates six PDFs inside  <out_dir>/input_softmax/ :
        psmx_dim0.pdf / _log.pdf   (P_ttH)
        psmx_dim1.pdf / _log.pdf   (P_tH)
        psmx_dim2.pdf / _log.pdf   (P_bkg)

    * background colours from  plotter.get_process_colour
    * signals boosted:   ttH × 150 ,  tH × 300
    * cp-odd & data samples are ignored
    """
    plot_dir = os.path.join(out_dir, "input_softmax")
    os.makedirs(plot_dir, exist_ok=True)

    # nice axis titles that match the definition in build_pseudo_softmax
    comp_labels = [r"$P_{ttH}$", r"$P_{tH}$", r"$P_{bkg}$"]

    def is_signal(proc):
        return proc.startswith("ttH") or proc.startswith("tHq") or proc.startswith("tHW")

    def bkg_colour(proc):
        try:
            return plotter.get_process_colour(proc)
        except Exception:
            return None   # matplotlib default

    for dim in range(3):

        # 1) raw histograms per process
        per_proc_hlists = {}
        for proc, df in data_dict.items():
            if df.empty:                                      continue
            if "cpodd" in proc.lower() or "data" in proc.lower():
                continue

            x3   = build_pseudo_softmax(df, channel)          # (N,3)
            vals = x3[:, dim]

            per_proc_hlists[proc] = [create_hist(
                vals, df["weights"].values,
                bins=50, low=0.0, high=1.0
            )]

        # 2)  group *lists*  → unwrap single hist per group
        grouped = {g: hlist[0] for g, hlist in
                plotBackgrounds.group_histograms(per_proc_hlists,).items()}

        # 3) split into background / signal, collect colours & scales
        bkg_h, bkg_lbl, bkg_col = [], [], []
        sig_h, sig_lbl          = [], []

        for grp, h in grouped.items():
            if is_signal(grp):
                scale = 150.0 if (grp.startswith("ttH") or grp.startswith("tHW")) else 300.0
                sig_h.append(h * scale)
                sig_lbl.append(f"{grp} ×{int(scale)}")
            else:
                bkg_h.append(h)
                bkg_lbl.append(grp)
                bkg_col.append(bkg_colour(grp))

        # 4) linear & log plots
        for log in (False, True):
            tag  = "_log" if log else ""
            fout = os.path.join(plot_dir, f"psmx_dim{dim}{tag}.pdf")

            plot_stacked_histograms(
                stacked_hists   = bkg_h,
                process_labels  = bkg_lbl,
                colors = bkg_col,
                signal_hists    = sig_h,
                signal_labels   = sig_lbl,
                log             = log,
                output_filename = fout,
                axis_labels     = (comp_labels[dim], "Events"),
            )
            print(f"[INFO] wrote {fout}")


def compute_bias_piecewise_sigmoid(model, tensor_data, channel, eps=1e-8):
    """
    Compute per-category bias for the piecewise sigmoid scheme used in the
    sigmoid runner: one split on ttH_vs_tH, then sub-bins on the relevant
    sig-vs-bkg discriminant per side.

    Returns np.ndarray of shape (n_cats,) with (hard - soft)/hard per bin.
    """
    b0 = model.calculate_boundaries(0)
    b1 = model.calculate_boundaries(1)
    b2 = model.calculate_boundaries(2)
    k0 = model.var_cfg[0]["k"]
    k1 = model.var_cfg[1]["k"]
    k2 = model.var_cfg[2]["k"]

    ntth = int(model.var_cfg[1]["bins"])
    nth  = int(model.var_cfg[2]["bins"])
    ncat = ntth + nth

    def soft_1d(x, boundaries, k):
        x = tf.convert_to_tensor(x, tf.float32)
        b = tf.convert_to_tensor(boundaries, tf.float32)
        k = tf.convert_to_tensor(k, tf.float32)
        m = tf.shape(b)[0]
        def when_empty():
            N = tf.shape(x)[0]
            return tf.ones((N, 1), tf.float32)
        def when_nonempty():
            sig = 1.0 / (1.0 + tf.exp(tf.clip_by_value(-k * (tf.expand_dims(x, 1) - tf.expand_dims(b, 0)), -75.0, 75.0)))
            left = 1.0 - sig[:, :1]
            middle = sig[:, :-1] - sig[:, 1:]
            right = sig[:, -1:]
            return tf.concat([left, middle, right], axis=1)
        return tf.cond(m == 0, when_empty, when_nonempty)

    hard_tot = tf.zeros(ncat, tf.float32)
    soft_tot = tf.zeros(ncat, tf.float32)

    for proc, t in tensor_data.items():
        if "cpodd" in proc.lower() or "data" in proc.lower():
            continue
        if channel in ("lep", "electron", "muon"):
            col_tth = "sig_vs_bkg_NN_ttH_lep"
            col_th  = "sig_vs_bkg_NN_tH_lep"
        else:
            col_tth = "sig_vs_bkg_NN_ttH_had"
            col_th  = "sig_vs_bkg_NN_tH_had"

        x_region = t["ttH_vs_tH_NN"]
        x_ttH = t[col_tth]
        x_tH  = t[col_th]
        w = t["weights"]

        wr = soft_1d(x_region, b0, k0)
        w_ttH = soft_1d(x_ttH, b1, k1)
        w_tH  = soft_1d(x_tH,  b2, k2)

        N = tf.shape(x_region)[0]
        zeros_ttH = tf.zeros((N, nth), tf.float32)
        zeros_tH  = tf.zeros((N, ntth), tf.float32)
        soft_left  = wr[:, :1] * tf.concat([w_ttH, zeros_ttH], axis=1)
        soft_right = wr[:, 1:] * tf.concat([zeros_tH, w_tH], axis=1)
        soft = soft_left + soft_right

        hard_idx = tf.argmax(soft, axis=1, output_type=tf.int32)
        hard_tot += tf.math.unsorted_segment_sum(w, hard_idx, ncat)
        soft_tot += tf.reduce_sum(soft * w[:, None], axis=0)

    bias = (hard_tot - soft_tot) / tf.maximum(hard_tot, eps)
    return bias.numpy()


def plot_boundary_histories(out_dir, b0_hist, b1_hist, b2_hist,
                            range_tth_vs_th=(0.0, 1.0),
                            range_ttH=(0.0, 1.0),
                            range_tH=(0.0, 1.0)):
    """
    Plot evolution of learned boundary positions across epochs for the
    sigmoid model: the region split (ttH_vs_tH) and the two sets of sub-cuts.

    - b0_hist: list[float] of the single ttH_vs_tH cut per epoch
    - b1_hist: list[list[float]] of ttH sub-cuts per epoch (length = n_ttH-1)
    - b2_hist: list[list[float]] of tH  sub-cuts per epoch (length = n_tH-1)
    """
    os.makedirs(out_dir, exist_ok=True)

    epochs = np.arange(len(b0_hist))

    # 1) Region split boundary
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, b0_hist, marker='o', linewidth=2, label='ttH_vs_tH')
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Boundary', fontsize=16)
    ax.set_ylim(range_tth_vs_th)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'boundaryHistory_tth_vs_th.pdf'))
    plt.close(fig)

    def _plot_family(bhist, yrange, title, fname):
        if len(bhist) == 0:
            return
        arr = np.array(bhist)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(arr.shape[1]):
            ax.plot(epochs, arr[:, i], linewidth=2, label=f'cut {i}')
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Boundary', fontsize=16)
        ax.set_ylim(yrange)
        ax.legend(ncol=2, fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)

    _plot_family(b1_hist, range_ttH, 'ttH sub-cuts', 'boundaryHistory_ttH.pdf')
    _plot_family(b2_hist, range_tH,  'tH sub-cuts',  'boundaryHistory_tH.pdf')


def plot_boundary_histories(
    out_dir,
    b0_hist,
    b1_hist,
    b2_hist,
    range_tth_vs_th=(0.0, 1.0),
    range_ttH=(0.0, 1.0),
    range_tH=(0.0, 1.0),
):
    """
    Plot evolution of learned boundary positions across epochs in a single figure,
    styled similarly to differentiableCategories.py.

    Inputs
    - b0_hist: list[float]           # single cut (ttH_vs_tH) per epoch
    - b1_hist: list[list[float]]     # ttH sub-cuts per epoch (length n_ttH-1)
    - b2_hist: list[list[float]]     # tH  sub-cuts per epoch (length n_tH-1)
    - range_* : y-axis ranges for each family; used to guide log-scale ylim if needed
    """
    os.makedirs(out_dir, exist_ok=True)

    # Epoch indexing
    epochs = np.arange(len(b0_hist))

    # Style: consistent with differentiableCategories
    linestyles = ["solid", "dashed", "dotted", "dashdot"] * 100
    lw = 3

    # Convert to arrays for robust indexing (handle empty gracefully)
    ttH_tH_arr = np.array(b0_hist).reshape(-1, 1) if len(b0_hist) else np.zeros((0, 1))
    b1_arr = np.array(b1_hist) if len(b1_hist) else np.zeros((len(b0_hist), 0))
    b2_arr = np.array(b2_hist) if len(b2_hist) else np.zeros((len(b0_hist), 0))

    # Combined (linear) figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # ttH vs tH boundary (color C0)
    if ttH_tH_arr.shape[1] > 0:
        for i in range(ttH_tH_arr.shape[1]):
            ax.plot(
                epochs,
                ttH_tH_arr[:, i],
                linestyle=linestyles[i],
                color="C0",
                linewidth=lw,
                label=rf"$t\bar{{t}}H$ vs. $tH$ cut {i+1}",
            )

    # sig_vs_bkg_ttH cuts (color C1)
    if b1_arr.ndim == 2 and b1_arr.shape[1] > 0:
        for i in range(b1_arr.shape[1]):
            ax.plot(
                epochs,
                b1_arr[:, i],
                linestyle=linestyles[i],
                color="C1",
                linewidth=lw,
                label=rf"Sig. vs. bkg. $t\bar{{t}}H$ cut {i+1}",
            )

    # sig_vs_bkg_tH cuts (color C2)
    if b2_arr.ndim == 2 and b2_arr.shape[1] > 0:
        for i in range(b2_arr.shape[1]):
            ax.plot(
                epochs,
                b2_arr[:, i],
                linestyle=linestyles[i],
                color="C2",
                linewidth=lw,
                label=rf"Sig. vs. bkg. $tH$ cut {i+1}",
            )

    ax.set_xlabel("Iteration", fontsize=22)
    ax.set_ylabel("Boundary values", fontsize=22)
    ax.legend(ncol=2, fontsize=16, loc="upper right", labelspacing=0.4, columnspacing=1.5)

    # Expand a bit on linear scale (if finite)
    y0, y1 = ax.get_ylim()
    if np.isfinite(y0) and np.isfinite(y1) and (y1 > 0):
        ax.set_ylim(y0, 1.3 * y1)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boundariesHistory.pdf"))

    # Log-scale version: guard against non-positive lower bounds
    ax.set_yscale("log")
    y0, y1 = ax.get_ylim()
    # Choose a reasonable lower bound > 0
    lower = max(1e-6, y0, min(range_tth_vs_th[0], range_ttH[0], range_tH[0]))
    ax.set_ylim(lower, 3 * max(y1, lower * 10.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boundariesHistory_log.pdf"))
    plt.close(fig)
