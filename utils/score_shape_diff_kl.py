import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplhep
import os

plt.style.use(mplhep.style.CMS)

def preselection(events, score):
        
    mass_bool = ((events.mass > 100) & (events.mass < 180))
    dijet_mass_bool = ((events.nonResReg_dijet_mass_DNNreg > 70) & (events.nonResReg_dijet_mass_DNNreg < 190))

    lead_mvaID_bool = (events.lead_mvaID > -0.7)
    sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

    events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    score = score[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    return events, score


def plot_score_shape_diff_kl(folder, mhh_bounds=None):
    mHH_var = "nonResReg_M_X"

    save_folder = folder.split('/')[0]
    if len(folder.split('/')) == 4:
        save_folder += f"/{folder.split('/')[1]}"

    kl_sample_list = ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
                      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10",
                      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00", "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24"]

    era_list = ["2016preVFP", "2016postVFP", "2017", "2018", "preEE", "postEE", "preBPix", "postBPix", "2024"]
    # Edit these two lists to define your Run2/Run3 era grouping.
    run2_era_list = ["2016preVFP", "2016postVFP", "2017", "2018"]
    run3_era_list = ["preEE", "postEE", "preBPix", "postBPix", "2024"]

    era_group_dict = {
        "all": era_list,
        "run2": run2_era_list,
        "run3": run3_era_list,
    }
    
    if mhh_bounds and len(mhh_bounds) > 1:
        mhh_bins = [(mhh_bounds[i], mhh_bounds[i+1], f"{int(mhh_bounds[i])}-{int(mhh_bounds[i+1])}") for i in range(len(mhh_bounds)-1)]
        mhh_bins.append((mhh_bounds[-1], None, f"gt{int(mhh_bounds[-1])}"))
    else:
        mhh_bins = [(None, None, "")]
    
    legend_dict = {
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00': r"SM ($k_{\lambda}$=1)",
        'GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00': r"$k_{\lambda}$=0",
        'GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00': r"$k_{\lambda}$=2.45",
        'GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00': r"$k_{\lambda}$=5",
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00': r"$c_{2}$=3",
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35': r"$c_{2}$=0.35",
        'GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00': r"$k_{\lambda}$=0, $c_{2}$=1",
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10': r"$c_{2}$=0.1",
        'GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00': r"$c_{2}$=-2",
        'GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24': r"$k_{\lambda}$=-20, $c_{2}$=2.24"
    }

    # Create color map for 10 samples
    cmap = cm.get_cmap('tab10')
    colors = {sample: cmap(i) for i, sample in enumerate(kl_sample_list)}

    for mhh_min, mhh_max, bin_name in mhh_bins:
        for era_tag, selected_eras in era_group_dict.items():
            events_dict = {}

            for sample in kl_sample_list:
                events_list = []
                score_list = []

                for era in selected_eras:
                    if not os.path.exists(f"{folder}/{era}/{sample}/events.parquet") or not os.path.exists(f"{folder}/{era}/{sample}/y.npy"):
                        print(f"Warning: Missing data for {sample} in {era}. Skipping.")
                        continue

                    events = ak.from_parquet(f"{folder}/{era}/{sample}/events.parquet", columns=["weight_tot", "mass", mHH_var, "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
                    score = np.load(f"{folder}/{era}/{sample}/y.npy")

                    # Apply mHH binning if specified
                    if mhh_min is not None and mhh_max is not None:
                        mask = (events[mHH_var] >= mhh_min) & (events[mHH_var] < mhh_max)
                        events = events[mask]
                        score = score[mask]
                    elif mhh_min is not None and mhh_max is None:
                        mask = events[mHH_var] >= mhh_min
                        events = events[mask]
                        score = score[mask]

                    events_list.append(events)
                    score_list.append(score)

                if len(events_list) == 0:
                    print(f"Warning: No valid events found for {sample} in era group '{era_tag}'.")
                    continue

                events = ak.concatenate(events_list, axis=0)
                score = np.concatenate(score_list, axis=0)

                events, score = preselection(events, score)

                # Add score columns and discriminators
                score_names = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH"] #, "VBFToHH_sig"]
                for idx, name in enumerate(score_names):
                    events[f"{name}_score"] = score[:, idx]

                events["D_sig_vs_ttH"] = events.GluGluToHH_score / (events.GluGluToHH_score + events.ttH_score)
                events["D_sig_vs_nonres"] = events.GluGluToHH_score / (events.GluGluToHH_score + events.non_resonant_bkg_score + events.other_single_H_score)

                events_dict[sample] = events

            if len(events_dict) == 0:
                print(f"Warning: No plots produced for era group '{era_tag}' and mHH bin '{bin_name}'.")
                continue

            for plot_col in ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score", "D_sig_vs_ttH", "D_sig_vs_nonres"]:
                plt.subplots(figsize=(7, 6))
                for sample in events_dict:
                    y = events_dict[sample][plot_col].to_numpy()
                    mask = np.isfinite(y)
                    plt.hist(y[mask], bins=50, range=(0.9, 1), weights=events_dict[sample]['weight_tot'].to_numpy()[mask],
                             label=legend_dict[sample], histtype='step', density=True, linewidth=2, color=colors[sample])

                plt.xlabel(plot_col, fontsize=16)
                plt.ylabel("a.u.", fontsize=16)
                plt.legend(fontsize=16)
                plt.yscale('log')
                bin_suffix = f"_{bin_name}" if bin_name else ""
                era_suffix = "" if era_tag == "all" else f"_{era_tag}"
                out_dir = f"{save_folder}/score_shape_diff/0p9_to_1/"
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(f"{out_dir}/ggHH_score_dist_{plot_col}{era_suffix}{bin_suffix}.png", bbox_inches='tight')
                print(f"Saved plot: {out_dir}/ggHH_score_dist_{plot_col}{era_suffix}{bin_suffix}.png")
                plt.clf()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot score shape differences and discriminators')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the parquet files')
    parser.add_argument('--mhh-bin', type=float, nargs='*', help='mHH bin boundaries (e.g., 0 350 650). If not provided, uses all events.')
    args = parser.parse_args()

    plot_score_shape_diff_kl(args.folder, mhh_bounds=args.mhh_bin if args.mhh_bin else None)