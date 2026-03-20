import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep

plt.style.use(mplhep.style.CMS)

def preselection(events, score):
        
    mass_bool = ((events.mass > 100) & (events.mass < 180))
    dijet_mass_bool = ((events.nonResReg_dijet_mass_DNNreg > 70) & (events.nonResReg_dijet_mass_DNNreg < 190))

    lead_mvaID_bool = (events.lead_mvaID > -0.7)
    sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

    events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    score = score[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]

    return events, score

def plot_score_shape_diff_kl(folder):
    save_folder = folder.split('/')[0]
    if len(folder.split('/')) == 4:
        save_folder += f"/{folder.split('/')[1]}"

    kl_sample_list = ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
                      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10",
                      "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00", "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24"]

    events_dict = {}
    score_dict = {}
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

    for sample in kl_sample_list:
        preVFP = ak.from_parquet(f"{folder}/2016preVFP/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        y2017 = ak.from_parquet(f"{folder}/2017/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        y2018 = ak.from_parquet(f"{folder}/2018/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        preEE = ak.from_parquet(f"{folder}/preEE/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        postEE = ak.from_parquet(f"{folder}/postEE/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        preBPix = ak.from_parquet(f"{folder}/preBPix/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        postBPix = ak.from_parquet(f"{folder}/postBPix/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
        y2024 = ak.from_parquet(f"{folder}/2024/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])


        score_preVFP = np.load(f"{folder}/2016preVFP/{sample}/y.npy")
        score_2017 = np.load(f"{folder}/2017/{sample}/y.npy")
        score_2018 = np.load(f"{folder}/2018/{sample}/y.npy")
        score_preEE = np.load(f"{folder}/preEE/{sample}/y.npy")
        score_postEE = np.load(f"{folder}/postEE/{sample}/y.npy")
        score_preBPix = np.load(f"{folder}/preBPix/{sample}/y.npy")
        score_postBPix = np.load(f"{folder}/postBPix/{sample}/y.npy")
        score_2024 = np.load(f"{folder}/2024/{sample}/y.npy")

        if sample != "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
            postVFP = ak.from_parquet(f"{folder}/2016postVFP/{sample}/events.parquet", columns=["weight_tot", "mass", "nonResReg_dijet_mass_DNNreg", "lead_mvaID", "sublead_mvaID"])
            score_postVFP = np.load(f"{folder}/2016postVFP/{sample}/y.npy")

            events = ak.concatenate([preVFP, postVFP, y2017, y2018, preEE, postEE, preBPix, postBPix, y2024], axis=0)
            score = np.concatenate([score_preVFP, score_postVFP, score_2017, score_2018, score_preEE, score_postEE, score_preBPix, score_postBPix, score_2024], axis=0)
        else:
            events = ak.concatenate([preVFP, y2017, y2018, preEE, postEE, preBPix, postBPix, y2024], axis=0)
            score = np.concatenate([score_preVFP, score_2017, score_2018, score_preEE, score_postEE, score_preBPix, score_postBPix, score_2024], axis=0)

        # apply preselection
        events, score = preselection(events, score)

        events_dict[sample] = events
        score_dict[sample] = score

    score_names = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH", "VBFToHH_sig"]
    for score_idx in range(4):
        plt.subplots(figsize=(7, 6))
        for i, sample in enumerate(events_dict.keys()):
        
            y = score_dict[sample][:, score_idx]
            weight = events_dict[sample]['weight_tot'].to_numpy()
        
            plt.hist(y, bins=50, range=(0, 1), weights=weight, label=legend_dict[sample], histtype='step', density=True, linewidth=2)

        plt.xlabel(f"{score_names[score_idx]}_score", fontsize=16)
        plt.ylabel("a.u.", fontsize=16)
        plt.legend(fontsize=16)
        plt.yscale('log')
        plt.savefig(f"{save_folder}/ggHH_score_dist_{score_names[score_idx]}.png", bbox_inches='tight')
        plt.clf()

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot score shape differences using KL divergence')
    parser.add_argument('--folder', type=str, help='Path to the folder containing the parquet files')
    args = parser.parse_args()

    plot_score_shape_diff_kl(args.folder)