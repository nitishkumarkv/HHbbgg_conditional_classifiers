import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import pyarrow as pa
import yaml

if not hasattr(pa.lib, "PyExtensionType") and hasattr(pa.lib, "ExtensionType"):
    pa.lib.PyExtensionType = pa.lib.ExtensionType

# Apply CMS style
hep.style.use("CMS")


def plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, bins=40, mass_window=(120, 130), mjj_mass_window=(110, 140), signal_scale=100, only_MC=False):
    """
    Load data first, then loop over variables to plot stacked histograms with MC and Data, including ratio plots.

    Parameters:
        sim_folder (str): Path to the directory containing MC samples.
        data_folder (str): Path to the directory containing data samples.
        sim_samples (list): List of MC sample names.
        variables (list): List of variables to plot.
        bins (int or array-like): Number of bins or bin edges.
        mass_window (tuple): Mass range to blind (default: (120, 130) GeV).
        signal_scale (int): Scale factor for signal visualization.
    """
    # create output directory if it does not exist
    if only_MC:
        out_path = os.path.join(out_path, "MC_Plots_70_190")
    else:
        out_path = os.path.join(out_path, "Data_MC_Plots_70_190")
    os.makedirs(out_path, exist_ok=True)
    mc_colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "gold", "brown", "pink", "lime"]
    mc_colors = [
    "#FF8A50",  # Darker Peach
    "#FFB300",  # Golden Yellow
    "#66BB6A",  # Rich Green
    # "#42A5F5",  # Deeper Sky Blue
    "#AB47BC",  # Strong Lavender Purple
    # "#EC407A",  # Deeper Pink
    "#C0CA33",  # Darker Lime
    "#26A69A",  # Deep Teal
    "#1976D2",  # Lighter Blue
    "#EF5350",  # Lighter Red
    # "blue",  # Deep Brown
    # "red",  # Vibrant Orange
    "#795548",  # Deep Brown
    "#757575",  # Medium Gray
    "#66BB6A",  # Light Green
    ]

    label_dict = {
        "GGJets": "GGJets",
        "GJetPt20To40": "GJetPt20To40",
        "GJetPt40": "GJetPt40",
        "TTGG": "TTGG",
        "ttHtoGG_M_125": "ttH",
        "BBHto2G_M_125": "bbH",
        "GluGluHToGG_M_125": "ggH",
        "VBFHToGG_M_125": "VBFH",
        "VHtoGG_M_125": "VH",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "ggHH SM",
        "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "ggHH kl=5.00",
        "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "ggHH kl=0.00",
        "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "ggHH kl=2.45",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10": "ggHH c2=0.10",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35": "ggHH c2=0.35",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00": "ggHH c2=3.00",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00": "ggHH c2=-2.00",
        "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00": "ggHH kl=0.00, c2=1.00",
        "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24": "ggHH kl=-20.00, c2=2.24",
        "VBFHH_CV_1_C2V_1_C3_1": "VBFHH SM",
        "VBFHH_CV_1_C2V_0_C3_1": "VBFHH C2V=0",
        "VBFHH_CV_1p74_C2V_1p37_C3_14p4": "VBFHH CV=1.74, C2V=1.37, C3=14.4",
        "VBFHH_CV_2p12_C2V_3p87_C3_m5p96": "VBFHH CV=2.12, C2V=3.87, C3=-5.96",
        "VBFHH_CV_m0p012_C2V_0p030_C3_10p2": "VBFHH CV=-0.012, C2V=0.030, C3=10.2",
        "VBFHH_CV_m0p758_C2V_1p44_C3_m19p3": "VBFHH CV=-0.758, C2V=1.44, C3=-19.3",
        "VBFHH_CV_m0p962_C2V_0p959_C3_m1p43": "VBFHH CV=-0.962, C2V=0.959, C3=-1.43",
        "VBFHH_CV_m1p21_C2V_1p94_C3_m0p94": "VBFHH CV=-1.21, C2V=1.94, C3=-0.94",
        "VBFHH_CV_m1p60_C2V_2p72_C3_m1p36": "VBFHH CV=-1.60, C2V=2.72, C3=-1.36",
        "VBFHH_CV_m1p83_C2V_3p57_C3_m3p39": "VBFHH CV=-1.83, C2V=3.57, C3=-3.39",

        "VBFHHto2B2G_CV_1_C2V_1_C3_1": "VBFHH",
        "DDQCDGJET": "DDQCDGJets",
        "TTG_10_100": "TTG_10_100",
        "TTG_100_200": "TTG_100_200",
        "TTG_200": "TTG_200",
        "TT": "TT",
    }

    # Load MC and Signal Data First
    stack_mc_dict = {}
    signal_mc_dict = {}

    def deltaR(eta1, phi1, eta2, phi2, fill_none=True):
        eta1 = ak.mask(eta1, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        phi1 = ak.mask(phi1, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        eta2 = ak.mask(eta2, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))
        phi2 = ak.mask(phi2, (eta1 != -999) & (phi1 != -999) & (eta2 != -999) & (phi2 != -999))

        dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
        deta = eta1 - eta2
        delta_r = np.sqrt(deta**2 + dphi**2)

        if fill_none:
            return ak.fill_none(delta_r, -999.0)
        else:
            return delta_r
        
    def add_var(events, era):
        print(f"era:{era}")

        # events["diphoton_PtOverM_ggjj"] = events.pt / events.nonResReg_HHbbggCandidate_mass
        # events["nonResReg_dijet_PtOverM_ggjj"] = events.nonResReg_dijet_pt / events.nonResReg_HHbbggCandidate_mass

        # events["diphoton_PtOverM_X"] = events.pt / events.nonResReg_vbfpair_M_X
        # events["nonResReg_dijet_PtOverM_X"] = events.nonResReg_dijet_pt / events.nonResReg_vbfpair_M_X

        events["nonResReg_lead_bjet_over_M_regressed"] = events.nonResReg_vbfpair_lead_bjet_pt / events.nonResReg_vbfpair_dijet_mass
        events["nonResReg_sublead_bjet_over_M_regressed"] = events.nonResReg_vbfpair_sublead_bjet_pt / events.nonResReg_vbfpair_dijet_mass

        # add deltaR between lead and sublead photon
        # events["deltaR_gg"] = self.deltaR(events.lead_eta, events.lead_phi, events.sublead_eta, events.sublead_phi)

        btagVariable = "btag"
        # Use PNetB for NanoAODv12/v13 
        if era == "preEE":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.961, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.047, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.245, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6734, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7862, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.961, int)
        elif era == "postEE":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9664, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0499, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.2605, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6915, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.8033, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9664, int)
        elif era == "preBPix":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9659, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0358, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.1917, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6172, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7515, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9659, int)
        elif era == "postBPix":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagPNetB"] > 0.9688, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.0359, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.1919, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.6133, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.7544, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagPNetB"] > 0.9688, int)
        # Use UParT for NanoAODv15
        elif era == "2024" or era == "2025":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9739, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0246, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1272, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.4648, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6298, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9739, int)
        elif era == "2016preVFP":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0387, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1847, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5467, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6777, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9218, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0387, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1847, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5467, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6777, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9218, int)
        elif era == "2016postVFP":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0400, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1898, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5538, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6872, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9353, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0400, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1898, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5538, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6872, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9353, int)
        elif era == "2017":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0331, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1776, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5755, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.7274, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9666, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0331, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1776, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5755, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.7274, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9666, int)
        elif era == "2018":
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.0308, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.1610, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.5405, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.6992, int)
            events["nonResReg_vbfpair_lead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_lead_bjet_btagUParTAK4B"] > 0.9655, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_L"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.0308, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_M"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.1610, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_T"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.5405, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.6992, int)
            events["nonResReg_vbfpair_sublead_bjet_"+btagVariable+"_WP_XXT"] = ak.values_astype(events["nonResReg_vbfpair_sublead_bjet_btagUParTAK4B"] > 0.9655, int)
        else:
            raise ValueError(f"Era '{era}' not recognized for b-tagging WP assignment")

        return events


    def add_preselection(events):
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        #dijet_mass_bool = ((events.Res_mjj_regressed > 80) & (events.Res_mjj_regressed < 180))
        dijet_mass_bool = ((events.nonResReg_vbfpair_dijet_mass > 70) & (events.nonResReg_vbfpair_dijet_mass < 190))

        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        events = events[mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool]
        #events = events[mass_bool & dijet_mass_bool]

        return events

    class_names = ["non_resonant_bkg_score", "ttH_score", "other_single_H_score", "GluGluToHH_score"] #, "VBFToHH_sig_score"]
    events_path = samples_info["samples_path"]

    # Load Data First
    data_combined = None

    data_samples = training_config["samples_info"]["data"]

    sample_to_era = {
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
    
    for data_sample, path in data_samples.items():
        if os.path.exists(f"{data_folder}/{data_sample}/events.parquet"):
            print(f"Loading data from {data_folder}/{data_sample}/events.parquet")
            data_part = ak.from_parquet(f"{data_folder}/{data_sample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"])
        else:
            data_part = ak.from_parquet(f"{events_path}/{path}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"])
        print(f"Loading data scores from {data_folder}/{data_sample}/y.npy")
        if os.path.exists(f"{data_folder}/{data_sample}/y.npy"):
            data_score = np.load(f"{data_folder}/{data_sample}/y.npy")
            num_classes = data_score.shape[1]
            for i, class_name in enumerate(class_names):
                if i < num_classes:
                    data_part[class_name] = data_score[:, i]

            # # we don't need the add_var as they're already added when preparing inputs
            # data_part = add_var(data_part, sample_to_era.get(data_sample, "2024"))

        if data_combined is None:
            data_combined = data_part
        else:
            data_combined = ak.concatenate([data_combined, data_part], axis=0)

    if "minMVAID" in variables:
        data_combined["minMVAID"] = np.min([data_combined.lead_mvaID, data_combined.sublead_mvaID], axis = 0)
        data_combined["maxMVAID"] = np.max([data_combined.lead_mvaID, data_combined.sublead_mvaID], axis = 0)

    # Apply preselection
    data_combined = add_preselection(data_combined)
    print("before: ", len(data_combined))
    
    # get number of data events in sideband
    int_data_sideband = len(data_combined)

    eras = samples_info["eras"]
    print("eras: ", eras)
    for sample in sim_samples:
        sample_combined = []
        for era in eras:
            if (era == "2024" or era == "2025") & (sample == "VHtoGG_M_125"):
                VHsample = "WmHtoGG"
                if os.path.exists(f"{sim_folder}/{era}/{VHsample}/events.parquet"):
                    events_ = ak.from_parquet(f"{sim_folder}/{era}/{VHsample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])
                else:
                    events_ = ak.from_parquet(f"{events_path}/{samples_info[era][VHsample]}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])
                for VHsample in ["WpHtoGG", "ZHtoGG"]:
                    if os.path.exists(f"{sim_folder}/{era}/{VHsample}/events.parquet"):
                        events_vh = ak.from_parquet(f"{sim_folder}/{era}/{VHsample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])
                    else:
                        events_vh = ak.from_parquet(f"{events_path}/{samples_info[era][VHsample]}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])

                    events_ = ak.concatenate([events_, events_vh])
            else:
                if os.path.exists(f"{sim_folder}/{era}/{sample}/events.parquet"):
                    events_ = ak.from_parquet(f"{sim_folder}/{era}/{sample}/events.parquet", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])
                elif sample in samples_info[era]:
                    events_ = ak.from_parquet(f"{events_path}/{samples_info[era][sample]}", columns=variables+["lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_genPartFlav", "sublead_genPartFlav", "weight_tot"])
                else:
                    print(f"File not found for sample {sample} in era {era}. Skipping.")
                    continue

            if (era == "2024" or era == "2025") & (sample == "VHtoGG_M_125"):
                scores_ = np.load(f"{sim_folder}/{era}/WmHtoGG/y.npy")
                for VHsample in ["WpHtoGG", "ZHtoGG"]:
                    scores_vh = np.load(f"{sim_folder}/{era}/{VHsample}/y.npy")
                    scores_ = np.concatenate((scores_, scores_vh))
            else:
                print(f"Loading scores from {sim_folder}/{era}/{sample}/y.npy")
                scores_ = np.load(f"{sim_folder}/{era}/{sample}/y.npy")
            # select prompt photons for TTG and TT samples
            #if (("TTG_" in sample) or (sample == "TT")):
            #    print("selecting prompt photons for TTG and TT samples")
            #    prompt_photon_bool = ((events_.lead_genPartFlav == 1) | (events_.sublead_genPartFlav == 1))
            #    events_ = events_[prompt_photon_bool]
            #    rel_w_ = rel_w_[prompt_photon_bool]
            #    scores_ = scores_[prompt_photon_bool]

            #events_["weight_tot"] = rel_w_
            for i, class_name in enumerate(class_names):
                if i < scores_.shape[1]:
                    num_classes = scores_.shape[1]
                    events_[class_name] = scores_[:, i]
            sample_combined.append(events_)
        sample_combined = ak.concatenate(sample_combined, axis=0)

        if "minMVAID" in variables:
            sample_combined["minMVAID"] = np.min([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)
            sample_combined["maxMVAID"] = np.max([sample_combined.lead_mvaID, sample_combined.sublead_mvaID], axis = 0)

        # Apply preselection
        sample_combined = add_preselection(sample_combined)
        if sample == "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
            print("number of events in GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", sum(sample_combined["weight_tot"]))


        # Separate signal from background
        if sample in ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00", "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24"]:
            signal_mc_dict[label_dict[sample]] = sample_combined
        else:
            stack_mc_dict[label_dict[sample]] = sample_combined


    variables = class_names + variables

    var_config = {
        "mass": {"label": r"$m_{\gamma\gamma}$ [GeV]", "bins": 40, "range": (100, 180), "log": True},
        "nonRes_dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "dijet_mass": {"label": r"$m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonRes_mjj_regressed": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_vbfpair_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_DNNpair_dijet_mass": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "nonResReg_DNNpair_dijet_mass_DNNreg": {"label": r"$m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "Res_mjj_regressed": {"label": r"Resonant $m_{jj}^{reg}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "Res_dijet_mass": {"label": r" Resonant $m_{jj}$ [GeV]", "bins": 30, "range": (80, 180), "log": True},
        "non_resonant_bkg_score": {"label": "non_resonant_bkg_score", "bins": 30, "range": (0, 1), "log": True},
        "ttH_score": {"label": "ttH_score", "bins": 30, "range": (0, 1), "log": True},
        "other_single_H_score": {"label": "other_single_H_score", "bins": 30, "range": (0, 1), "log": True},
        "GluGluToHH_score": {"label": "GluGluToHH_score", "bins": 30, "range": (0, 1), "log": True},
        "VBFToHH_sig_score": {"label": "VBFToHH_sig_score", "bins": 30, "range": (0, 1), "log": True},
        "minMVAID": {"label": "minMVAID", "bins": 30, "range": (-0.7, 1), "log": True},
        "maxMVAID": {"label": "maxMVAID", "bins": 30, "range": (-0.7, 1), "log": True},
        "n_jets": {"label": "n_jets", "bins": 10, "range": (0, 10), "log": False},
        "sublead_eta": {"label": "sublead_eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "lead_eta": {"label": "lead_eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "sublead_pt": {"label": "sublead_pt [GeV]", "bins": 30, "range": (0, 200), "log": True},
        "lead_pt": {"label": "lead_pt [GeV]", "bins": 30, "range": (0, 200), "log": True},
        "pt": {"label": "Diphoton pt [GeV]", "bins": 30, "range": (0, 400), "log": True},
        "eta": {"label": "Diphoton eta", "bins": 30, "range": (-3.2, 3.2), "log": False},
        "lead_mvaID": {"label": "lead_mvaID", "bins": 30, "range": (-0.7, 1), "log": True},
        "sublead_mvaID": {"label": "sublead_mvaID", "bins": 30, "range": (-0.7, 1), "log": True},
        "nonResReg_chi_t0": {"label": "nonResReg_chi_t0", "bins": 30, "range": (0, 1000), "log": False},
        "nonResReg_chi_t1": {"label": "nonResReg_chi_t1", "bins": 30, "range": (0, 1000), "log": False}
    }

    # Loop Over Variables and Create Plots
    for variable in variables:
        if variable not in data_combined.fields:
            print(f"Variable {variable} not found in data fields. Skipping...")
            continue
        #print(f"Processing variable: {variable}")

        if variable not in var_config.keys():
            #get the min and max of the variable
            min_ = 0
            max_ = 0
            for i, (sample, data) in enumerate(stack_mc_dict.items()):
                ak_min = ak.min(data[variable])
                if ak_min != -999:
                    min_ = min(min_, ak.min(data[variable]))
                max_ = max(max_, ak.max(data[variable]))
            
            # check if a list of values is in the variable
            keywords = ["pt", "mass", "btag"]
            if any(key in variable for key in keywords):
            #if "pt" in variable:
                var_config[variable] = {"label": variable, "bins": 30, "range": (min_, max_), "log": True}
            else:
                var_config[variable] = {"label": variable, "bins": 30, "range": (min_, max_), "log": False}

        # Histogram binning
        bin_edges = np.linspace(*var_config[variable]["range"], var_config[variable]["bins"] + 1)



        # Compute MC histograms with weights
        mc_hist = []
        mc_err = np.zeros(len(bin_edges) - 1)
        mc_labels = []
        mc_colors_used = []

        for i, (sample, data) in enumerate(stack_mc_dict.items()):
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            print(variable, "sample: ", sample, "hist sum : ", sum(hist))
            mc_hist.append(hist)
            mc_labels.append(sample)
            mc_colors_used.append(mc_colors[i % len(mc_colors)])

            # Sum of squared weights for uncertainty calculation
            hist_err, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"]))**2)
            mc_err += hist_err

        # compute counts for

        mc_total = np.sum(mc_hist, axis=0)
        mc_err = np.sqrt(mc_err)  # Statistical uncertainty

        data_hist, _ = np.histogram(ak.to_numpy((data_combined[variable])), bins=bin_edges)
        data_err = np.sqrt(data_hist)  # Poisson errors

        signal_color_list = ["red", "green", "blue", "purple", "orange", "cyan", "magenta", "yellow", "brown", "pink"]
        # Compute signal histograms with weights
        signal_histograms = {}
        for signal, data in signal_mc_dict.items():
            hist, _ = np.histogram(ak.to_numpy((data[variable])), bins=bin_edges, weights=ak.to_numpy((data["weight_tot"])))
            if "ggHH" in signal:
                signal_histograms[signal] = hist * signal_scale
            else:
                signal_histograms[signal] = hist * signal_scale * 10

        # Plot
        if only_MC:
            fig, ax = plt.subplots(figsize=(10, 10))
        else: 
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, figsize=(10, 10), sharex=True )
            ax, ax_ratio = axs

        luminosities = {
        "2016preVFP": 19.5,
        "2016postVFP": 16.8,
        "2017": 42.07,
        "2018": 59.56,
        "preEE": 7.98,  # Integrated luminosity for preEE in fb^-1
        "postEE": 26.67,  # Integrated luminosity for postEE in fb^-1
        "preBPix": 18.06,  # Integrated luminosity for preEE in fb^-1
        "postBPix": 9.89,  # Integrated luminosity for postEE in fb^-1
        "2024": 108.82,
        "2025": 110.58
        }

        run2_eras = {"2016preVFP", "2016postVFP", "2017", "2018"}
        run3_eras = {"preEE", "postEE", "preBPix", "postBPix", "2024", "2025"}

        eras_present = set(eras)
        has_run2 = any(era in run2_eras for era in eras_present)
        has_run3 = any(era in run3_eras for era in eras_present)

        lumi_run2 = 0.0
        lumi_run3 = 0.0
        for era in eras:
            if era not in luminosities:
                continue
            if era in run2_eras:
                lumi_run2 += luminosities[era]
            elif era in run3_eras:
                lumi_run3 += luminosities[era]

        if has_run2 and has_run3:
            lumi_label = f"{lumi_run2:.2f} / {lumi_run3:.2f}"
            com_label = "13 / 13.6"
        elif has_run2:
            lumi_label = round(lumi_run2, 2)
            com_label = 13
        else:
            lumi_label = round(lumi_run3, 2)
            com_label = 13.6

        # set luminosity, CMS label, and legend
        hep.cms.label(data=True, lumi=lumi_label, ax=ax, loc=0, fontsize=16, label="Private Work", com=com_label)
        
        # Stacked MC histograms
        hep.histplot(
            mc_hist,
            bin_edges,
            histtype="fill",
            stack=True,
            label=mc_labels,
            color=mc_colors_used,
            edgecolor="black",
            ax=ax,
            #alpha=0.7
        )

        ax.fill_between(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            mc_total - mc_err,
            mc_total + mc_err,
            color="gray",
            alpha=0.5,
            step="mid",
            #label="MC Stat. Unc."
        )

        # Data points
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

        # Plot signal as step histogram
        color_idx = 0
        for signal, hist in signal_histograms.items():

            if "ggHH" in signal:
                ax.step(
                    (bin_edges[:-1] + bin_edges[1:]) / 2,
                    hist,
                    where="mid",
                    linestyle="dashed",
                    color=signal_color_list[color_idx],
                    label=f"{signal} x {signal_scale}"
                )
                color_idx += 1
            
            else:
                signal_scale_ = signal_scale*10
                ax.step(
                    (bin_edges[:-1] + bin_edges[1:]) / 2,
                    hist,
                    where="mid",
                    linestyle="dashed",
                    color="yellow",
                    label=f"{signal} x {signal_scale_}"
                )

        ax.legend(fontsize=14, ncol=2)

        # Labels
        ax.set_ylabel("Events")
        ax.set_xlim(var_config[variable]["range"])

        if var_config[variable]["log"]:
            ax.set_yscale("log")
            ax.set_ylim(0.1, 600 * np.max(data_hist))
        else:
            ax.set_ylim(0, 1.7 * np.max(data_hist))
        # Ratio plot (Data / MC)
        if not only_MC:

            ratio = abs(data_hist / mc_total)
            data_ratio_err = abs(data_err / mc_total)
            mc_ratio_err =abs( mc_err / mc_total)

            ax_ratio.errorbar(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                ratio,
                yerr=data_ratio_err,
                fmt="o",
                color="black",
                markersize=5,
            )

            ax_ratio.fill_between(
                (bin_edges[:-1] + bin_edges[1:]) / 2,
                1 - mc_ratio_err,
                1 + mc_ratio_err,
                color="gray",
                alpha=0.3,  # Keep it slightly transparent
                hatch="xx",  # Adds cross-hatched pattern
                edgecolor="black",  # Ensures visibility of hatch
                linewidth=0.0,  # Removes additional border
                step="mid",
            )

            ax_ratio.axhline(1, linestyle="dashed", color="gray")
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_ylabel("Data / MC")
            ax_ratio.set_xlabel(var_config[variable]["label"])
            plt.savefig(f"{out_path}/{variable}.png", dpi=300, bbox_inches="tight")
            plt.clf()
        else:
            ax.set_xlabel(var_config[variable]["label"])
            plt.savefig(f"{out_path}/{variable}.png", dpi=300, bbox_inches="tight")
            plt.clf()
        
        
    print("output saved in ", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot stacked histograms for MC and Data.")
    parser.add_argument("--base-path", type=str, required=True, help="Path to the base directory containing MC and Data folders.")
    parser.add_argument("--training_config_path", type=str, required=True, help="Path to the training config file.")
    args = parser.parse_args()

    # Load training configuration
    with open(f"{args.training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)
    
    samples_info = training_config["samples_info"]

    base_path = args.base_path
    sim_folder = f"{base_path}/individual_samples"
    data_folder = f"{base_path}/individual_samples_data"

    # sim_samples = ["GGJets", "DDQCDGJET", "TTGG", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "VBFHToGG_M_125", "VHtoGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1"]
    # sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "VBFHHto2B2G_CV_1_C2V_1_C3_1", "TTGG", "GGJets", "DDQCDGJET"]
    #sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET"]
    #sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET", "TTG_10_100", "TTG_100_200", "TTG_200", "TT"]
    # sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET"]#, "TTG_100_200", "TTG_200"]
    sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "GluGluHToGG_M_125", "TTGG", "GGJets", "DDQCDGJET"] # no bbHto2G
    # sim_samples = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "TTGG", "GGJets", "DDQCDGJET", "TTG_100_200", "TTG_200"]
    variables_ = ["Res_mjj_regressed", "Res_dijet_mass", "nonRes_mjj_regressed", "mass", "nonRes_dijet_mass", "minMVAID", "maxMVAID", "n_jets", "sublead_eta", "lead_eta", "sublead_pt", "lead_pt", "pt", "eta", "lead_mvaID", "sublead_mvaID", "nonResReg_dijet_mass_DNNreg", "nonResReg_vbfpair_dijet_mass", "nonResReg_HHbbggCandidate_mass", "nonResReg_M_X", "nonResReg_vbfpair_M_X", "nonResReg_dijet_pt", "nonResReg_lead_bjet_eta", "nonResReg_sublead_bjet_eta", "nonResReg_lead_bjet_pt", "nonResReg_sublead_bjet_pt"]
    extra_vars = ["mass", "nonRes_dijet_mass", "Res_dijet_mass", "weight", "pt", "nonRes_dijet_pt", "Res_dijet_pt", "Res_lead_bjet_pt", "Res_sublead_bjet_pt", "Res_lead_bjet_ptPNetCorr", "Res_sublead_bjet_ptPNetCorr", "nonRes_HHbbggCandidate_mass", "Res_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_mjj_regressed", "Res_mjj_regressed", "nonRes_lead_bjet_ptPNetCorr", "nonRes_sublead_bjet_ptPNetCorr", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "jet1_mass", "jet2_mass", "jet3_mass", "jet4_mass", "jet5_mass", "jet6_mass", "Res_lead_bjet_jet_idx", "Res_sublead_bjet_jet_idx", "jet1_index", "jet2_index", "jet3_index", "jet4_index", "jet5_index", "jet6_index",
                            "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "jet5_pt", "jet6_pt", "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta", "jet5_eta", "jet6_eta", "jet1_phi", "jet2_phi", "jet3_phi", "jet4_phi", "jet5_phi", "jet6_phi", "nonResReg_vbfpair_lead_bjet_btagPNetB", "nonResReg_vbfpair_sublead_bjet_btagPNetB"]

    # variables_ = ["mass", "nonRes_dijet_mass", "nonResReg_dijet_mass", "nonResReg_dijet_mass_DNNreg", "nonResReg_DNNpair_dijet_mass", "nonResReg_DNNpair_dijet_mass_DNNreg", "pt", "nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass", "eta", "nBTight","nBMedium","nBLoose", "nonRes_lead_bjet_pt", "nonRes_sublead_bjet_pt", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE", "lead_mvaID", "sublead_mvaID", "lead_eta", "lead_phi", "sublead_eta", "sublead_phi"]

    for BSM_sample in ["GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p10", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p35", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_3p00", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_m2p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_1p00", "GluGlutoHHto2B2G_kl_m20p00_kt_1p00_c2_2p24"]:
        if BSM_sample in training_config["sample_to_class"].keys():
            sim_samples.append(BSM_sample)    

    input_vars_path = args.training_config_path.replace("training_config.yaml", "input_variables.yaml")
    with open(input_vars_path, "r") as f:
        input_vars = yaml.safe_load(f)
    
    variables = variables_ + extra_vars + input_vars["mlp"]["vars"]
    # remove duplicate variables in this
    variables = list(set(variables))


    out_path = f"{base_path}/"
    plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000)
    # plot_stacked_histogram(samples_info, sim_folder, data_folder, sim_samples, variables, out_path, signal_scale=1000, only_MC=True)
