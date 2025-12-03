import numpy as np
import os
import sys
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
    "BBHto2G_M_125": "BBHToGG",
    "GluGluHToGG_M_125": "GluGluHToGG",
    "VBFHToGG_M_125": "VBFHToGG",
    "VHtoGG_M_125": "VHToGG",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGluToHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGluToHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGluToHH_kl-5p00_kt-1p00_c2-0p00",
}

def load_samples(base_path, samples, data=False, syst=""):
    """Load predictions and weights, scaling weights by luminosity."""
    # Example MC file to get the weight columns
    parquet_file = pq.ParquetFile(base_path+"/individual_samples/preEE/ttHtoGG_M_125/"+syst+"/events.parquet")
    all_columns = parquet_file.schema.names
    # weight_columns = [col for col in all_columns if 'weight' in col]
    weight_columns = ["weight_tot"]
    dijet_mass_key = "nonResReg_dijet_mass_DNNreg"
    HH_mass_key = "nonResReg_HHbbggCandidate_mass"

    #"nonResReg_lead_bjet_hFlav", "nonResReg_sublead_bjet_hFlav", "event", "run", "lumi"]#, "is_boosted", "y_proba"] 
    columns = [
        "mass",
        dijet_mass_key,
        HH_mass_key,
        "eta",
        "lead_eta",
        "lead_phi",
        "lead_mvaID",
        "nonResReg_pholead_PtOverM",
        "sublead_eta",
        "sublead_phi",
        "sublead_mvaID",
        "nonResReg_phosublead_PtOverM",
        "nonResReg_lead_bjet_eta",
        "nonResReg_lead_bjet_phi",
        "nonResReg_lead_bjet_btagPNetB",
        "nonResReg_sublead_bjet_eta",
        "nonResReg_sublead_bjet_phi",
        "nonResReg_sublead_bjet_btagPNetB",
        "nonResReg_DeltaR_j1g1",
        "nonResReg_DeltaR_j2g1",
        "nonResReg_DeltaR_j1g2",
        "nonResReg_DeltaR_j2g2",
        "nonResReg_DeltaR_jg_min",
        "nonResReg_CosThetaStar_CS",
        "nonResReg_CosThetaStar_gg",
        "nonResReg_CosThetaStar_jj",
        "puppiMET_phi",
        "puppiMET_pt",
        "n_leptons",
        "n_jets",
        "nonResReg_chi_t0",
        "nonResReg_chi_t1",
        "nonResReg_DeltaPhi_j1MET",
        "nonResReg_DeltaPhi_j2MET",
        "VBF_first_jet_eta",
        "VBF_first_jet_phi",
        "VBF_second_jet_eta",
        "VBF_second_jet_phi",
        "VBF_first_jet_PtOverM",
        "VBF_second_jet_PtOverM",
        "VBF_jet_eta_prod",
        "VBF_jet_eta_diff",
        "VBF_jet_eta_sum",
        "VBF_DeltaR_j1b1",
        "VBF_DeltaR_j1b2",
        "VBF_DeltaR_j2b1",
        "VBF_DeltaR_j2b2",
        "VBF_DeltaR_j1g1",
        "VBF_DeltaR_j1g2",
        "VBF_DeltaR_j2g1",
        "VBF_DeltaR_j2g2",
        "VBF_DeltaR_jb_min",
        "VBF_DeltaR_jg_min",
        "VBF_dijet_mass",
        "nonResReg_HHbbggCandidate_eta",
        "diphoton_PtOverM_ggjj",
        "nonResReg_dijet_PtOverM_ggjj",
        "deltaR_gg",
        "nonResReg_lead_bjet_over_M_regressed",
        "nonResReg_sublead_bjet_over_M_regressed"
        ]

    samples_input = {
            "lumi": [],
            "event": [],
            "run": [],
            "nonResReg_lead_bjet_hFlav": [],
            "nonResReg_sublead_bjet_hFlav": [],
            "mass": [], 
            "dijet_mass": [], 
            "HHbbggCandidate_mass": [],
            "sample": [],
            "year": [],
            "score": [],
            "nonRes_score": [],
            "ttH_score": [],
            "singleH_score" :[],
            "ggHH_score":[],
            "is_boosted": [],
            "y_proba":[]
            "eta" : [],
            "lead_eta" : [],
            "lead_phi" : [],
            "lead_mvaID" : [],
            "nonResReg_pholead_PtOverM" : [],
            "sublead_eta" : [],
            "sublead_phi" : [],
            "sublead_mvaID" : [],
            "nonResReg_phosublead_PtOverM" : [],
            "nonResReg_lead_bjet_eta" : [],
            "nonResReg_lead_bjet_phi" : [],
            "nonResReg_lead_bjet_btagPNetB" : [],
            "nonResReg_sublead_bjet_eta" : [],
            "nonResReg_sublead_bjet_phi" : [],
            "nonResReg_sublead_bjet_btagPNetB" : [],
            "nonResReg_DeltaR_j1g1" : [],
            "nonResReg_DeltaR_j2g1" : [],
            "nonResReg_DeltaR_j1g2" : [],
            "nonResReg_DeltaR_j2g2" : [],
            "nonResReg_DeltaR_jg_min" : [],
            "nonResReg_CosThetaStar_CS" : [],
            "nonResReg_CosThetaStar_gg" : [],
            "nonResReg_CosThetaStar_jj" : [],
            "puppiMET_phi" : [],
            "puppiMET_pt" : [],
            "n_leptons" : [],
            "n_jets" : [],
            "nonResReg_chi_t0" : [],
            "nonResReg_chi_t1" : [],
            "nonResReg_DeltaPhi_j1MET" : [],
            "nonResReg_DeltaPhi_j2MET" : [],
            "VBF_first_jet_eta" : [],
            "VBF_first_jet_phi" : [],
            "VBF_second_jet_eta" : [],
            "VBF_second_jet_phi" : [],
            "VBF_first_jet_PtOverM" : [],
            "VBF_second_jet_PtOverM" : [],
            "VBF_jet_eta_prod" : [],
            "VBF_jet_eta_diff" : [],
            "VBF_jet_eta_sum" : [],
            "VBF_DeltaR_j1b1" : [],
            "VBF_DeltaR_j1b2" : [],
            "VBF_DeltaR_j2b1" : [],
            "VBF_DeltaR_j2b2" : [],
            "VBF_DeltaR_j1g1" : [],
            "VBF_DeltaR_j1g2" : [],
            "VBF_DeltaR_j2g1" : [],
            "VBF_DeltaR_j2g2" : [],
            "VBF_DeltaR_jb_min" : [],
            "VBF_DeltaR_jg_min" : [],
            "VBF_dijet_mass" : [],
            "nonResReg_HHbbggCandidate_eta" : [],
            "diphoton_PtOverM_ggjj" : [],
            "nonResReg_dijet_PtOverM_ggjj" : [],
            "deltaR_gg" : [],
            "nonResReg_lead_bjet_over_M_regressed" : [],
            "nonResReg_sublead_bjet_over_M_regressed" : [],
    }
    for weight in weight_columns:
        samples_input.update({weight: []})

    eras = ["preEE", "postEE", "preBPix", "postBPix"]
    if data:
        eras = ["2022_EraC","2022_EraD","2022_EraE","2022_EraF","2022_EraG","2023_EraC","2023_EraD"]

    for era in eras:
        print("###########")
        print(era)
        print("###########")
        print()
        for sample in samples:
            if (sample in ["GGJets", "DDQCDGJET", "TTGG", "TT", "TTG_10_100", "TTG_100_200", "TTG_200"]) and (syst != ""):
                continue
            
            if era != "postEE":
              if ("TTG_" in sample) or (sample == "TT"):
                continue
            if data:
                path = os.path.join(base_path, "individual_samples_data", era, sample)
            else:
                path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
            y_path = os.path.join(path, 'y.npy')
            w_path = os.path.join(path, 'rel_w.npy')
            events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=columns+weight_columns)  # Load events

            # Check if files exist
            if not (os.path.exists(y_path)):
                print(f"Missing y for {path}. Skipping.")
                continue
            y = np.load(y_path)
            samples_input["score"].append(y)
            
            # samples_input["lumi"].append(np.array(events['lumi']))
            # samples_input["event"].append(np.array(events['event']))
            # samples_input["run"].append(np.array(events['run']))

            # samples_input["nonResReg_lead_bjet_hFlav"].append(np.array(events['nonResReg_lead_bjet_hFlav']))
            # samples_input["nonResReg_sublead_bjet_hFlav"].append(np.array(events['nonResReg_sublead_bjet_hFlav']))

            # samples_input["mass"].append(np.array(events['mass']))
            samples_input["dijet_mass"].append(np.array(events[dijet_mass_key]))
            samples_input["HHbbggCandidate_mass"].append(np.array(events[HH_mass_key]))
            for col in columns:
                if (col != dijet_mass_key) & (col != HH_mass_key):
                    samples_input[col].append(np.array(events[col]))

            if sample == "":
                sample = "Data"
            if sample in ff_sampledict.keys():
                sample = ff_sampledict[sample]
            print(sample)
            samples_input["sample"].append(np.full(y.shape[0], sample))

            if "22" in era or "EE" in era:
                year = 2022
            elif "23" in era or "BPix" in era:
                year = 2023
            else:
                raise ValueError(f"Unknown era: {era}")
            samples_input["year"].append(np.full(y.shape[0], year))

            # samples_input["is_boosted"].append(np.array(events["is_boosted"]))  
            # samples_input["y_proba"].append(np.array(events['y_proba']))

            for weight in weight_columns:
                if weight in events.fields:
                    samples_input[weight].append(np.array(events[weight]))
                else:
                    samples_input[weight].append(np.array(ak.ones_like(events['mass'])))  # Default weight if not provided

    # Concatenate all data
    # samples_input["lumi"] = np.concatenate(samples_input["lumi"], axis=0)
    # samples_input["event"] = np.concatenate(samples_input["event"], axis=0)
    # samples_input["run"] = np.concatenate(samples_input["run"], axis=0)
    #samples_input["nonResReg_lead_bjet_hFlav"] = np.concatenate(samples_input["nonResReg_lead_bjet_hFlav"], axis=0)
    #samples_input["nonResReg_sublead_bjet_hFlav"] = np.concatenate(samples_input["nonResReg_sublead_bjet_hFlav"], axis=0)
    # samples_input["mass"] = np.concatenate(samples_input["mass"], axis=0)
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["HHbbggCandidate_mass"] = np.concatenate(samples_input["HHbbggCandidate_mass"], axis=0)

    for col in columns:
        if (col != dijet_mass_key) & (col != HH_mass_key):
            samples_input[col] = np.concatenate(samples_input[col], axis=0)

    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["year"] = np.concatenate(samples_input["year"], axis=0)
    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["score"] = [row for row in scores]
    samples_input["nonRes_score"] = [row[0] for row in scores]
    samples_input["ttH_score"] = [row[1] for row in scores]   
    samples_input["singleH_score"] = [row[2] for row in scores]
    samples_input["ggHH_score"] = [row[3] for row in scores]
    # samples_input["is_boosted"] = np.concatenate(samples_input["is_boosted"], axis=0)
    # samples_input["y_proba"] = np.concatenate(samples_input["y_proba"], axis=0)
    for weight in weight_columns:
        samples_input[weight] = np.concatenate(samples_input[weight], axis=0)

    # convert to pandas dataframe
    samples_input = pd.DataFrame(samples_input)

    return samples_input

if __name__ == "__main__":
    # Output multiclass folder, one folder up from individual_samples.
    base_path = sys.argv[1]
    print(base_path)

    samples = [
            "GGJets",
            "DDQCDGJET",
            "TTGG",
            # "TT",
            # "TTG_10_100",
            "TTG_100_200",
            "TTG_200",
            "ttHtoGG_M_125",
            "BBHto2G_M_125",
            "GluGluHToGG_M_125",
            "VBFHToGG_M_125",
            "VHtoGG_M_125",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
    ]

    systs = [
            "ScaleEB2G_IJazZ_down",
            "ScaleEB2G_IJazZ_up",
            "ScaleEE2G_IJazZ_down",
            "ScaleEE2G_IJazZ_up",
            "Smearing2G_IJazZ_down",
            "Smearing2G_IJazZ_up",
            "jec_syst_Total_down",
            "jec_syst_Total_up",
            "jer_syst_down",
            "jer_syst_up",
            "FNUF_down",
            "FNUF_up",
            "Material_down",
            "Material_up",
    ]
    
    merged_samples_MC = load_samples(base_path, samples)
    merged_samples_data = load_samples(base_path, [""] ,data=True)

    merged_samples = pd.concat([merged_samples_MC, merged_samples_data], ignore_index=True)
    merged_samples.to_parquet("merged_samples.parquet", engine='pyarrow')

    for syst in systs:
        print(syst)
        merged_samples_MC = load_samples(base_path, samples, syst=syst)
        merged_samples_MC.to_parquet("merged_samples_"+syst+".parquet", engine='pyarrow')
        print()
