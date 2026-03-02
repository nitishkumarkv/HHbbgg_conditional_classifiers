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
    "WmHtoGG": "VHToGG",
    "WpHtoGG": "VHToGG",
    "ZHtoGG": "VHToGG",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGluToHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGluToHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGluToHH_kl-5p00_kt-1p00_c2-0p00",
}

def load_samples(base_path, eras, samples, data=False, syst=""):
    """Load predictions and weights, scaling weights by luminosity."""
    # Example MC file to get the weight columns
    parquet_file = pq.ParquetFile(base_path+"/individual_samples/preEE/ttHtoGG_M_125/"+syst+"/events.parquet")
    all_columns = parquet_file.schema.names
    # weight_columns = [col for col in all_columns if 'weight' in col]
    weight_columns = ["weight_tot", "weight"]
    dijet_mass_key = "nonResReg_dijet_mass_DNNreg"
    HH_mass_key = "nonResReg_vbfpair_HHbbggCandidate_mass"

    #"nonResReg_lead_bjet_hFlav", "nonResReg_sublead_bjet_hFlav", "event", "run", "lumi"]#, "is_boosted", "y_proba"] 
    columns = [
        "mass",
        dijet_mass_key,
        HH_mass_key,
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
        "nonResReg_vbfpair_pholead_PtOverM",
        "nonResReg_vbfpair_phosublead_PtOverM",
        "nonResReg_vbfpair_FirstJet_PtOverM",
        "nonResReg_vbfpair_SecondJet_PtOverM",
        "nonResReg_vbfpair_VBF_first_jet_btagPNetB",
        "nonResReg_vbfpair_VBF_second_jet_btagPNetB",
        "nonResReg_vbfpair_VBF_first_jet_btagPNetQvG",
        "nonResReg_vbfpair_VBF_second_jet_btagPNetQvG",
        "nonResReg_vbfpair_CosThetaStar_CS",
        "nonResReg_vbfpair_CosThetaStar_gg",
        "nonResReg_vbfpair_CosThetaStar_jj",
        "nonResReg_vbfpair_M_X",
        "nonResReg_vbfpair_HHbbggCandidate_pt",
        "nonResReg_vbfpair_VBF_first_jet_PtOverM",
        "nonResReg_vbfpair_VBF_second_jet_PtOverM",
        "nonResReg_vbfpair_VBF_jet_eta_prod",
        "nonResReg_vbfpair_VBF_jet_eta_diff",
        "nonResReg_vbfpair_VBF_jet_eta_sum",
        "nonResReg_vbfpair_VBF_DeltaR_jb_min",
        "nonResReg_vbfpair_VBF_DeltaR_jg_min",
        "nonResReg_vbfpair_VBF_Cgg",
        "nonResReg_vbfpair_VBF_Cbb",
        "nonResReg_vbfpair_VBF_dijet_mass",
        "nonResReg_vbfpair_VBF_dijet_vbfpair_Score_jj",
        "nonResReg_vbfpair_lead_bjet_btagPNetB",
        "nonResReg_vbfpair_sublead_bjet_btagPNetB",
        "nonResReg_vbfpair_lead_bjet_eta",
        "nonResReg_vbfpair_sublead_bjet_eta",
        "nonResReg_vbfpair_DeltaR_jg_min",
        "nonResReg_vbfpair_dijet_mass",
        "nonResReg_CosThetaStar_CS",
        "nonResReg_HHbbggCandidate_eta",
        "nonResReg_HHbbggCandidate_pt",
        "nonResReg_M_X"
        ]

    columns_gen = ["lead_genPartFlav", "sublead_genPartFlav"]

    samples_input = {
        "dijet_mass": [],
        "HHbbggCandidate_mass": [],
        "sample": [],
        "year": [],
        "era": [],
        "score": [],
        "nonRes_score": [],
        "ttH_score": [],
        "singleH_score" :[],
        "ggHH_score":[]
    }
    
    for col in columns + columns_gen:
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

            if (sample == "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00") & (era == "2016postVFP"):
                continue
            elif (sample == "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00") & (era == "2017"):
                continue

            if (era == "2024") & (sample == "VHtoGG_M_125"):
                VHsample = "WmHtoGG"
                path_VH = os.path.join(base_path, "individual_samples"+"/", era, VHsample, syst)
                y_path_VH = os.path.join(path_VH, 'y.npy')
                w_path_VH = os.path.join(path_VH, 'rel_w.npy')
                events = ak.from_parquet(os.path.join(path_VH, 'events.parquet'), columns=columns+weight_columns+columns_gen)
                y = np.load(y_path_VH)

                for VHsample in ["WpHtoGG", "ZHtoGG"]:
                    path_VH = os.path.join(base_path, "individual_samples"+"/", era, VHsample, syst)
                    y_path_VH = os.path.join(path_VH, 'y.npy')
                    w_path_VH = os.path.join(path_VH, 'rel_w.npy')
                    events_VH = ak.from_parquet(os.path.join(path_VH, 'events.parquet'), columns=columns+weight_columns+columns_gen)
                    y_VH = np.load(y_path_VH)

                    events = ak.concatenate([events, events_VH])
                    y = np.concatenate([y, y_VH])
                
            else:
                if era != "postEE":
                    if ("TTG_" in sample) or (sample == "TT"):
                        continue
                if data:
                    path = os.path.join(base_path, "individual_samples_data", era, sample)
                    y_path = os.path.join(path, 'y.npy')
                    w_path = os.path.join(path, 'rel_w.npy')
                    events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=columns+weight_columns)
                else:
                    if sample == "DDQCDGJET":
                        path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
                        y_path = os.path.join(path, 'y.npy')
                        w_path = os.path.join(path, 'rel_w.npy')
                        events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=columns+weight_columns)
                    else:
                        path = os.path.join(base_path, "individual_samples"+"/", era, sample, syst)
                        y_path = os.path.join(path, 'y.npy')
                        w_path = os.path.join(path, 'rel_w.npy')
                        events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=columns+weight_columns+columns_gen)
                
                # Check if files exist
                if not (os.path.exists(y_path)):
                    print(f"Missing y for {path}. Skipping.")
                    continue
                y = np.load(y_path)

            samples_input["score"].append(y)
            
            # samples_input["lumi"].append(np.array(events['lumi']))
            # samples_input["event"].append(np.array(events['event']))
            # samples_input["run"].append(np.array(events['run']))

            samples_input["dijet_mass"].append(np.array(events[dijet_mass_key]))
            samples_input["HHbbggCandidate_mass"].append(np.array(events[HH_mass_key]))
            for col in columns + columns_gen:
                if (col != dijet_mass_key) & (col != HH_mass_key):
                    if not(("gen" in col) & ((sample == "DDQCDGJET") | (data))):
                        samples_input[col].append(np.array(events[col]))

            if (data | (sample == "DDQCDGJET")):
                samples_input["lead_genPartFlav"].append(np.array([-999] * len(events[dijet_mass_key])))
                samples_input["sublead_genPartFlav"].append(np.array([-999] * len(events[dijet_mass_key])))

            if sample == "":
                sample = "Data"
            if sample in ff_sampledict.keys():
                sample = ff_sampledict[sample]
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

    for col in columns + columns_gen:
        if (col != dijet_mass_key) & (col != HH_mass_key):
            samples_input[col] = np.concatenate(samples_input[col], axis=0)

    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["year"] = np.concatenate(samples_input["year"], axis=0)
    samples_input["era"] = np.concatenate(samples_input["era"], axis=0)
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
            # "TTG_100_200",
            # "TTG_200",
            "ttHtoGG_M_125",
            # "BBHto2G_M_125",
            "GluGluHToGG_M_125",
            "VBFHToGG_M_125",
            "VHtoGG_M_125",
            "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00",
            "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00",
    ]


    # Split up eras to merge for memory
    dict_run_eras = {}
    dict_run_eras["2016"] = {"mc" : ["2016preVFP", "2016postVFP"], "data": ["2016preVFP", "2016postVFP"]}
    dict_run_eras["2017"] = {"mc" : ["2017"], "data": ["2017"]}
    dict_run_eras["2018"] = {"mc" : ["2018"], "data": ["2018"]}
    dict_run_eras["2022"] = {"mc" : ["preEE", "postEE"], "data": ["2022_EraC","2022_EraD","2022_EraE","2022_EraF","2022_EraG"]}
    dict_run_eras["2023"] = {"mc" : ["preBPix", "postBPix"], "data": ["2023_EraC","2023_EraD"]}
    dict_run_eras["2024"] = {"mc" : ["2024"], "data": ["2024_EraC_EG0", "2024_EraC_EG1", "2024_EraD_EG0", "2024_EraD_EG1", "2024_EraE_EG0", "2024_EraE_EG1", "2024_EraF_EG0", "2024_EraF_EG1", "2024_EraG_EG0", "2024_EraG_EG1", "2024_EraH_EG0", "2024_EraH_EG1", "2024_EraIv1_EG0", "2024_EraIv1_EG1", "2024_EraIv2_EG0", "2024_EraIv2_EG1"]}

    systs = []
    # systs = [
    #         "ScaleEB2G_IJazZ_down",
    #         "ScaleEB2G_IJazZ_up",
    #         "ScaleEE2G_IJazZ_down",
    #         "ScaleEE2G_IJazZ_up",
    #         "Smearing2G_IJazZ_down",
    #         "Smearing2G_IJazZ_up",
    #         "jec_syst_Total_down",
    #         "jec_syst_Total_up",
    #         "jer_syst_down",
    #         "jer_syst_up",
    #         "FNUF_down",
    #         "FNUF_up",
    #         "Material_down",
    #         "Material_up",
    # ]
    
    for era in dict_run_eras.keys():
        print(f"Loading samples for {era}.")
        dict_era = dict_run_eras[era]

        merged_samples_MC = load_samples(base_path, dict_era["mc"], samples)
        merged_samples_data = load_samples(base_path, dict_era["data"], [""] ,data=True)

        merged_samples = pd.concat([merged_samples_MC, merged_samples_data], ignore_index=True)
        merged_samples.to_parquet(f"{base_path}/merged_samples_{era}.parquet", engine='pyarrow')

        for syst in systs:
            print(syst)
            merged_samples_MC = load_samples(base_path, dict_era["mc"], samples, syst=syst)
            merged_samples_MC.to_parquet(f"merged_samples_{syst}_{era}.parquet", engine='pyarrow')
            print()
