import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import mplhep
import awkward as ak

ff_sampledict = {
    "GGJets": "GGJets", 
    "DDQCDGJET": "DDQCDGJET",
    "TTGG": "TTGG",
    "TT": "TT",
    "TTG_10_100": "TTG_10_100",
    "TTG_100_200": "TTG_100_200",
    "TTG_200": "TTG_200",
    "ttHtoGG_M_125": "ttHtoGG",
    "BBHto2G_M_125": "BBHtoGG",
    "GluGluHToGG_M_125": "GluGluHToGG",
    "VBFHToGG_M_125": "VBFHToGG",
    "VHtoGG_M_125": "VHtoGG",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGlutoHH_kl-1p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00": "GluGlutoHH_kl-0p00_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00": "GluGlutoHH_kl-2p45_kt-1p00_c2-0p00",
    "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00": "GluGlutoHH_kl-5p00_kt-1p00_c2-0p00",
}

def load_samples(base_path, samples ,data=False):
    """Load predictions and weights, scaling weights by luminosity."""
    samples_input = {"score": [], "diphoton_mass": [], "dijet_mass": [], "weights": [], 
                     "labels": [], "sample": [], "arg_max_score": [], "nonRes_has_two_btagged_jets": [], "mass": [], 
                     "mHH": [], "mHH_res": [],"weight_tot":[],  "year": [],#"is_boosted": [],, "y_proba":[]
                     "nonRes_score": [], "ttH_score": [], "singleH_score" :[],"ggHH_score":[]}
    eras = ["preEE", "postEE", "preBPix", "postBPix"]
    if data:
        eras = ["2022_EraC","2022_EraD","2022_EraE","2022_EraF","2022_EraG","2023_EraCv1to3","2023_EraCv4","2023_EraD"]
    dijet_mass_key = "Res_mjj_regressed"
    #dijet_mass_key = "dijet_mass"
    # dijet_mass_key = "nonRes_dijet_mass"

    apply_selection = False
    def selection(events, apply_selection):
        # Apply selection criteria here
        # For example, you can filter events based on certain conditions
        # For now, let's just return all events
        if not apply_selection:
            return events
        else:
            print("Applying selection criteria...")
            #nBLoose_cut = (events["nBLoose"] >= 2)
            #events = events[nBLoose_cut]
            nBMedium_cut = (events["nBMedium"] >= 1)
            events = events[nBMedium_cut]

            return events

    for era in eras:
        for sample in samples:
            if era != "postEE":
              if "TTG_" in sample or sample == "TT":
                continue
            if data:
                path = os.path.join(base_path, "individual_samples_data", era, sample)
            else:
                path = os.path.join(base_path, "individual_samples", era, sample)
            y_path = os.path.join(path, 'y.npy')
            print("sample, y_shape[1]", sample, np.load(y_path).shape[1])
            w_path = os.path.join(path, 'rel_w.npy')
            events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=["mass", dijet_mass_key, 
                                                                                    "nonRes_HHbbggCandidate_mass",
                                                                                    "Res_HHbbggCandidate_mass",
                                                                                    "weight_tot",
                                                                                    "nonRes_has_two_btagged_jets", 
                                                                                    # "is_boosted",
                                                                                    # "y_proba"
                                                                                    ])  # Load events
            
            #diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
            #dijet_mass_cut = ((events.nonRes_mjj_regressed > 70) & (events.nonRes_mjj_regressed < 190))
            #nonRes = (events.nonRes_has_two_btagged_jets == True)
            #events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #print((diphoton_mass_cut & dijet_mass_cut & nonRes).shape)
            #print(y_path.shape)

            #w_path = w_path[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #y_path = y_path[diphoton_mass_cut & dijet_mass_cut & nonRes]
            # eras= events['era']
            weight_tot = ak.ones_like(events['mass'])  # Default weight if not provided
            if "weight_tot" in events.fields:
              weight_tot= events['weight_tot']
            mHH = events['nonRes_HHbbggCandidate_mass']
            mHH_res = events['Res_HHbbggCandidate_mass']
            diphoton_mass = events['mass']
            dijet_mass = events[dijet_mass_key]
            nonRes_has_two_btagged_jets = events['nonRes_has_two_btagged_jets']
            # is_boosted = events["is_boosted"]
            # y_proba = events['y_proba'] 
            if "22" in era or "EE" in era:
                year = 2022
            elif "23" in era or "BPix" in era:
                year = 2023
            else:
                raise ValueError(f"Unknown era: {era}")

            # nBLoose = events['nBLoose']
            # nBMedium = events['nBMedium']

            # Check if files exist
            # if not (os.path.exists(y_path) and os.path.exists(w_path) ):
            if not (os.path.exists(y_path)):
                print(f"Missing y for {path}. Skipping.")
                continue

            y = np.load(y_path)
            try:
                weights = np.load(w_path)  # Weights
                
            except :
                print(f"Missing weights for {path}. Using default weights.")
                weights = np.full(y.shape[0],1.0)  # Default weights if not provided
            label = 1 if "GluGlutoHHto2B2G_kl" in sample else 0  # Signal = 1, Background = 0
            samples_input["score"].append(y)
            samples_input["weights"].append(weights)
            samples_input["labels"].append(np.full(y.shape[0], label))
            samples_input["diphoton_mass"].append(np.array(diphoton_mass))
            samples_input["mass"].append(np.array(diphoton_mass))
            samples_input["dijet_mass"].append(np.array(dijet_mass))
            samples_input["mHH"].append(np.array(mHH))
            samples_input["mHH_res"].append(np.array(mHH_res))
            samples_input["weight_tot"].append(np.array(weight_tot))
            # samples_input["is_boosted"].append(np.array(is_boosted))  
            # samples_input["y_proba"].append(np.array(y_proba))  # Assuming y_proba is part of y
            samples_input["year"].append(np.full(y.shape[0], year))
            if sample == "":
                sample = "Data"
            print(sample)
            if sample in ff_sampledict.keys():
                # Use the mapped sample name from ff_sampledict
                  sample = ff_sampledict[sample]
            print(sample)
            samples_input["sample"].append(np.full(y.shape[0], sample))
            samples_input["nonRes_has_two_btagged_jets"].append(np.array(nonRes_has_two_btagged_jets))
            # samples_input["nBLoose"].append(np.array(nBLoose))
            # samples_input["nBMedium"].append(np.array(nBMedium))

    # Concatenate all data
    samples_input["weights"] = np.concatenate(samples_input["weights"], axis=0)
    samples_input["labels"] = np.concatenate(samples_input["labels"], axis=0)
    samples_input["diphoton_mass"] = np.concatenate(samples_input["diphoton_mass"], axis=0)
    samples_input["mass"] = np.concatenate(samples_input["mass"], axis=0)
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["mHH"] = np.concatenate(samples_input["mHH"], axis=0)
    samples_input["mHH_res"] = np.concatenate(samples_input["mHH_res"], axis=0)
    samples_input["weight_tot"] = np.concatenate(samples_input["weight_tot"], axis=0)   
    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["nonRes_has_two_btagged_jets"] = np.concatenate(samples_input["nonRes_has_two_btagged_jets"], axis=0)
    # samples_input["is_boosted"] = np.concatenate(samples_input["is_boosted"], axis=0)
    # samples_input["y_proba"] = np.concatenate(samples_input["y_proba"], axis=0)
    samples_input["year"] = np.concatenate(samples_input["year"], axis=0)
    # samples_input["nBLoose"] = np.concatenate(samples_input["nBLoose"], axis=0)
    # samples_input["nBMedium"] = np.concatenate(samples_input["nBMedium"], axis=0)

    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["arg_max_score"] = np.argmax(scores, axis=1)
    samples_input["score"] = [row for row in scores]
    samples_input["nonRes_score"] = [row[0] for row in scores]
    samples_input["ttH_score"] = [row[1] for row in scores]   
    samples_input["singleH_score"] = [row[2] for row in scores]
    samples_input["ggHH_score"] = [row[3] for row in scores]
    
    # print shape of each array
    for key, value in samples_input.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)}")
        else:
            print(f"{key}: {value.shape}")
    # concovert to pandas dataframe
    samples_input = pd.DataFrame(samples_input)

    # apply selection
    samples_input = selection(samples_input, apply_selection)

    # apply preselection
    #samples_input = samples_input[(samples_input["diphoton_mass"] > 100) & (samples_input["diphoton_mass"] < 180)]
    #samples_input = samples_input[(samples_input["dijet_mass"] > 70) & (samples_input["dijet_mass"] < 190)]
    #samples_input = samples_input[samples_input["nonRes_has_two_btagged_jets"] == True]

    # print sum of total background and signal weights
    print(f"Total background weight: {samples_input['weights'][samples_input['labels'] == 0].sum()}")
    print(f"Total signal weight: {samples_input['weights'][samples_input['labels'] == 1].sum()}")

    # print sum of total background and signal weights under the peak
    print(f"Total background weight under the peak: {samples_input['weights'][(samples_input['labels'] == 0) & (samples_input['diphoton_mass'] > 120) & (samples_input['diphoton_mass'] <130)].sum()}")
    print(f"Total signal weight under the peak: {samples_input['weights'][(samples_input['labels'] == 1) & (samples_input['diphoton_mass'] > 120) & (samples_input['diphoton_mass'] <130)].sum()}")


    return samples_input

def asymptotic_significance(df):
    s = df[df["labels"] == 1].weights.sum()
    b = df[df["labels"] == 0].weights.sum()
    print(f"Signal: {s}, Background: {b}")
    z = np.sqrt(2 * ((s + b + 1e-10) * np.log(1 + (s / (b + 1e-10))) - s))
    #z = np.sqrt(2 * (((s + b + 1e-10) * np.log(1 + s / (b + 1e-10))) - s))
    print(f"Signal: {s}, Background: {b}, Significance: {z}")
    return z

def approx_significance(df):
    s = df[df["labels"] == 1].weights.sum()
    b = df[df["labels"] == 0].weights.sum()
    z = s / np.sqrt(s + b + 1e-10)
    return z

if __name__ == "__main__":
    base_path = "/ceph/cms/store/user/azecchin/ScoredParquets/kl5_training/"
    # base_path = "/ceph/cms/store/user/azecchin/ScoredParquets/optuna_categorization_VBF_version_20250524/"
    # base_path = "/home/users/evourlio/HHbbgg_conditional_classifiers/VBF_version_20250524_MVAIDGr-0p7_mvaIDsAsInput/"
    # base_path = "/home/users/evourlio/HHbbgg_conditional_classifiers/VBF_version_20250524_MVAIDGr-0p7_noMjjCutInParquetsForBoosted/"
    # base_path = "/ceph/cms/store/user/bdanzi/ScoredParquets/VBF_version_20250524_MVAIDGr-0p7_mvaIDsAsInput_noMjjCutInParquetsForBoosted_FIXED/"
    # base_path = "/home/users/evourlio/HHbbgg_conditional_classifiers/VBF_version_20250524_MVAIDGr-0p7_mvaIDsAsInput_noMjjCutInParquetsForBoosted/"
    # base_path = "/ceph/cms/store/user/bdanzi/ScoredParquets/HHbbgg_conditional_classifiersVBF_version_20250524_MVAIDGr-0p7_noMjjCutInParquetsForBoosted_FIXED/"
    samples = ["GGJets",
                   "DDQCDGJET",
                   "TTGG",
                   "TT",
                   "TTG_10_100",
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
    
    merged_samples_MC = load_samples(base_path, samples)
    merged_samples_data = load_samples(base_path, [""] ,data=True)
    print(merged_samples_data)
    print(merged_samples_MC)

    merged_samples = pd.concat([merged_samples_MC, merged_samples_data], ignore_index=True)
    merged_samples.to_parquet("merged_samples.parquet", engine='pyarrow')
