import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import mplhep
import awkward as ak


def load_samples(base_path, samples):
    """Load predictions and weights, scaling weights by luminosity."""
    samples_input = {"score": [], "diphoton_mass": [], "dijet_mass": [], "weights": [], "labels": [], "sample": [], "arg_max_score": [], "nonRes_has_two_btagged_jets": [], "nBLoose": [], "nBMedium": []}
    eras = ["preEE", "postEE", "preBPix", "postBPix"]
    #dijet_mass_key = "nonRes_mjj_regressed"
    #dijet_mass_key = "dijet_mass"
    dijet_mass_key = "nonRes_dijet_mass"

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
            path = os.path.join(base_path, "individual_samples", era, sample)
            y_path = os.path.join(path, 'y.npy')
            print("sample, y_shape[1]", sample, np.load(y_path).shape[1])
            w_path = os.path.join(path, 'rel_w.npy')
            events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=["mass", dijet_mass_key, "nonRes_has_two_btagged_jets", "nBLoose", "nBMedium"])  # Load events
            

            #diphoton_mass_cut = ((events.mass > 100) & (events.mass < 180))
            #dijet_mass_cut = ((events.nonRes_mjj_regressed > 70) & (events.nonRes_mjj_regressed < 190))
            #nonRes = (events.nonRes_has_two_btagged_jets == True)
            #events = events[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #print((diphoton_mass_cut & dijet_mass_cut & nonRes).shape)
            #print(y_path.shape)

            #w_path = w_path[diphoton_mass_cut & dijet_mass_cut & nonRes]
            #y_path = y_path[diphoton_mass_cut & dijet_mass_cut & nonRes]
            
            diphoton_mass = events['mass']
            dijet_mass = events[dijet_mass_key]
            nonRes_has_two_btagged_jets = events['nonRes_has_two_btagged_jets']
            nBLoose = events['nBLoose']
            nBMedium = events['nBMedium']

            # Check if files exist
            if not (os.path.exists(y_path) and os.path.exists(w_path)):
                print(f"Missing y or weights for {path}. Skipping.")
                continue

            y = np.load(y_path)
            weights = np.load(w_path)  # Weights
            label = 1 if "GluGlutoHHto2B2G_kl" in sample else 0  # Signal = 1, Background = 0
            samples_input["score"].append(y)
            samples_input["weights"].append(weights)
            samples_input["labels"].append(np.full(y.shape[0], label))
            samples_input["diphoton_mass"].append(np.array(diphoton_mass))
            samples_input["dijet_mass"].append(np.array(dijet_mass))
            samples_input["sample"].append(np.full(y.shape[0], sample))
            samples_input["nonRes_has_two_btagged_jets"].append(np.array(nonRes_has_two_btagged_jets))
            samples_input["nBLoose"].append(np.array(nBLoose))
            samples_input["nBMedium"].append(np.array(nBMedium))

    # Concatenate all data
    samples_input["weights"] = np.concatenate(samples_input["weights"], axis=0)
    samples_input["labels"] = np.concatenate(samples_input["labels"], axis=0)
    samples_input["diphoton_mass"] = np.concatenate(samples_input["diphoton_mass"], axis=0)
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)
    samples_input["nonRes_has_two_btagged_jets"] = np.concatenate(samples_input["nonRes_has_two_btagged_jets"], axis=0)
    samples_input["nBLoose"] = np.concatenate(samples_input["nBLoose"], axis=0)
    samples_input["nBMedium"] = np.concatenate(samples_input["nBMedium"], axis=0)

    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["arg_max_score"] = np.argmax(scores, axis=1)
    samples_input["score"] = [row for row in scores]
    

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
    base_path = "/net/data_cms3a-1/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/data/inputs_for_MLP_202411226/individual_samples/"
    samples = ["GGJets",
                   "DDQCDGJET",
                   "TTGG",
                   "ttHtoGG_M_125",
                   "BBHto2G_M_125",
                   "GluGluHToGG_M_125",
                   "VBFHToGG_M_125",
                   "VHtoGG_M_125",
                   "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
                   "VBFHHto2B2G_CV_1_C2V_1_C3_1"]
    print(load_samples(base_path, samples))