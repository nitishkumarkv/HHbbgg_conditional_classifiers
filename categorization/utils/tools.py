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
    samples_input = {"score": [], "diphoton_mass": [], "dijet_mass": [], "weights": [], "labels": [], "sample": [], "arg_max_score": []}
    eras = ["preEE", "postEE"]
    for era in eras:
        for sample in samples:
            path = os.path.join(base_path, "individual_samples", era, sample)
            y_path = os.path.join(path, 'y.npy')
            print("sample, y_shape[1]", sample, np.load(y_path).shape[1])
            w_path = os.path.join(path, 'rel_w.npy')
            events = ak.from_parquet(os.path.join(path, 'events.parquet'), columns=["mass", "nonRes_dijet_mass"])  # Load events
            diphoton_mass = events['mass']
            dijet_mass = events['nonRes_dijet_mass']

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

    # Concatenate all data
    samples_input["weights"] = np.concatenate(samples_input["weights"], axis=0)
    samples_input["labels"] = np.concatenate(samples_input["labels"], axis=0)
    samples_input["diphoton_mass"] = np.concatenate(samples_input["diphoton_mass"], axis=0)
    samples_input["dijet_mass"] = np.concatenate(samples_input["dijet_mass"], axis=0)
    samples_input["sample"] = np.concatenate(samples_input["sample"], axis=0)

    scores = np.concatenate(samples_input["score"], axis=0)
    samples_input["arg_max_score"] = np.argmax(scores, axis=1)
    samples_input["score"] = [row for row in scores]
    

    # concovert to pandas dataframe
    samples_input = pd.DataFrame(samples_input)

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