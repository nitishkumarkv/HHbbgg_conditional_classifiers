#!/usr/bin/env python3
"""
Categorisation optimisation with gato-hep on a 3D discriminant.

This runner uses the gato-hep package to optimise differentiable categories via
an n-component 3D Gaussian Mixture Model (GMM). The 3D input is built from:
  - ttH_vs_tH NN output,
  - sig_vs_bkg NN for ttH,
  - sig_vs_bkg NN for tH.

Signal/background data are read and reweighted with the existing ML-prep
utilities in this repo; soft assignments, temperature annealing and model I/O
are provided by gato-hep. The script reports per-epoch significances, tracks
bias, saves diagnostics, and writes checkpoints (last and best epoch).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import awkward as ak
from collections import defaultdict
import pandas as pd
# We reuse your logic from the ML prep script:
from joblib import Memory
from gato_utils import (
    load_dataframes_for_diff_cats, plot_mass_spectrum_by_subcat, save_metrics_to_txt,
    compute_metrics_for_region, compute_combined_totals, create_mass_hist,
    compute_nonres_reweight_factors_3d, group_histograms,
    plot_significance_and_loss_histories, plot_bias_history_with_temp,
    plot_significance_and_loss_histories_,
    plot_yield_histories, make_schedule, plot_yield_histories_
)
memory = Memory(location="./cache_dir2", verbose=0)

from gatohep.models import gato_gmm_model
from gatohep.utils import asymptotic_significance, TemperatureScheduler
from gatohep.losses import low_bkg_penalty

from optimizer_helpers import (
    assign_trainable_variables,
    lbfgs_with_restarts,
    pack_trainable_variables,
)

import tensorflow_probability as tfp
tfd = tfp.distributions

groupings = {
        "VBFHToGG_M_125": "VBFHToGG_M_125",
        "VHtoGG_M_125": "VHtoGG_M_125",
        "ttHtoGG_M_125": "ttHtoGG_M_125",
        #"BBHto2G_M_125": "BBHto2G_M_125",
        "GluGluHToGG_M_125": "GluGluHToGG_M_125",
        "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
        "VBFHH_CV_1p000_C2V_1p000_C3_1p000": "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
        "TTGG": "TTGG",
        "GGJets": "GGJets",
        "DDQCDGJET": "DDQCDGJET",
        "TTG_100_200": "TTG_100_200",
        "TTG_200": "TTG_200",
}
      

resonant_processes = [
    "VBFHToGG_M_125", 
    "VHtoGG_M_125", 
    "ttHtoGG_M_125", 
    #"BBHto2G_M_125", 
    "GluGluHToGG_M_125",
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000"
]
samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_5p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_0p00_kt_1p00_c2_0p00", "GluGlutoHHto2B2G_kl_2p45_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "VBFHH_CV_1p000_C2V_1p000_C3_1p000"]
samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "VBFHH_CV_1p000_C2V_1p000_C3_1p000"]
samples_list = ["VBFHToGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "GluGluHToGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00","TTGG", "GGJets", "DDQCDGJET", "VBFHH_CV_1p000_C2V_1p000_C3_1p000"]
signal_samples = ["GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"]

def preselection(events, scores):
        
        mass_bool = ((events.mass > 100) & (events.mass < 180))
        dijet_mass_bool = ((events.nonResReg_vbfpair_dijet_mass_DNNreg > 70) & (events.nonResReg_vbfpair_dijet_mass_DNNreg < 190))

        lead_mvaID_bool = (events.lead_mvaID > -0.7)
        sublead_mvaID_bool = (events.sublead_mvaID > -0.7)

        boosted_mask = (events.is_boosted == False)

        #events = events[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool)]
        #scores = scores[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool)]

        score_bool = (scores[:, 3]<0.774)

        events = events[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool & boosted_mask & score_bool)]
        scores = scores[(mass_bool & dijet_mass_bool & lead_mvaID_bool & sublead_mvaID_bool & boosted_mask & score_bool)]

        return events, scores

import multiprocessing
from functools import partial

def _load_one_sample_era(base_path, sample, era, dijet_mass_key, dim=3):
    samp_dir = os.path.join(base_path, "individual_samples", era, sample)
    y_file = os.path.join(samp_dir, "y.npy")
    evt_file = os.path.join(samp_dir, "events_boostedCat.parquet")

    if not (os.path.exists(y_file) and os.path.exists(evt_file)):
        print(f"[load_samples] WARNING: missing files for {samp_dir}, skipping.")
        return None

    try:
        y = np.load(y_file, mmap_mode="r")
        
        
        events = ak.from_parquet(
            evt_file,
            columns=[
                "mass", dijet_mass_key,
                "lead_genPartFlav", "sublead_genPartFlav",
                "weight_tot",
                "lead_mvaID", "sublead_mvaID",
                "is_boosted"
            ],
        )

        events, y = preselection(events, y)
        print("shape of y before removing last column:", y.shape)
        # remove the last column
        if (y.ndim > 1 and y.shape[1] > 1) and dim == 3:
            y = y[:, :-1]
        print("shape of y after removing last column:", y.shape)

        if sample.startswith("TTG_") or sample in {"TT", "TTGG"}:
            sel = (
                (events["lead_genPartFlav"] == 1)
                & (events["sublead_genPartFlav"] == 1)
            )
            events = events[sel]
            y = y[sel]

        if len(y) == 0:
            return None

        return {
            "NN_output": np.asarray(y),
            "diphoton_mass": np.asarray(events["mass"]),
            "massH": np.asarray(events["mass"]),
            "dijet_mass": np.asarray(events[dijet_mass_key]),
            "weights": np.asarray(events["weight_tot"]),
            "labels": np.full(
                len(y),
                1 if sample in signal_samples else 0,
                dtype=int,
            ),
            "sample": np.repeat(sample, len(y)),
        }

    except Exception as exc:
        print(f"[load_samples] ERROR while reading {samp_dir}: {exc}")
        return None


def load_samples(base_path, sample_list, n_workers=1, dim=4):

        eras = ("2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025")
        dijet_mass_key = "nonResReg_vbfpair_dijet_mass_DNNreg"

        #data_dict_MC = {}
        data_dict_MC = defaultdict(pd.DataFrame)

        for sample in sample_list:

            data = {k: [] for k in (
            "NN_output", "diphoton_mass", "dijet_mass", "massH",
            "weights", "labels", "sample"
            )}
            print(sample)
            
            tasks = [
                (base_path, sample, era, dijet_mass_key, dim)
                for era in eras
            ]

            if n_workers > 1:
                with multiprocessing.Pool(processes=n_workers) as pool:
                    results = pool.starmap(_load_one_sample_era, tasks)
            else:
                results = [_load_one_sample_era(*t) for t in tasks]

            for result in results:
                if result is None:
                    continue
                
                for key in data:
                    data[key].append(result[key])

            # Concatenate each list into one array
            for key in data:
                data[key] = np.concatenate(data[key], axis=0)

            scores = data.pop("NN_output")          # shape (N, n_scores)
            argmax = np.argmax(scores, axis=1)  # safest when scores is 2-D

            # Assemble DataFrame
            df = pd.DataFrame(data)
            df["NN_output"] = list(scores)          # store per-event score vectors
            df["arg_max_score"] = argmax

            data_dict_MC[sample] = df

        # Quick bookkeeping
        #in_peak = (df["diphoton_mass"] > 120) & (df["diphoton_mass"] < 130)
        #print(f"Background weight: {df.loc[df['labels'] == 0, 'weights'].sum():.3g}")
        #print(f"Signal weight:     {df.loc[df['labels'] == 1, 'weights'].sum():.3g}")
        #print(f"Bkg weight 120-130 GeV: {df.loc[(df['labels'] == 0) & in_peak, 'weights'].sum():.3g}")
        #print(f"Sig weight 120-130 GeV: {df.loc[(df['labels'] == 1) & in_peak, 'weights'].sum():.3g}")

        return data_dict_MC

def _load_one_data_sample(base_path, data, dijet_mass_key, dim=3):
    data_dir = os.path.join(base_path, "individual_samples_data", data)
    evt_file = os.path.join(data_dir, "events_boostedCat.parquet")
    y_file = os.path.join(data_dir, "y.npy")

    try:
        df = pd.read_parquet(
            evt_file,
            columns=[
                "mass", dijet_mass_key,
                "lead_mvaID", "sublead_mvaID", "is_boosted"
            ],
            engine="pyarrow",
        )

        y = np.load(y_file, mmap_mode="r")
        

        df, y = preselection(df, y)
        # remove the last column
        if (y.ndim > 1 and y.shape[1] > 1) and dim == 3:
            y = y[:, :-1]
        print("shape of y:", y.shape)
        df["NN_output"] = list(y)
        df["massH"] = df["mass"]  # for consistency with MC data_dict
        df["weights"] = np.ones(len(df))  # dummy weights for data

        return df

    except FileNotFoundError:
        print(f"[load_data_samples] WARNING: missing files for {data_dir}, skipping.")
        return None

    except Exception as exc:
        print(f"[load_data_samples] ERROR while reading {data_dir}: {exc}")
        return None

def load_data_samples(base_path, data_list, n_workers=1, dim=4):

    data_frames = []
    dijet_mass_key = "nonResReg_vbfpair_dijet_mass_DNNreg"

    tasks = [
        (base_path, data, dijet_mass_key, dim)
        for data in data_list
    ]

    if n_workers > 1:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.starmap(_load_one_data_sample, tasks)
    else:
        results = [_load_one_data_sample(*t) for t in tasks]

    for df in results:
        if df is not None and not df.empty:
            data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df

# ------------------------------------------------------------------
# Single 3-D GMM model for CMS analysis
# ------------------------------------------------------------------
class GMMMultiDim(gato_gmm_model):
    """
    Generic N-dimensional GMM wrapper for gato-hep used for categorisation.

    * n_cats : number of GMM components (categories)
    * dim    : dimensionality of the input discriminant vector (e.g. 4)
    * signal_pattern : substring to identify the signal process in data_dict keys

    call() returns: (loss, B, nonres, S_signal, B_w2, Z_signal)
    where loss = -Z_signal (negative so optimizer maximises significance)
    """
    def __init__(self, n_cats, dim=4, temperature=0.5, mean_norm="softmax", mean_range=None, name="GMMMultiDim", mass_low=120.0, mass_high=130.0):
        if mean_range is None:
            mean_range = [(0.0, 1.0)] * dim
        super().__init__(
            n_cats=n_cats,
            dim=dim,
            temperature=temperature,
            mean_norm=mean_norm,
            mean_range=mean_range,
            name=name,
        )
        self.mass_low = tf.constant(float(mass_low), dtype=tf.float32)
        self.mass_high = tf.constant(float(mass_high), dtype=tf.float32)

    @tf.function
    def call(self, data_dict, rewt=None):

        # per-category accumulators
        S_ggHH = tf.zeros(self.n_cats, dtype=tf.float32)
        B     = tf.zeros(self.n_cats, dtype=tf.float32)
        B_w2  = tf.zeros(self.n_cats, dtype=tf.float32)
        nonres = [tf.constant(0.0) for _ in range(self.n_cats)]

        for proc, t in data_dict.items():
            #if "cpodd" in proc.lower() or "data" in proc.lower():
            #    continue
            X = t["NN_output"]               # (N,3)
            w = t["weights"]                 # (N,)
            # mass‐window for resonant processes: only keep 123–127 GeV
            if proc in resonant_processes:
                m = t["massH"]
                mask = tf.logical_and(m > self.mass_low, m < self.mass_high)
                X = tf.boolean_mask(X, mask)
                w = tf.boolean_mask(w, mask)
            w2 = w**2

            # Use gatohep model's differentiable assignments
            memberships = self.get_probs(X)            # (N, n_cats)

            yields = tf.reduce_sum(memberships * w[:,None],  axis=0)
            sumw2  = tf.reduce_sum(memberships * w2[:,None], axis=0)

            if rewt is None:
                fac = 1.0
            elif proc in resonant_processes:
                fac = tf.ones_like(rewt)
            else:
                fac = rewt

            if proc.startswith("GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"):
                S_ggHH += yields
            else:
                B     += yields * fac
                B_w2  += sumw2 * fac
                if proc not in resonant_processes:
                    nonres += yields

        # Significances (Asimov); treat other signal as bkg when computing each sig.
        Z_ggHH = tf.sqrt(tf.reduce_sum(asymptotic_significance(S_ggHH, B)**2))

        # geometric mean loss (negative, to maximise)
        loss = -Z_ggHH

        return loss, B, nonres, S_ggHH, B_w2, Z_ggHH
    
    # (get_soft_membership removed; rely on model.get_probs from gatohep)

@tf.function
def train_step(model, tensor_data, opt, rewt, lam=0.0, penalty_threshold=10.0):
    with tf.GradientTape() as tape:
        significance_loss, B, yield_nonres, S_ggHH, B_w2, Z_ggHH = model.call(tensor_data, rewt)
        penalty = low_bkg_penalty(yield_nonres, threshold=penalty_threshold)
        total_loss = significance_loss + lam * penalty
    grads = tape.gradient(total_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, significance_loss, penalty, Z_ggHH, B, yield_nonres, S_ggHH,
###############################################################################
# 3) Main optimize + diagnostics
###############################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True)
    parser.add_argument("--stop", type=int, default=None, help="entry_stop in reading parquet files")
    parser.add_argument("--n_cats",  type=int, default=7, help="(2,3,...) => # categories in 3D discriminant space")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--penalty-threshold", type=float, default=10.0, help="Threshold applied in low background penalty term")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--optimizer", choices=["adam", "rmsprop", "lbfgs"], default="rmsprop")
    # temperature annealing controls (via gatohep TemperatureScheduler)
    parser.add_argument("--temp_init", type=float, default=0.3, help="Initial GMM temperature")
    parser.add_argument("--temp_final", type=float, default=0.05, help="Final GMM temperature")
    parser.add_argument("--vetoBoosted", action="store_true", help="Veto boosted signal regions")
    parser.add_argument("--temp_mode", choices=["exponential", "cosine"], default="cosine", help="Annealing schedule")
    parser.add_argument("--out_folder", type=str, default="opt", help="Base output folder for results")
    parser.add_argument("--input_dim", type=int, default=4, help="Dimensionality of input discriminant vector (default: 4)")

    args = parser.parse_args()



    lam = args.lam # for penalty term
    penalty_threshold = args.penalty_threshold
    lam_str = str(lam).replace("0.", "0p")
    penalty_str = str(penalty_threshold).replace("0.", "0p")
    path_output_plots = (
        f"optimisationOutput/"
        f"{args.out_folder}_{args.n_cats}_cats_lam_{lam_str}_pen_{penalty_str}_opt_{args.optimizer}_epochs_{args.epochs}_dim_{args.input_dim}/"
    )
    os.makedirs(path_output_plots, exist_ok=True)

    # 1) load data with your approach
    #data_dict_MC = load_samples(base_path=args.channel, sample_list=samples_list)
    data_dict_MC = load_samples(base_path=args.channel, sample_list=samples_list, n_workers=4, dim=args.input_dim)

    data_list = [
        "2016postVFP",
        "2016preVFP",
        "2017",
        "2018",
        "2022_EraC",
        "2022_EraD",
        "2022_EraE",
        "2022_EraF",
        "2022_EraG",
        "2023_EraC",
        "2023_EraD",
        "2024",
        "2025"
    ]
    data_list = ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]
    df_data_obs = load_data_samples(base_path=args.channel, data_list=data_list, n_workers=4, dim=args.input_dim)

    for proc in [*data_dict_MC]:
        print(proc, "Sum of weights, max weight and N(events):", np.sum(data_dict_MC[proc].weights), np.max(data_dict_MC[proc].weights), len(data_dict_MC[proc]))
    #print(data_dict_MC)

    #if input_dim is None:
    input_dim = args.input_dim
    #input_dim = 3

    tensor_data = {}
    for proc, df in data_dict_MC.items():
        #X = np.vstack(df["NN_output"].values).astype(np.float32)
        X = np.stack(df["NN_output"].values).astype(np.float32)  # shape (N,3)
        tensor_data[proc] = {
            "NN_output": tf.constant(X, dtype=tf.float32),
            "weights":  tf.constant(np.asarray(df["weights"].values), dtype=tf.float32),
            "massH":    tf.constant(np.asarray(df["massH"].values), dtype=tf.float32),
        }

    # Now downstream, each df has a column "NN_output" of shape (N,) where each entry is length-3.
    n_cats = args.n_cats
    # create model with inferred input dimension and user-specified signal pattern
    model = GMMMultiDim(
        n_cats=n_cats,
        dim=input_dim,
        temperature=float(args.temp_init),
        mean_norm="softmax",
        mean_range=None,
    )

    optimizer_name = args.optimizer

    # store training history
    z_ggHH_history = []
    loss_history = []
    regularisation_history = []
    S_ggHH_history = []
    B_history = []
    B_nonres_history = []
    bias_history = []  # track soft-hard assignment bias per category
    bias_epochs = []   # epochs at which bias was evaluated
    temp_points = []   # temperature values at those epochs

    history_filename = os.path.join(path_output_plots, "history.csv")
    if not os.path.exists(history_filename):
        with open(history_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "z_ggHH", "loss"])

    best_sig_loss = None
    best_epoch = None
    best_weights = None

    def update_best(sig_loss_val, epoch_idx):
        nonlocal best_sig_loss, best_epoch, best_weights
        if ((best_sig_loss is None) or (sig_loss_val < best_sig_loss)) and (epoch_idx >= args.epochs * 0.8):
            best_sig_loss = sig_loss_val
            best_epoch = epoch_idx
            best_weights = [v.numpy() for v in model.trainable_variables]

    def append_history(epoch_idx, significance_loss, reg, z_ggHH, S_ggHH, B, yield_nonres, *, force_bias=False):
        z_ggHH_val = z_ggHH.numpy()
        loss_val = significance_loss.numpy()
        z_ggHH_history.append(z_ggHH_val)
        loss_history.append(loss_val)
        regularisation_history.append(reg.numpy())
        S_ggHH_history.append(tf.stack(S_ggHH).numpy())
        B_history.append(tf.stack(B).numpy())
        B_nonres_history.append(yield_nonres.numpy())

        if epoch_idx % 10 == 0:
            print(
                f"Epoch {epoch_idx}: z_ggHH={z_ggHH_val:.3f}, "

                f"S_ggHH={S_ggHH_history[-1]}, B={B_history[-1]}"
            )

        if (epoch_idx % 5) == 0 or force_bias:
            bias_input = {
                p: {"NN_output": tensor_data[p]["NN_output"], "weight": tensor_data[p]["weights"]}
                for p in tensor_data
            }
            bias_vec = model.get_bias(bias_input)
            bias_history.append(bias_vec)
            bias_epochs.append(epoch_idx)
            try:
                cur_temp = float(model.temperature.numpy())
            except Exception:
                cur_temp = float(model.temperature)
            temp_points.append(cur_temp)

        with open(history_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch_idx, z_ggHH_val, loss_val])

    def build_optimizer(name: str, lr: float):
        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=make_schedule(lr, steps=args.epochs // 7))
        if name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=make_schedule(0.25*lr, steps=args.epochs // 7))
        raise ValueError(f"Unsupported optimizer {name}")

    warmup_epochs = min(50, args.epochs)

    if optimizer_name in ("adam", "rmsprop"):
        opt = build_optimizer(optimizer_name, 0.5)
        temp_sched = TemperatureScheduler(
            model,
            t_initial=float(args.temp_init),
            t_final=float(args.temp_final),
            total_epochs=int(args.epochs),
            mode=str(args.temp_mode),
        )
        factors_tf = tf.constant(np.full(n_cats, 0.062, dtype=np.float32))
        for epoch in range(args.epochs):
            if epoch % 50 == 0:
                factors = compute_nonres_reweight_factors_3d(model, data_dict_MC, channel=args.channel, mass_sig_low=120, mass_sig_high=130)
                factors_tf = tf.constant(factors, dtype=tf.float32)

            _, significance_loss, reg, z_ggHH, B, yield_nonres, S_ggHH = train_step(
                model,
                tensor_data,
                opt,
                rewt=factors_tf,
                lam=args.lam,
                penalty_threshold=penalty_threshold,
            )
            print(f"Loss in epoch {epoch + 1}: {float(significance_loss.numpy()):.6f}")
            update_best(float(significance_loss.numpy()), epoch)
            temp_sched.update(epoch)
            append_history(epoch, significance_loss, reg, z_ggHH, S_ggHH, B, yield_nonres, force_bias=(epoch == args.epochs - 1))

        warm_factors = factors_tf.numpy()
        current_step = args.epochs
    else:
        # Warm-up phase with Adam before LBFGS
        opt_warmup = tf.keras.optimizers.Adam(learning_rate=make_schedule(0.02, steps=max(1, warmup_epochs // 2)))
        temp_sched = TemperatureScheduler(
            model,
            t_initial=float(args.temp_init),
            t_final=float(args.temp_final),
            total_epochs=max(1, warmup_epochs),
            mode=str(args.temp_mode),
        )
        factors_tf = tf.constant(np.full(n_cats, 0.062, dtype=np.float32))
        for epoch in range(warmup_epochs):
            if epoch % 50 == 0:
                factors = compute_nonres_reweight_factors_3d(model, data_dict_MC, channel=args.channel, mass_sig_low=120, mass_sig_high=130)
                factors_tf = tf.constant(factors, dtype=tf.float32)
            _, significance_loss, reg, z_ggHH, B, yield_nonres, S_ggHH = train_step(
                model,
                tensor_data,
                opt_warmup,
                rewt=factors_tf,
                lam=args.lam,
                penalty_threshold=penalty_threshold,
            )
            print(f"Loss in epoch {epoch + 1}: {float(significance_loss.numpy()):.6f}")
            update_best(float(significance_loss.numpy()), epoch)
            temp_sched.update(epoch)
            append_history(epoch, significance_loss, reg, z_ggHH, S_ggHH, B, yield_nonres, force_bias=(epoch == warmup_epochs - 1))

        current_step = warmup_epochs
        warm_factors = factors_tf.numpy()
        # Freeze temperature at final value for LBFGS
        final_temp = tf.constant(float(args.temp_final), dtype=tf.float32)
        try:
            model.temperature.assign(final_temp)
        except AttributeError:
            model.temperature = final_temp

        factors_var = tf.Variable(warm_factors, dtype=tf.float32, trainable=False)
        trainable_vars = model.trainable_variables
        initial_flat, var_shapes, var_sizes = pack_trainable_variables(trainable_vars)
        step_cap = 0.25

        eval_count = current_step
        def assign_with_cap(vector, cap):
            assign_trainable_variables(trainable_vars, vector, var_shapes, var_sizes, step_cap=cap)

        def loss_and_grad(weights):
            nonlocal eval_count
            weights = np.asarray(weights, dtype=np.float64)
            assign_with_cap(weights, step_cap)

            if eval_count % 50 == 0:
                factors = compute_nonres_reweight_factors_3d(model, data_dict_MC, channel=args.channel, mass_sig_low=120, mass_sig_high=130)
                factors_var.assign(tf.convert_to_tensor(factors, dtype=tf.float32))

            with tf.GradientTape() as tape:
                loss_vals = model.call(tensor_data, rewt=factors_var)
                sig_loss, B, yield_nonres, S_ggHH, B2, Z_ggHH = loss_vals
                penalty = low_bkg_penalty(yield_nonres, threshold=penalty_threshold)
                total = sig_loss + args.lam * penalty

            grads = tape.gradient(total, trainable_vars)
            grad_flat = np.concatenate([g.numpy().reshape(-1) for g in grads]).astype(np.float64)

            sig_val = float(sig_loss.numpy())
            update_best(sig_val, eval_count)

            if (eval_count % 5) == 0:
                print(
                    f"[Eval {eval_count}] loss={sig_val:.6f} total={float(total.numpy()):.6f} "
                    f"Z_ggHH={float(Z_ggHH.numpy()):.3f}"
                )

            append_history(eval_count, sig_loss, penalty, Z_ggHH, S_ggHH, B, yield_nonres)
            eval_count += 1
            return float(total.numpy()), grad_flat

        def assign_exact(vector):
            assign_with_cap(vector, None)

        def log_status(attempt, result):
            message = result.message.decode() if isinstance(result.message, bytes) else result.message
            print(f"[LBFGS] attempt={attempt} status={result.status} message='{message}'")

        lbfgs_options = {"maxiter": args.epochs, "maxcor": 7, "ftol": 1e-9, "gtol": 5e-6, "maxls": 12}
        lbfgs_with_restarts(
            loss_and_grad,
            initial_flat,
            options=lbfgs_options,
            restarts=3,
            assign_initial=assign_exact,
            status_callback=log_status,
            jitter_sigma=0.05,
        )

    
    # save last-epoch model and write a small record with best-epoch info
    model.save(path_output_plots)
    if best_epoch is not None:
        with open(os.path.join(path_output_plots, "best_epoch.txt"), "w") as f:
            f.write(f"best_epoch={best_epoch}\n")
            f.write(f"best_significance_loss={best_sig_loss}\n")
        final_epoch = args.epochs - 1
        if (best_weights is not None) and (best_epoch != final_epoch):
            for var, val in zip(model.trainable_variables, best_weights):
                var.assign(val)
            best_dir = os.path.join(path_output_plots, f"best_epoch_{best_epoch:03d}")
            os.makedirs(best_dir, exist_ok=True)
            model.save(best_dir)

    # Plot histories via utils
    plot_significance_and_loss_histories_(
        path_output_plots, z_ggHH_history, loss_history, regularisation_history
    )

    # Bias history (sparse)
    plot_bias_history_with_temp(
        path_output_plots, bias_epochs, bias_history, temp_points
    )

    # Yield histories
    epochs_arr = np.arange(len(z_ggHH_history))
    plot_yield_histories_(
        path_output_plots, epochs_arr,
        S_ggHH_history, B_nonres_history,
        args.channel, args.n_cats
    )


    # ------------------------------------------------------------------
    # 1) Prepare per–process, per–category mass histograms
    # ------------------------------------------------------------------
    ncat = model.n_cats
    hists_per_proc = {}        # {proc: [hist_k]}

    for proc, df in list(data_dict_MC.items()) + [("Data", df_data_obs)]:
        if df.empty:
            continue

        # get bins
        cat_index = model.get_bin_indices(np.vstack(df["NN_output"].values))  # (N, n_cats) tf.Tensor
        #print(cat_index)
        df["cat_index"] = cat_index

        hists_per_proc[proc] = [create_mass_hist() for _ in range(ncat)]

        m = df.massH.values
        w = df.weights.values

        for k in range(ncat):
            mask = (cat_index == k)
            if np.any(mask):
                hists_per_proc[proc][k].fill(m[mask], weight=w[mask])

    # save the updated dictonery of df with updated cat_index for later use in plotting in parquet files
    for proc, df in data_dict_MC.items():
        out_dir = os.path.join(path_output_plots, "per_process_dfs")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{proc}_with_cat_index.parquet")
        df.to_parquet(out_file)
        print(f"[INFO] Saved DataFrame with cat_index for {proc} to {out_file}")

    # also save the data DataFrame with cat_index
    out_dir = os.path.join(path_output_plots, "per_process_dfs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Data_with_cat_index.parquet")
    df_data_obs.to_parquet(out_file)
    print(f"[INFO] Saved DataFrame with cat_index for Data to {out_file}")


    # ------------------------------------------------------------------
    # 2) Group backgrounds / signals exactly like before
    # ------------------------------------------------------------------
    combined_totals = compute_combined_totals(hists_per_proc, {})
    metrics = compute_metrics_for_region(hists_per_proc, combined_totals)

    out_txt = os.path.join(path_output_plots, "categoryMetrics.txt")
    save_metrics_to_txt(metrics, metrics, out_txt)
    print(f"[INFO] Category metrics saved to {out_txt}")

    #groupings = physicsHelper.get_process_grouping()
    grouped_hists = group_histograms(hists_per_proc, groupings)

    # ------------------------------------------------------------------
    # 4) Plot per-category mass spectra
    # ------------------------------------------------------------------
    for k in range(ncat):
        out_pdf = os.path.join(path_output_plots, f"mass_spectrum_cat{k}.pdf")
        plot_mass_spectrum_by_subcat("Inclusive", args.channel, grouped_hists, k, out_pdf)


    # plot mass sculpting plots
    import mplhep as hep
    def test_mass_sculpting(data_dict_MC_, ncat):

        path_for_plots = f"{path_output_plots}/mass_sculpting_plots/"
        os.makedirs(path_for_plots, exist_ok=True)

        columns_to_load = ["massH", "weights"]

        df_GGjets = data_dict_MC_["GGJets"]
        df_TTGG = data_dict_MC_["TTGG"]

        df = pd.concat([df_GGjets, df_TTGG], ignore_index=True)

        cat_to_label = {}
        for cat in range(ncat):
            cat_to_label[f"cat_{cat}"] = f"cat_{cat}"
        
        cat_list = list(cat_to_label.keys())

        cat_events = {}
        cat_index = df["cat_index"]
        for cat in range(ncat):
            cat_events[f"cat_{cat}"] = df[df["cat_index"] == cat]

        def plot_with_errorbars(sample, var, range_, label, ax):
            hist, bin_edges = np.histogram(np.array(sample[var]), bins=20, range=range_, weights=np.array(sample["weights"]))
            sumw2, _ = np.histogram(np.array(sample[var]), bins=20, range=range_, weights=np.array(sample["weights"])**2)

            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_widths = np.diff(bin_edges)

            # Normalize to density if requested
            norm_factor = np.sum(hist * bin_widths)
            if norm_factor > 0:
                hist /= norm_factor
                sumw2 /= norm_factor**2

            errors = np.sqrt(sumw2)
            hep.histplot(
                hist,
                bin_edges,
                yerr=errors,
                label=label,
                histtype='step',
                ax=ax,
                linewidth=2
            )

        # plot the mass sculpting for each category
        fig, ax = plt.subplots()
        plot_with_errorbars(df, "massH", [100, 180], "Pre-selection", ax)
        for cat in cat_list:
            plot_with_errorbars(cat_events[cat], "massH", [100, 180], cat_to_label[cat], ax)
        ax.set_xlabel("Di-photon mass (GeV)")
        ax.set_ylabel("Normalized events")
        ax.legend()
        plt.title("GGJets+TTGG")
        plt.tight_layout()
        fig.savefig(f"{path_for_plots}/mass.png")
        plt.clf()

    print("plotting sculpting plots")
    test_mass_sculpting(data_dict_MC, ncat)

    # get event yields

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet


    GGHH = "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"
    VBFHH = "VBFHH_CV_1p000_C2V_1p000_C3_1p000"


    def asymptotic_significance(s, b):
        if b <= 0 or s <= 0:
            return 0.0
        return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


    def weighted_count_and_unc(df):
        w = df["weights"].to_numpy()
        count = np.sum(w)
        unc = np.sqrt(np.sum(w**2))
        return count, unc


    def data_count_and_unc(df):
        n = len(df)
        return n, np.sqrt(n)
    
    def data_sideband_count(total_data_df, cat, signal_min=120, signal_max=130):
        cat_df = total_data_df[
            (total_data_df["cat_index"] == cat)
            & ~((total_data_df["massH"] >= signal_min) & (total_data_df["massH"] <= signal_max))
        ]
        return data_count_and_unc(cat_df)


    def fmt_count(count, unc):
        return f"{count:.2f} +/- {unc:.2f}"


    def signal_significance(mc_dict, signal_name, cat, mass_min, mass_max):
        sig_df = mc_dict[signal_name]

        sig_sel = sig_df[
            (sig_df["cat_index"] == cat)
            & (sig_df["massH"] >= mass_min)
            & (sig_df["massH"] <= mass_max)
        ]

        s = sig_sel["weights"].sum()

        b = 0.0
        for name, df in mc_dict.items():
            if name == signal_name:
                continue

            bkg_sel = df[
                (df["cat_index"] == cat)
                & (df["massH"] >= mass_min)
                & (df["massH"] <= mass_max)
            ]

            b += bkg_sel["weights"].sum()

        return asymptotic_significance(s, b)

    def total_mc_sideband_count(mc_dict, cat, signal_min=120, signal_max=130):
        total = 0.0
        total_var = 0.0

        for df in mc_dict.values():
            cat_df = df[
                (df["cat_index"] == cat)
                & ~((df["massH"] >= signal_min) & (df["massH"] <= signal_max))
            ]

            w = cat_df["weights"].to_numpy()

            total += np.sum(w)
            total_var += np.sum(w**2)

        return total, np.sqrt(total_var)


    def make_event_count_pdf(
        mc_dict_of_df,
        total_data_df,
        output_pdf="event_counts.pdf",
    ):
        categories = sorted(
            set(total_data_df["cat_index"].dropna().unique())
            | set(
                cat
                for df in mc_dict_of_df.values()
                for cat in df["cat_index"].dropna().unique()
            )
        )

        columns = ["Process"] + [f"cat_{cat}" for cat in categories] + ["Quadrature sum"]
        table_rows = [columns]

        # MC process rows
        for proc, df in mc_dict_of_df.items():
            row = [proc]

            for cat in categories:
                cat_df = df[df["cat_index"] == cat]
                count, unc = weighted_count_and_unc(cat_df)
                row.append(fmt_count(count, unc))

            row.append("")
            table_rows.append(row)

        # Significance rows
        significance_specs = [
            ("ggHH Z [120,130]", GGHH, 120, 130),
            ("ggHH Z [123,127]", GGHH, 123, 127),
            ("VBFHH Z [120,130]", VBFHH, 120, 130),
            ("VBFHH Z [123,127]", VBFHH, 123, 127),
        ]

        for label, signal_name, mass_min, mass_max in significance_specs:
            z_values = [
                signal_significance(
                    mc_dict_of_df,
                    signal_name,
                    cat,
                    mass_min,
                    mass_max,
                )
                for cat in categories
            ]

            row = [label]
            row += [f"{z:.3f}" for z in z_values]
            row += [f"{np.sqrt(np.sum(np.array(z_values) ** 2)):.3f}"]

            table_rows.append(row)

        # Total MC row
        total_mc_row = ["Total MC"]

        for cat in categories:
            total = 0.0
            total_var = 0.0

            for df in mc_dict_of_df.values():
                cat_df = df[df["cat_index"] == cat]
                w = cat_df["weights"].to_numpy()

                total += np.sum(w)
                total_var += np.sum(w**2)

            total_mc_row.append(fmt_count(total, np.sqrt(total_var)))

        total_mc_row.append("")
        table_rows.append(total_mc_row)

        # Total MC sideband row
        sideband_row = ["Total MC sideband excl. [120,130]"]

        for cat in categories:
            count, unc = total_mc_sideband_count(
                mc_dict_of_df,
                cat,
                signal_min=120,
                signal_max=130,
            )
            sideband_row.append(fmt_count(count, unc))

        sideband_row.append("")
        table_rows.append(sideband_row)


        # Data sideband row
        data_sideband_row = ["Data sideband excl. [120,130]"]

        for cat in categories:
            count, unc = data_sideband_count(
                total_data_df,
                cat,
                signal_min=120,
                signal_max=130,
            )
            data_sideband_row.append(fmt_count(count, unc))

        data_sideband_row.append("")
        table_rows.append(data_sideband_row)


        # Total MC row
        total_mc_row = ["Total MC"]

        for cat in categories:
            total = 0.0
            total_var = 0.0

            for df in mc_dict_of_df.values():
                cat_df = df[df["cat_index"] == cat]
                w = cat_df["weights"].to_numpy()

                total += np.sum(w)
                total_var += np.sum(w**2)

            total_mc_row.append(fmt_count(total, np.sqrt(total_var)))

        total_mc_row.append("")
        table_rows.append(total_mc_row)


        # Total Data row
        data_row = ["Total Data"]

        for cat in categories:
            cat_df = total_data_df[total_data_df["cat_index"] == cat]
            count, unc = data_count_and_unc(cat_df)
            data_row.append(fmt_count(count, unc))

        data_row.append("")
        table_rows.append(data_row)

        # PDF
        doc = SimpleDocTemplate(
            output_pdf,
            pagesize=landscape(A3),
            rightMargin=10,
            leftMargin=10,
            topMargin=20,
            bottomMargin=20,
        )

        styles = getSampleStyleSheet()

        elements = [
            Paragraph("Event Counts by Category", styles["Title"]),
            Spacer(1, 12),
        ]

        table = Table(table_rows, repeatRows=1)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),

            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, -4), (-1, -1), "Helvetica-Bold"),

            ("BACKGROUND", (0, -4), (-1, -4), colors.whitesmoke),
            ("BACKGROUND", (0, -3), (-1, -3), colors.beige),
            ("BACKGROUND", (0, -2), (-1, -2), colors.whitesmoke),
            ("BACKGROUND", (0, -1), (-1, -1), colors.beige),

            ("FONTSIZE", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        elements.append(table)
        doc.build(elements)

        return pd.DataFrame(table_rows[1:], columns=table_rows[0])

    summary_df = make_event_count_pdf(
    data_dict_MC,
    df_data_obs,
    output_pdf=f"{path_output_plots}/event_counts.pdf",
    )




if __name__=="__main__":
    main()
