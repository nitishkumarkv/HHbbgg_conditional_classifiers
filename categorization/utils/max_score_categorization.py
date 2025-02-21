import awkward as ak
from tools import load_samples, asymptotic_significance, approx_significance
import numpy as np
import matplotlib.pyplot as plt
import os


def get_best_cut_values_on_max_score_for_n_cuts(n_cuts, samples_input, out_dir):

    # get the ggF category based on the max score
    ggF_cat_based_on_max_score = samples_input[samples_input["arg_max_score"] == 3]
    ggF_cat_based_on_max_score["max_score"] = np.max(np.stack(ggF_cat_based_on_max_score["score"]), axis=1)

    grid_score_cuts = np.linspace(0, 1, 200)

    # create the output directory
    cat_path = f"{out_dir}/max_categorization/"
    os.makedirs(cat_path, exist_ok=True)

    best_cut_values = [1.0]

    best_asy_sig_values = []

    for i in range(n_cuts):

        cut_val_lst = []
        num_bkg_side_bands_lst = []
        num_sig_under_peak_lst = []
        asy_sig_under_peak_lst = []
        appr_sig_under_peak_lst = []

        # get events left after previous cut
        samples_left_after_previous_cut = ggF_cat_based_on_max_score[ggF_cat_based_on_max_score["max_score"] < best_cut_values[i]]

        for cut in grid_score_cuts:
            samples_left_after_cut = samples_left_after_previous_cut[samples_left_after_previous_cut["max_score"] > cut]
            cut_val_lst.append(cut)

            # get the number of background in side bands
            num_bkg_side_bands = samples_left_after_cut["weights"][(samples_left_after_cut["labels"] == 0) & 
                                                                   ((samples_left_after_cut["diphoton_mass"] < 120) | (samples_left_after_cut["diphoton_mass"] > 130))].sum()
            num_bkg_side_bands_lst.append(num_bkg_side_bands)

            # get the number of signal under the peak
            num_sig_under_peak = samples_left_after_cut["weights"][(samples_left_after_cut["labels"] == 1) & (samples_left_after_cut["diphoton_mass"] > 120) & (samples_left_after_cut["diphoton_mass"] < 130)].sum()
            num_sig_under_peak_lst.append(num_sig_under_peak)

            # calculate significance under the peak
            samples_left_after_cut_under_peak = samples_left_after_cut[(samples_left_after_cut["diphoton_mass"] > 120) & (samples_left_after_cut["diphoton_mass"] < 130)]
            asy_sig_under_peak = asymptotic_significance(samples_left_after_cut_under_peak)
            appr_sig_under_peak = approx_significance(samples_left_after_cut_under_peak)
            asy_sig_under_peak_lst.append(asy_sig_under_peak)
            appr_sig_under_peak_lst.append(appr_sig_under_peak)

        # convert into numpy arrays
        cut_val_lst = np.array(cut_val_lst)
        num_bkg_side_bands_lst = np.array(num_bkg_side_bands_lst)
        num_sig_under_peak_lst = np.array(num_sig_under_peak_lst)
        asy_sig_under_peak_lst = np.array(asy_sig_under_peak_lst)
        appr_sig_under_peak_lst = np.array(appr_sig_under_peak_lst)

        # first get all values for which the number of background events in the side bands is greater than 10
        #mask = (num_bkg_side_bands_lst > 10)
        #cut_val_lst = cut_val_lst[mask]
        #num_bkg_side_bands_lst = num_bkg_side_bands_lst[mask]
        #asy_sig_under_peak_lst = asy_sig_under_peak_lst[mask]
        #appr_sig_under_peak_lst = appr_sig_under_peak_lst[mask]
        mask = (num_bkg_side_bands_lst > 10)

        # get the best cut value
        best_val_idx = np.argmax(asy_sig_under_peak_lst[mask])
        best_cut_value = cut_val_lst[best_val_idx]
        best_cut_values.append(best_cut_value)
        print(f"Best cut value for category {i+1}: {best_cut_value}")

        best_asy_sig = asy_sig_under_peak_lst[best_val_idx]
        best_asy_sig_values.append(best_asy_sig)

        best_bkg_side_band = num_bkg_side_bands_lst[best_val_idx]
        best_sig_under_peak = num_sig_under_peak_lst[best_val_idx]

        # plot the optimization values
        plot_name = cat_path + f"cat_{i+1}"
        plot_optimization(cut_val_lst, asy_sig_under_peak_lst, num_sig_under_peak_lst, num_bkg_side_bands_lst, best_cut_value, best_asy_sig, best_sig_under_peak, best_bkg_side_band, plot_name)

        # plot the diphoton and dijet mass for each category
        plot_mass(best_cut_values, ggF_cat_based_on_max_score, cat_path)

        # plot the diphoton and dijet mass for each category for data
        plot_mass_sideband_data(best_cut_values, out_dir, cat_path)

    # plot the sum of Z in quadrature
    plot_Z_sum_quad(best_asy_sig_values, cat_path)




    return best_cut_values, best_asy_sig_values

def plot_optimization(cut_values, asy_sig, sig_under_peak, bkg_side_band, best_cut_value, best_asy_sig, best_sig_under_peak, best_bkg_side_band, plot_name):

    # plot cut value vs asymptotic significance, sig under the peak and number of background events in the side bands along wih the best cut value

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0})
    axs[0].plot(cut_values, asy_sig, color='b')
    axs[0].axvline(x=best_cut_value, color='r', linestyle='--', label="Best cut value")
    axs[0].set_ylabel("Asymptotic Significance", fontsize=12)
    axs[0].annotate(f"{best_asy_sig:.4f}", xy=(best_cut_value, best_asy_sig),
                xytext=(best_cut_value + 0.06, best_asy_sig + (best_asy_sig * 0.0)),
                arrowprops=dict(facecolor='black', arrowstyle="->"))

    axs[1].plot(cut_values, sig_under_peak, color='b')
    axs[1].axvline(x=best_cut_value, color='r', linestyle='--')
    axs[1].set_ylabel("Signal under the peak", fontsize=12)
    axs[1].annotate(f"{best_sig_under_peak:.4f}", xy=(best_cut_value, best_sig_under_peak),
                xytext=(best_cut_value + 0.06, best_sig_under_peak + (best_sig_under_peak * 0.0)),
                arrowprops=dict(facecolor='black', arrowstyle="->"))

    axs[2].plot(cut_values, bkg_side_band, color='b')
    axs[2].axvline(x=best_cut_value, color='r', linestyle='--')
    axs[2].set_ylabel("Background in side bands", fontsize=12)
    axs[2].annotate(f"{best_bkg_side_band:.2f}", xy=(best_cut_value, best_bkg_side_band),
                xytext=(best_cut_value + 0.06, best_bkg_side_band + (best_bkg_side_band * 0.2)),
                arrowprops=dict(facecolor='black', arrowstyle="->"))
    
    for ax in axs:
        ax.axvline(x=best_cut_value, color='r', linestyle='--', label=f"Best cut value: {best_cut_value:.4f}")

    
    #plt.title(f"{plot_name.split('/')[-1]}")
    plt.suptitle(f"{plot_name.split('/')[-1]}", fontsize=12)
    axs[2].set_xlabel("ggFHH max score cut", fontsize=12)
    plt.legend(fontsize=12)

    
    plt.savefig(f"{plot_name}.png")
    plt.clf()

def plot_categories(best_cut_values, samples):
    # plot verticle bar plots for each category showing the fraction of events from each sample

    return None

def plot_mass(best_cut_values, samples_input, plot_dir):
    # plot the diphoton mass and dijet mass for each category

    for i in range(len(best_cut_values)-1):
        samples_for_cut = samples_input[(samples_input["max_score"] < best_cut_values[i]) & (samples_input["max_score"] > best_cut_values[i+1])]
        non_res_bkg_mask = ((samples_for_cut["sample"] == "GGJets") | (samples_for_cut["sample"] == "GJetPt20To40") | (samples_for_cut["sample"] == "GJetPt40") | (samples_for_cut["sample"] == "TTGG"))
        #bkg_diphoton_mass = samples_for_cut["diphoton_mass"][samples_for_cut["labels"] == 0]
        bkg_diphoton_mass = samples_for_cut["diphoton_mass"][non_res_bkg_mask]
        sig_diphoton_mass = samples_for_cut["diphoton_mass"][samples_for_cut["labels"] == 1]
        #bkg_dijet_mass = samples_for_cut["dijet_mass"][samples_for_cut["labels"] == 0]
        bkg_dijet_mass = samples_for_cut["dijet_mass"][non_res_bkg_mask]
        sig_dijet_mass = samples_for_cut["dijet_mass"][samples_for_cut["labels"] == 1]

        # plot the diphoton mass in range 100-180 from the sample weights
        plt.hist(bkg_diphoton_mass, bins=20, range=(100, 180), weights=samples_for_cut["weights"][non_res_bkg_mask], histtype='step', label="Background")
        #plt.hist(sig_diphoton_mass, bins=80, range=(100, 180), weights=samples_for_cut["weights"][samples_for_cut["labels"] == 1]*1000, histtype='step', label="Signal x 1000")
        plt.xlabel("$m_{\gamma\gamma}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/diphoton_mass_cat_{i+1}.png")
        plt.clf()

        # plot the dijet mass in range 100-180 from the sample weights
        plt.hist(bkg_dijet_mass, bins=20, range=(100, 180), weights=samples_for_cut["weights"][non_res_bkg_mask], histtype='step', label="Background")
        #plt.hist(sig_dijet_mass, bins=80, range=(100, 180), weights=samples_for_cut["weights"][samples_for_cut["labels"] == 1]*1000, histtype='step', label="Signal x 1000")
        plt.xlabel("$m_{jj}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/dijet_mass_cat_{i+1}.png")
        plt.clf()

    return

def plot_mass_sideband_data(best_cut_values, base_path, plot_dir):
    # plot the diphoton mass and dijet mass for each category for data

    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    events = []
    scores = []
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        events.append(ak.from_parquet(inputs_path))
        scores.append(np.load(f"{inputs_path}/y.npy"))

    # concatenate the data samples
    events = ak.concatenate(events, axis=0)
    scores = np.concatenate(scores, axis=0)
    arg_max_score = np.argmax(scores, axis=1)
    max_score = np.max(scores, axis=1)
    arg_max_score_mask = (arg_max_score == 3)
    side_band_mask = ((events["mass"] < 120) | (events["mass"] > 130))
    events = events[arg_max_score_mask & side_band_mask]
    scores = scores[arg_max_score_mask & side_band_mask]
    max_score = max_score[arg_max_score_mask & side_band_mask]
    # only consider sidebands data

    for i in range(len(best_cut_values)-1):
        # get diphoton and dijet mass for each category
        data_diphoton_mass = events[(max_score < best_cut_values[i]) & (max_score > best_cut_values[i+1])]["mass"]
        data_dijet_mass = events[(max_score < best_cut_values[i]) & (max_score > best_cut_values[i+1])]["nonRes_dijet_mass"]

        # plot the diphoton mass in range 100-180 from the sample weights
        plt.hist(data_diphoton_mass, bins=20, range=(100, 180), histtype='bar', label="Data 2022")
        plt.xlabel("$m_{\gamma\gamma}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/diphoton_mass_data_cat_{i+1}.png")
        plt.clf()

        # plot the dijet mass in range 100-180 from the sample weights
        plt.hist(data_dijet_mass, bins=20, range=(100, 180), histtype='bar', label="Data 2022")
        plt.xlabel("$m_{jj}$ [GeV]")
        plt.ylabel("Number of events")
        plt.legend()
        plt.savefig(f"{plot_dir}/dijet_mass_data_cat_{i+1}.png")
        plt.clf()
    

def plot_Z_sum_quad(best_asy_sig_values, plot_dir):
    
    Z_sum_quad = []
    for i in range(1, len(best_asy_sig_values)+1):
        Z_sum_quad.append(np.sqrt(np.sum(np.array(best_asy_sig_values[:i])**2)))

    plt.figure(figsize=(10, 6))
    plt.plot([i+1 for i in range(len(best_asy_sig_values))], Z_sum_quad, "b.")

    #also annotote the values
    for i in range(len(Z_sum_quad)):
        plt.annotate(f"{Z_sum_quad[i]:.4f}", xy=(i+1, Z_sum_quad[i]),
                xytext=(i+1 + 0.4, Z_sum_quad[i] + (Z_sum_quad[i] * 0.0)),
                arrowprops=dict(facecolor='black', arrowstyle="->"))

    plt.xlim(0, len(best_asy_sig_values)+1)
    plt.grid(linestyle='--')
    plt.xlabel("Number of categories")
    plt.ylabel("Sum of Z in quadrature")
    plt.savefig(f"{plot_dir}/Z_sum_quad.png")

def store_categorization_events_with_score(base_path, best_cut_values):

    # create the output directory
    out_dir = f"{base_path}/max_categorization/"
    os.makedirs(out_dir, exist_ok=True)

    # category sim
    samples = [
                "GGJets",
                "GJetPt20To40",
                "GJetPt40",
                "TTGG",
                "ttHtoGG_M_125",
                "BBHto2G_M_125",
                "GluGluHToGG_M_125",
                "VBFHToGG_M_125",
                "VHtoGG_M_125",
                "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
                "VBFHHto2B2G_CV_1_C2V_1_C3_1"
                ]
    for era in ["preEE", "postEE"]:
        for sample in samples:
            inputs_path = f"{base_path}/individual_samples/{era}/{sample}"
            # events parquet
            events = ak.from_parquet(inputs_path)
            

            # load score and get max score
            scores = np.load(f"{inputs_path}/y.npy")
            arg_max_score = np.argmax(scores, axis=1)
            max_score = np.max(scores, axis=1)

            # load rel_w
            rel_w = np.load(f"{inputs_path}/rel_w.npy")

            for i in range(len(best_cut_values)-1):

                # get mask for category
                max_score_mask = ((max_score < best_cut_values[i]) & (max_score > best_cut_values[i+1]))
                ggFHH_cat_mask = (arg_max_score == 3)
                mask = max_score_mask & ggFHH_cat_mask

                # store events, y, rel_w
                os.makedirs(f"{out_dir}/cat{i+1}/{era}/{sample}", exist_ok=True)
                ak.to_parquet(events[mask], f"{out_dir}/cat{i+1}/{era}/{sample}/events.parquet")
                np.save(f"{out_dir}/cat{i+1}/{era}/{sample}/y.npy", scores[mask])
                np.save(f"{out_dir}/cat{i+1}/{era}/{sample}/rel_w.npy", rel_w[mask])

    # category data
    data_samples = [
        "Data_EraE",
        "Data_EraF",
        "Data_EraG",
        "DataC_2022",
        "DataD_2022",
    ]
    for data_sample in data_samples:
        inputs_path = f"{base_path}/individual_samples_data/{data_sample}"
        # events parquet
        events = ak.from_parquet(inputs_path)

        # load score and get max score
        scores = np.load(f"{inputs_path}/y.npy")
        arg_max_score = np.argmax(scores, axis=1)
        max_score = np.max(scores, axis=1)

        for i in range(len(best_cut_values)-1):

            # get mask for category
            max_score_mask = ((max_score < best_cut_values[i]) & (max_score > best_cut_values[i+1]))
            ggFHH_cat_mask = (arg_max_score == 3)
            mask = max_score_mask & ggFHH_cat_mask

            # store events, y, rel_w
            os.makedirs(f"{out_dir}/cat{i+1}/{data_sample}", exist_ok=True)
            ak.to_parquet(events[mask], f"{out_dir}/cat{i+1}/{data_sample}/events.parquet")
            np.save(f"{out_dir}/cat{i+1}/{data_sample}/y.npy", scores[mask])

    return 0




    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cuts", type=int, default=6, help="Number of categories")
    parser.add_argument("--base_path", type=str, help="Base path to the input samples")
    args = parser.parse_args()



    # load the samples
    base_path = args.base_path
    samples = ["GGJets",
                   "GJetPt20To40",
                   "GJetPt40",
                   "TTGG",
                   "ttHtoGG_M_125",
                   "BBHto2G_M_125",
                   "GluGluHToGG_M_125",
                   "VBFHToGG_M_125",
                   "VHtoGG_M_125",
                   "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
                   "VBFHHto2B2G_CV_1_C2V_1_C3_1"]
    
    samples_input = load_samples(base_path, samples)
    # get the best cut values
    best_cut_values, best_asy_sig_values = get_best_cut_values_on_max_score_for_n_cuts(args.n_cuts, samples_input, base_path)

    # store the categorization events
    #store_categorization_events_with_score(base_path, best_cut_values)

    # plot the categories
    plot_categories(best_cut_values, samples_input)





        


                





