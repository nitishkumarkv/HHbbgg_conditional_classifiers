import awkward as ak
from tools import load_samples, asymptotic_significance, approx_significance
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import mplhep

def categorize_using_n_cluster(n_clusters, samples_input, out_dir, clustering_method = "kmeans"):
     
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{n_clusters}_clusters")
    os.makedirs(save_path, exist_ok=True)

    # Extract the score
    y = np.stack(samples_input["score"])

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    # Perform clustering
    if clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(y_scaled)
    elif clustering_method == "SpectralClustering":
        sample_size = int(0.001 * len(y_scaled))  
        indices = np.random.choice(len(y_scaled), size=sample_size, replace=False)
        y_subset = y_scaled[indices]
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, random_state=42, verbose=True, n_jobs=2)

        # first fit and then get predictions for the entire dataset
        spectral.fit(y_subset)
        cluster_labels = spectral.predict(y_scaled)
    elif clustering_method == "GaussianMixture":
        sample_size = int(0.01 * len(y_scaled))  
        indices = np.random.choice(len(y_scaled), size=sample_size, replace=False)
        y_subset = y_scaled[indices]
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(y_scaled)
        cluster_labels = gmm.predict(y_scaled)

    #samples_input[f"{n_clusters}_cluster_idx"] = cluster_labels

    Z, s, b, b_side_band, sorted_clusters, diphoton_mass, dijet_mass = calculate_s_sb_per_cluster(samples_input, cluster_labels)

    s_sb_sorted, s_sorted, b_sorted = plot_s_sb(Z, s, b_side_band, f"{save_path}/{n_clusters}_asy_significance.png")

    return 0
    
def calculate_s_sb_per_cluster(samples_input, cluster_labels):
    """Calculate S/sqrt(B) for each cluster and also compute uncertainties for S and B."""
    unique_clusters = np.unique(cluster_labels)
    Z = {}
    s, b, b_side_band= {}, {}, {}
    diphoton_mass, dijet_mass = {}, {}


    for cluster in unique_clusters:
        cluster_indices = (cluster_labels == cluster)

        # Extract signal and background weights for this cluster
        #s_values_under_peak = samples_input[cluster_indices & (samples_input.labels == 1)  & (samples_input["dijet_mass"] > 120) & (samples_input["dijet_mass"] < 130)]
        #b_values_under_peak = samples_input[cluster_indices & (samples_input.labels == 0)  & (samples_input["dijet_mass"] > 120) & (samples_input["dijet_mass"] < 130)]
        #b_values_side_band = samples_input[cluster_indices & (samples_input.labels == 0)  & ((samples_input["dijet_mass"] < 120) | (samples_input["dijet_mass"] > 130))]
        s_values_under_peak = samples_input[cluster_indices & (samples_input.labels == 1)  & (samples_input["diphoton_mass"] > 120) & (samples_input["diphoton_mass"] < 130)]
        b_values_under_peak = samples_input[cluster_indices & (samples_input.labels == 0)  & (samples_input["diphoton_mass"] > 120) & (samples_input["diphoton_mass"] < 130)]
        b_values_side_band = samples_input[cluster_indices & (samples_input.labels == 0)  & ((samples_input["diphoton_mass"] < 120) | (samples_input["diphoton_mass"] > 130))]
        diphoton_mass[cluster] = samples_input[cluster_indices]["diphoton_mass"]
        dijet_mass[cluster] = samples_input[cluster_indices]["dijet_mass"]

        num_sig_under_peak = s_values_under_peak["weights"].sum()
        num_bkg_under_peak = b_values_under_peak["weights"].sum()
        num_bkg_side_band = b_values_side_band["weights"].sum()

        Z_value = np.sqrt(2 * ((num_sig_under_peak + num_bkg_under_peak + 1e-10) * np.log(1 + num_sig_under_peak / (num_bkg_under_peak + 1e-10)) - num_sig_under_peak))

        s[cluster] = num_sig_under_peak
        b[cluster] = num_bkg_under_peak
        b_side_band[cluster] = num_bkg_side_band
        Z[cluster] = Z_value

    sorted_clusters = sorted(Z.keys(), key=lambda cluster: Z[cluster], reverse=True)
    #print(sorted_clusters)
    #print([Z[cluster] for cluster in sorted_clusters])

    return Z, s, b, b_side_band, sorted_clusters, diphoton_mass, dijet_mass

def plot_s_sb(s_sb, s, b, save_path):
    """Plot S/sqrt(B), S, and B distributions with uncertainties."""
    #plt.style.use(mplhep.style.CMS)  # Use CMS style

    clusters = list(s_sb.keys())

    s_sb_values = [s_sb[c] for c in clusters]
    s_values = [s[c] for c in clusters]
    b_values = [b[c] for c in clusters]

    # Sort by S/sqrt(B) for consistent ordering
    sorted_indices = sorted(range(len(s_sb_values)), key=lambda i: s_sb_values[i], reverse=True)
    s_sb_sorted = [s_sb_values[i] for i in sorted_indices]
    s_sorted = [s_values[i] for i in sorted_indices]
    b_sorted = [b_values[i] for i in sorted_indices]

    ordered_idx = list(range(len(sorted_indices)))

    # Create figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 1, 1], 'hspace': 0})

    # Plot S/sqrt(B)
    axs[0].plot(ordered_idx, s_sb_sorted, ".b", markersize=10)
    axs[0].set_ylabel("Asymptotic Significance", labelpad=20, va='center')
    axs[0].grid(True)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot S with uncertainties
    axs[2].plot(ordered_idx, s_sorted, ".b", label="S", markersize=10)
    axs[2].set_ylabel("Signal under the peak", labelpad=20, va='center')
    axs[2].grid(True)
    axs[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot B with uncertainties
    axs[3].plot(ordered_idx, b_sorted, ".b", label="B", markersize=10)
    axs[3].set_xlabel("Cluster Index")
    axs[3].set_ylabel("Background in side bands", labelpad=20, va='center')
    axs[3].set_yscale("log")
    axs[3].grid(True)

    for i in range(min(3, len(b_sorted))):
        axs[3].annotate(
            f"{b_sorted[i]:.2f}",
            xy=(i, b_sorted[i]),
            xytext=(i, b_sorted[i] + (b_sorted[i] * 1)),
            fontsize=9,
            arrowprops=dict(color="black", arrowstyle="->"),
            ha="center"
        )

    # calculate Z sum in quadrature
    Z_sum_quad = []
    for i in range(1, len(s_sb_sorted)+1):
        Z_sum_quad.append(np.sqrt(np.sum(np.array(s_sb_sorted[:i])**2)))

    # Plot Z sum in quadrature
    axs[1].plot(ordered_idx, Z_sum_quad, ".b", markersize=10)
    axs[1].set_xlabel("Cluster Index")
    axs[1].set_ylabel("Z sum in quadrature", labelpad=20, va='center')
    axs[1].grid(True)
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return s_sb_sorted, s_sorted, b_sorted


if __name__ == "__main__":
    # load the samples
    base_path = "/net/data_cms3a-1/kasaraguppe/work/HHbbgg_classifier/HHbbgg_conditional_classifiers/MLP_inputs_20250218_with_mjj_mass/"
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
    print("len(samples_input)", len(samples_input))

    for n_clusters in [50, 100, 125, 150, 175, 200]:
        print(f"Processing {n_clusters} clusters")
        categorize_using_n_cluster(n_clusters, samples_input, f"{base_path}/knn_clustering_categorization")
    #categorize_using_n_cluster(10, samples_input, "spectral_clustering_categorization", clustering_method="GaussianMixture")
