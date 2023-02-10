import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from src.utils.gather_pa_spectra_from_dataset import calculate_mean_spectrum
matplotlib.use('TkAgg')


data_set_1_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"
data_set_2_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/validation"
data_set_3_root = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/testing/training"
data_set_4_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_01_21_20_14_58/testing/training"

files_data_set_1 = glob.glob(os.path.join(data_set_1_root, "*.npz"))
files_data_set_2 = glob.glob(os.path.join(data_set_2_root, "*.npz"))
files_data_set_3 = glob.glob(os.path.join(data_set_3_root, "*.npz"))
files_data_set_4 = glob.glob(os.path.join(data_set_4_root, "*.npz"))

VISUALIZE_MEAN_SPECTRA = False
CLUSTER_SPECTRA = True
CLUSTERING = "PCA"


if __name__ == "__main__":
    sim_spectra = calculate_mean_spectrum(files_data_set_1, return_std=True)
    real_spectra = calculate_mean_spectrum(files_data_set_2, return_std=True)
    gan_cinn_spectra = calculate_mean_spectrum(files_data_set_3, return_std=True)
    # unit_spectra = calculate_mean_spectrum(files_data_set_4, return_std=True)

    print("Arteries: \n")
    print(f"gan_cinn: {np.mean(np.abs(real_spectra['mean_artery_spectra'] - gan_cinn_spectra['mean_artery_spectra']))}")
    # print(f"unit: {np.mean(np.abs(real_spectra['mean_artery_spectra'] - unit_spectra['mean_artery_spectra']))}")

    print("Veins: \n")
    print(f"gan_cinn: {np.mean(np.abs(real_spectra['mean_vein_spectra'] - gan_cinn_spectra['mean_vein_spectra']))}")
    # print(f"unit: {np.mean(np.abs(real_spectra['mean_vein_spectra'] - unit_spectra['mean_vein_spectra']))}")

    if CLUSTER_SPECTRA:
        if CLUSTERING == "PCA":
            for vessel in ["artery", "vein"]:
                pca = PCA(n_components=2).fit(real_spectra[f"{vessel}_spectra_all"])

                sim_pca = pca.transform(sim_spectra[f"{vessel}_spectra_all"])
                real_pca = pca.transform(real_spectra[f"{vessel}_spectra_all"])
                gan_cinn_pca = pca.transform(gan_cinn_spectra[f"{vessel}_spectra_all"])

                pca_data = np.concatenate([
                    sim_pca,
                    real_pca,
                    gan_cinn_pca
                ])

                pca_df = pd.DataFrame(data=pca_data, columns=(f"PCA component 1", f"PCA component 2"))
                pca_df["data_type"] = ["simulation"] * np.shape(sim_pca)[0] \
                                       + ["real"] * np.shape(real_pca)[0] \
                                       + ["gan_cinn"] * np.shape(gan_cinn_pca)[0]

                # plt.scatter(sim_pca[:, 0], sim_pca[:, 1], alpha=0.1, label="simulation")
                # plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.1, label="real")
                # plt.scatter(gan_cinn_pca[:, 0], gan_cinn_pca[:, 1], alpha=0.1, label="gan_cinn")
                # plt.legend()
                sns.jointplot(data=pca_df, x="PCA component 1", y="PCA component 2", hue="data_type", kind="kde", fill=True,
                              alpha=0.5)
                plt.suptitle(f"{vessel} PCA components")
                plt.show()

        elif CLUSTERING == "TSNE":
            for vessel in ["artery", "vein"]:
                tsne_data = np.concatenate([
                    sim_spectra[f"{vessel}_spectra_all"],
                    real_spectra[f"{vessel}_spectra_all"],
                    gan_cinn_spectra[f"{vessel}_spectra_all"]
                                            ])

                tsne_data = TSNE(n_components=2).fit_transform(tsne_data)

                tsne_df = pd.DataFrame(data=tsne_data, columns=(f"TSNE_dim_1", f"TSNE_dim_2"))
                tsne_df["data_type"] = ["simulation"] * np.shape(sim_spectra[f"{vessel}_spectra_all"])[0] \
                                       + ["real"] * np.shape(real_spectra[f"{vessel}_spectra_all"])[0] \
                                       + ["gan_cinn"] * np.shape(gan_cinn_spectra[f"{vessel}_spectra_all"])[0]

                shape_0 = np.shape(sim_spectra[f"{vessel}_spectra_all"])[0]
                shape_1 = np.shape(gan_cinn_spectra[f"{vessel}_spectra_all"])[0]

                # plt.scatter(tsne_data[:shape_0, 0],
                #             tsne_data[:shape_0, 1],
                #             alpha=0.5, label="simulation")
                #
                # plt.scatter(tsne_data[shape_0:-shape_1, 0],
                #             tsne_data[shape_0:-shape_1, 1],
                #             alpha=0.5, label="real")
                #
                # plt.scatter(tsne_data[-shape_1:, 0],
                #             tsne_data[-shape_1:, 1],
                #             alpha=0.5, label="gan_cinn")

                # sns.kdeplot(x=tsne_data[:shape_0, 0], y=tsne_data[:shape_0, 1], cmap="Blues", fill=True)
                # sns.kdeplot(x=tsne_data[shape_0:-shape_1, 0], y=tsne_data[shape_0:-shape_1, 1], cmap="Reds", fill=True)
                # sns.kdeplot(x=tsne_data[-shape_1:, 0], y=tsne_data[-shape_1:, 1], cmap="Greens", fill=True)

                sns.jointplot(data=tsne_df, x="TSNE_dim_1", y="TSNE_dim_2", hue="data_type", kind="kde", fill=True, alpha=0.5)
                plt.suptitle(f"{vessel} TSNE components")
                plt.show()

    if VISUALIZE_MEAN_SPECTRA:
        wavelengths = np.arange(700, 855, 10)
        plt.subplot(4, 1, 1)
        plt.plot(wavelengths, sim_spectra["mean_vein_spectra"], label="sim veins")
        plt.fill_between(wavelengths, sim_spectra["mean_vein_spectra"] - sim_spectra["std_vein_spectra"],
                         sim_spectra["mean_vein_spectra"] + sim_spectra["std_vein_spectra"], alpha=0.3)
        plt.plot(wavelengths, sim_spectra["mean_artery_spectra"], label="sim arteries")
        plt.fill_between(wavelengths, sim_spectra["mean_artery_spectra"] - sim_spectra["std_artery_spectra"],
                         sim_spectra["mean_artery_spectra"] + sim_spectra["std_artery_spectra"], alpha=0.3)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(wavelengths, real_spectra["mean_vein_spectra"], label="real veins")
        plt.fill_between(wavelengths, real_spectra["mean_vein_spectra"] - real_spectra["std_vein_spectra"],
                         real_spectra["mean_vein_spectra"] + real_spectra["std_vein_spectra"], alpha=0.3)
        plt.plot(wavelengths, real_spectra["mean_artery_spectra"], label="real arteries")
        plt.fill_between(wavelengths, real_spectra["mean_artery_spectra"] - real_spectra["std_artery_spectra"],
                         real_spectra["mean_artery_spectra"] + real_spectra["std_artery_spectra"], alpha=0.3)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(wavelengths, gan_cinn_spectra["mean_vein_spectra"], label="gan_cinn veins")
        plt.fill_between(wavelengths, gan_cinn_spectra["mean_vein_spectra"] - gan_cinn_spectra["std_vein_spectra"],
                         gan_cinn_spectra["mean_vein_spectra"] + gan_cinn_spectra["std_vein_spectra"], alpha=0.3)
        plt.plot(wavelengths, gan_cinn_spectra["mean_artery_spectra"], label="gan_cinn arteries")
        plt.fill_between(wavelengths, gan_cinn_spectra["mean_artery_spectra"] - gan_cinn_spectra["std_artery_spectra"],
                         gan_cinn_spectra["mean_artery_spectra"] + gan_cinn_spectra["std_artery_spectra"], alpha=0.3)
        plt.legend()

        # plt.subplot(4, 1, 4)
        # plt.plot(wavelengths, unit_spectra["mean_vein_spectra"], label="unit veins")
        # plt.fill_between(wavelengths, unit_spectra["mean_vein_spectra"] - unit_spectra["std_vein_spectra"],
        #                  unit_spectra["mean_vein_spectra"] + unit_spectra["std_vein_spectra"], alpha=0.3)
        # plt.plot(wavelengths, unit_spectra["mean_artery_spectra"], label="unit arteries")
        # plt.fill_between(wavelengths, unit_spectra["mean_artery_spectra"] - unit_spectra["std_artery_spectra"],
        #                  unit_spectra["mean_artery_spectra"] + unit_spectra["std_artery_spectra"], alpha=0.3)
        # plt.legend()
        plt.show()
        plt.close()

        plt.subplot(2, 1, 1)
        plt.title("Artery spectra")

        plt.plot(wavelengths, sim_spectra["mean_artery_spectra"], label="sim arteries")
        plt.fill_between(wavelengths, sim_spectra["mean_artery_spectra"] - sim_spectra["std_artery_spectra"],
                         sim_spectra["mean_artery_spectra"] + sim_spectra["std_artery_spectra"], alpha=0.3)
        plt.plot(wavelengths, real_spectra["mean_artery_spectra"], label="real arteries")
        plt.fill_between(wavelengths, real_spectra["mean_artery_spectra"] - real_spectra["std_artery_spectra"],
                         real_spectra["mean_artery_spectra"] + real_spectra["std_artery_spectra"], alpha=0.3)

        plt.plot(wavelengths, gan_cinn_spectra["mean_artery_spectra"], label="gan_cinn arteries")
        plt.fill_between(wavelengths, gan_cinn_spectra["mean_artery_spectra"] - gan_cinn_spectra["std_artery_spectra"],
                         gan_cinn_spectra["mean_artery_spectra"] + gan_cinn_spectra["std_artery_spectra"], alpha=0.3)

        # plt.plot(wavelengths, unit_spectra["mean_artery_spectra"], label="unit arteries")
        # plt.fill_between(wavelengths, unit_spectra["mean_artery_spectra"] - unit_spectra["std_artery_spectra"],
        #                  unit_spectra["mean_artery_spectra"] + unit_spectra["std_artery_spectra"], alpha=0.3)

        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Vein spectra")

        plt.plot(wavelengths, sim_spectra["mean_vein_spectra"], label="sim veins")
        plt.fill_between(wavelengths, sim_spectra["mean_vein_spectra"] - sim_spectra["std_vein_spectra"],
                         sim_spectra["mean_vein_spectra"] + sim_spectra["std_vein_spectra"], alpha=0.3)

        plt.plot(wavelengths, real_spectra["mean_vein_spectra"], label="real veins")
        plt.fill_between(wavelengths, real_spectra["mean_vein_spectra"] - real_spectra["std_vein_spectra"],
                         real_spectra["mean_vein_spectra"] + real_spectra["std_vein_spectra"], alpha=0.3)

        plt.plot(wavelengths, gan_cinn_spectra["mean_vein_spectra"], label="gan_cinn veins")
        plt.fill_between(wavelengths, gan_cinn_spectra["mean_vein_spectra"] - gan_cinn_spectra["std_vein_spectra"],
                         gan_cinn_spectra["mean_vein_spectra"] + gan_cinn_spectra["std_vein_spectra"], alpha=0.3)

        # plt.plot(wavelengths, unit_spectra["mean_vein_spectra"], label="unit veins")
        # plt.fill_between(wavelengths, unit_spectra["mean_vein_spectra"] - unit_spectra["std_vein_spectra"],
        #                  unit_spectra["mean_vein_spectra"] + unit_spectra["std_vein_spectra"], alpha=0.3)

        plt.legend()

        plt.show()
        plt.close()
