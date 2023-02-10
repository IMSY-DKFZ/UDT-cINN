import numpy as np
import matplotlib.pyplot as plt


def calculate_mean_spectrum(image_files: list, return_std: bool = True, visualize: bool = False):

    artery_spectra_all = list()
    vein_spectra_all = list()

    for im_idx, image_file in enumerate(image_files):
        data = np.load(image_file)
        try:
            seg = np.squeeze(data["segmentation"])
        except KeyError:
            continue

        ms_image = np.squeeze(data["reconstruction"])
        if im_idx == 0 and visualize:
            plt.subplot(1, 2, 1)
            plt.imshow(seg)
            plt.subplot(1, 2, 2)
            plt.imshow(ms_image[0, :, :])
            plt.show()

        vein_spectra = ms_image[:, seg == 5]

        if vein_spectra.size:
            norm = np.linalg.norm(vein_spectra, axis=0)
            vein_spectra /= norm

            vein_spectra_all.extend([vein_spectra[:, spectrum] for spectrum in range(np.shape(vein_spectra)[1])])

        artery_spectra = ms_image[:, seg == 6]

        if artery_spectra.size:
            artery_spectra /= np.linalg.norm(artery_spectra, axis=0)

            artery_spectra_all.extend([artery_spectra[:, spectrum] for spectrum in range(np.shape(artery_spectra)[1])])

    artery_spectra_all = np.array(artery_spectra_all)
    vein_spectra_all = np.array(vein_spectra_all)

    mean_artery_spectrum = np.mean(artery_spectra_all, axis=0)
    mean_vein_spectrum = np.mean(vein_spectra_all, axis=0)

    return_dict = {
        "artery_spectra_all": artery_spectra_all,
        "vein_spectra_all": vein_spectra_all,
        "mean_artery_spectra": mean_artery_spectrum,
        "mean_vein_spectra": mean_vein_spectrum,
    }

    if return_std:
        std_artery_spectrum = np.std(np.array(artery_spectra_all), axis=0)
        std_vein_spectrum = np.std(np.array(vein_spectra_all), axis=0)

        return_dict["std_artery_spectra"] = std_artery_spectrum
        return_dict["std_vein_spectra"] = std_vein_spectrum

    return return_dict
