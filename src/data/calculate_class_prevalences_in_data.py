import numpy as np
import os
import glob
from omegaconf import DictConfig

from src.utils.config_io import load_config, save_config


def calculate_class_prevalences_in_data(dataset_path):
    data_config = load_config(os.path.join(dataset_path, "data_config.yaml"))
    n_classes = data_config.n_classes

    class_prevalences = np.zeros(n_classes)

    file_list = glob.glob(os.path.join(dataset_path, "training", "*.np*"))

    for file in file_list:
        segmentation = np.load(file)["segmentation"]
        unique, counts = np.unique(segmentation, return_counts=True)

        for label_index, occurring_label in enumerate(unique):
            class_prevalences[int(occurring_label)] += counts[int(label_index)]

    class_prevalences /= np.sum(class_prevalences)

    data_config["class_prevalences"] = DictConfig({label: float(prob) for label, prob in enumerate(class_prevalences)})

    save_config(os.path.join(dataset_path, "data_config.yaml"), data_config)

    print(list(data_config.class_prevalences.values()))


if __name__ == "__main__":
    calculate_class_prevalences_in_data("/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations")