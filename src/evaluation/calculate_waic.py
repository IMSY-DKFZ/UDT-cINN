import numpy as np
import os
import glob
import torch
from pathlib import Path
from src.trainers import WAICTrainer
from src.utils.config_io import load_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

here = Path(__file__)
np.random.seed(141)


def load_data():

    test_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/validation"
    sim_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"
    gan_cinn_root = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/testing/generated_image_data"
    unit_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_01_21_20_14_58/testing/generated_image_data"

    test_real_data = glob.glob(os.path.join(test_real_data_root, "Forearm*.npz"))
    sim_data = glob.glob(os.path.join(sim_root, "*.npz"))
    gan_cinn_data = glob.glob(os.path.join(gan_cinn_root, "*.npz"))
    unit_data = glob.glob(os.path.join(unit_root, "*.npz"))

    datasets = {
        "test_real_images": test_real_data,
        "sim_images": sim_data,
        "gan_cinn_images": gan_cinn_data,
        "unit_images": unit_data,
    }

    for dataset_name, dataset_file_list in datasets.items():
        if dataset_name in ["test_real_images", "sim_images"]:
            datasets[dataset_name] = [np.load(file)["reconstruction"] for file in dataset_file_list[:10]]
        elif dataset_name in ["gan_cinn_images", "unit_images"]:
            datasets[dataset_name] = [np.load(file)["images_ab"] for file in dataset_file_list[:10]]

    return datasets


def load_models():
    models = {
        "model_1": "/home/kris/Work/Data/DA_results/waic/2023_02_17_11_51_25",
        "model_2": "/home/kris/Work/Data/DA_results/waic/2023_02_17_13_35_34",
        "model_3": "/home/kris/Work/Data/DA_results/waic/2023_02_17_13_35_51",
        "model_4": "/home/kris/Work/Data/DA_results/waic/2023_02_17_17_46_04",
        "model_5": "/home/kris/Work/Data/DA_results/waic/2023_02_17_17_46_14",
    }

    for model, path in models.items():
        config = load_config(os.path.join(path, "version_0", "hparams.yaml"))
        checkpoint = os.path.join(path, "version_0", "checkpoints", "epoch=399-step=14400.ckpt")

        models[model] = WAICTrainer.load_from_checkpoint(checkpoint, experiment_config=config).cuda()

    return models


def calculate_waic_score():
    data = load_data()
    models = load_models()

    results_dict = {dataset_name: list() for dataset_name in data}

    for dataset_name, data_list in data.items():
        for sample in data_list:
            input_tensor = torch.from_numpy(sample).type(torch.float32).cuda()
            if len(input_tensor.size()) == 3:
                input_tensor = torch.unsqueeze(input_tensor, 0)

            model_results = [model.maximum_likelihood_loss(*model(input_tensor)).detach() for model in models.values()]
            model_results = [tensor.cpu().numpy() for tensor in model_results]

            results_dict[dataset_name].append(np.var(model_results) + np.mean(model_results))

    df = pd.melt(pd.DataFrame({key: np.array(value) for key, value in results_dict.items()}),
                 var_name="Model", value_name="WAIC Score")

    sns.violinplot(data=df, x="Model", y="WAIC Score", hue="Model", palette="viridis", dodge=False)
    plt.show()


if __name__ == "__main__":
    calculate_waic_score()
