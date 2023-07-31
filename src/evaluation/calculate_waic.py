import numpy as np
import os
import glob

import pandas as pd
import torch
from pathlib import Path
from src.trainers import WAICTrainer
from src.utils.config_io import load_config
from src.visualization.templates import cmap_qualitative
import matplotlib.pyplot as plt
import seaborn as sns
from src import settings

from src.utils.susi import ExperimentResults

here = Path(__file__)
np.random.seed(141)

CALCULATE_WAIC = True


def load_data():

    test_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/validation"
    sim_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"
    gan_cinn_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/gan_cinn/2023_02_27_21_22_58/testing/generated_image_data"
    unit_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_01_21_20_14_58/testing/generated_image_data"

    test_real_data = glob.glob(os.path.join(test_real_data_root, "Forearm*.npz"))
    sim_data = glob.glob(os.path.join(sim_root, "*.npz"))
    gan_cinn_data = glob.glob(os.path.join(gan_cinn_root, "*.npz"))
    unit_data = glob.glob(os.path.join(unit_root, "*.npz"))

    datasets = {
        "real": test_real_data,
        "simulated": sim_data,
        "cINN": gan_cinn_data,
        "UNIT": unit_data,
    }

    for dataset_name, dataset_file_list in datasets.items():
        if dataset_name in ["real", "simulated"]:
            datasets[dataset_name] = [np.load(file)["reconstruction"] for file in dataset_file_list]
        elif dataset_name in ["cINN", "UNIT"]:
            datasets[dataset_name] = [np.load(file)["images_ab"] for file in dataset_file_list]

    return datasets


# def load_data_hsi():
#
#     test_real_data_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/real_images/validation"
#     sim_root = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/good_simulations/test"
#     gan_cinn_root = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/testing/generated_image_data"
#     unit_root = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_02_27_21_45_59/testing/generated_image_data"
#
#     test_real_data = glob.glob(os.path.join(test_real_data_root, "*.npz"))
#     sim_data = glob.glob(os.path.join(sim_root, "*.npz"))
#     gan_cinn_data = glob.glob(os.path.join(gan_cinn_root, "*.npz"))
#     unit_data = glob.glob(os.path.join(unit_root, "*.npz"))
#
#     datasets = {
#         "real": test_real_data,
#         "simulated": sim_data,
#         "cINN": gan_cinn_data,
#         "UNIT": unit_data,
#     }
#
#     for dataset_name, dataset_file_list in datasets.items():
#         if dataset_name in ["real", "simulated"]:
#             datasets[dataset_name] = [np.load(file)["reconstruction"] for file in dataset_file_list]
#         elif dataset_name in ["cINN", "UNIT"]:
#             datasets[dataset_name] = [np.load(file)["images_ab"] for file in dataset_file_list]
#
#     return datasets


def load_models():
    # not standardized
    # models = {
    #     "model_1": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_04_52",
    #     "model_2": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_01",
    #     "model_3": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_18",
    #     "model_4": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_32",
    #     "model_5": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_22_14_47_52",
    # }

    # models = {
    #     "model_1": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_08_34",
    #     "model_2": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_09_04",
    #     "model_3": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_09_35",
    #     "model_4": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_09_51",
    #     "model_5": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_10_20"
    # }

    # standardized
    models = {
        "model_1": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_31/version_0/checkpoints/epoch=399-step=14000.ckpt",
        "model_2": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_57/version_0/checkpoints/epoch=399-step=14000.ckpt",
        "model_3": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_09_04/version_0/checkpoints/epoch=799-step=28000.ckpt",
        "model_4": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_09_51/version_0/checkpoints/epoch=799-step=28000.ckpt",
        "model_5": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_22_15_10_20/version_0/checkpoints/epoch=799-step=28000.ckpt"
    }

    for model, path in models.items():
        # config = load_config(os.path.join(path, "version_0", "hparams.yaml"))
        config = load_config(os.path.join("/", *(path.split("/")[:-2]), "hparams.yaml"))
        # checkpoint = os.path.join(path, "version_0", "checkpoints", "epoch=799-step=28000.ckpt")
        checkpoint = path

        models[model] = WAICTrainer.load_from_checkpoint(checkpoint, experiment_config=config).cuda()

    return models, config


def calculate_waic_score():
    data = load_data()
    models, config = load_models()

    results = ExperimentResults()
    for dataset_name, data_list in data.items():
        for sample in data_list:
            input_tensor = torch.from_numpy(sample).type(torch.float32).cuda()
            if len(input_tensor.size()) in [1, 3]:
                input_tensor = torch.unsqueeze(input_tensor, 0)
            input_tensor = (input_tensor - torch.min(input_tensor))/(torch.max(input_tensor) - torch.min(input_tensor))
            # if config.normalization == "standardize":
            #     if "sim" in dataset_name:
            #         mean, std = 0.12278456474240872, 0.082442302659046
            #     else:
            #         mean, std = 0.1241266841789031, 0.07419423455146855
            #     input_tensor = (input_tensor - mean) / std

            model_results = [model.maximum_likelihood_loss(*model(input_tensor)).detach() for model in models.values()]
            model_results = [tensor.cpu().numpy() for tensor in model_results]

            waic = np.var(model_results) + np.mean(model_results)
            results.append(name='waic', value=float(waic))
            results.append(name='data', value=dataset_name)
    df = results.get_df()
    df.to_csv(settings.results_dir / 'waic_pa' / 'waic_values_pa.csv', index=True)

    return df


if __name__ == "__main__":
    # if CALCULATE_WAIC:
    scores = calculate_waic_score()
    # else:
    #     scores = pd.read_csv(settings.results_dir / 'waic_pa' / 'waic_values_pa.csv', index_col=0)

    sns.violinplot(data=scores, x="data", y="waic", hue="data", palette=cmap_qualitative, dodge=False)
    # plt.ylim(-0.4, -0.15)
    plt.show()