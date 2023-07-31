import glob
import numpy as np
import torch
import os
from src.utils.gather_da_result_data import gather_da_result_data
from src.utils.config_io import load_config
from src.trainers import WAICTrainer, WAICTrainerHSI


def load_waic_models(data_type: str):

    if data_type == "HSI":
        models = {
            "model_1": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_04_52",
            "model_2": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_01",
            "model_3": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_18",
            "model_4": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_21_23_05_32",
            "model_5": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic_hsi/2023_02_22_14_47_52",
        }
        trainer = WAICTrainerHSI

    elif data_type == "PAI":
        models = {
            "model_1": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_29",
            "model_2": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_31",
            "model_3": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_51",
            "model_4": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_08_57",
            "model_5": "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/waic/2023_02_21_23_09_06",
        }
        trainer = WAICTrainer

    else:
        raise KeyError("Please select a valid 'data_type' from 'HSI' or 'PAI'!")

    configs = list()
    for model, path in models.items():
        config = load_config(os.path.join(path, "version_0", "hparams.yaml"))
        checkpoint_dir = os.path.join(path, "version_0", "checkpoints")
        last_checkpoint = sorted(os.listdir(checkpoint_dir))
        last_checkpoint = last_checkpoint[-5]
        checkpoint = os.path.join(checkpoint_dir, last_checkpoint)

        models[model] = trainer.load_from_checkpoint(checkpoint, experiment_config=config).cuda()
        configs.append(config)

    return models, configs


def get_model_data(model_type_path: str, data_type: str = "PAI") -> dict:

    model_paths = glob.glob(os.path.join(model_type_path, "*"))

    model_data_dict = dict()

    for model_path in model_paths:
        try:
            model_timestamp = os.path.split(model_path)[-1]
            model_data_dict[model_timestamp] = gather_da_result_data(model_path)
        except FileNotFoundError:
            continue

    return model_data_dict


def maximum_likelihood_loss(z: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
    """
    Computes the maximum likelihood loss.

    :param z: Latent space representation of the input spectrum.
    :param jac: Jacobian of the input spectrum.
    :return: Maximum likelihood loss,
    """

    p = torch.sum(z ** 2, dim=1)
    loss = 0.5 * p - jac
    ml_loss = loss / np.prod(list(z.size()))

    return ml_loss


def calculate_waic_scores(base_path: str):

    data_type = "HSI" if "hsi" in base_path else "PAI"
    data_key = "spectra_ab" if data_type == "HSI" else "images_ab"

    model_generated_data = get_model_data(base_path, data_type)
    waic_models, configs = load_waic_models(data_type)
    waic_results = {model_name: {"waic_results": list()} for model_name in model_generated_data}

    for model_name, model_data in model_generated_data.items():

        for sample_idx, sample in enumerate(model_data):
            input_tensor = torch.from_numpy(sample[data_key]).type(torch.float32).cuda()
            if data_type == "PAI" and len(input_tensor.size()) == 3:
                input_tensor = torch.unsqueeze(input_tensor, 0)
            if data_type == "HSI" and len(input_tensor.size()) == 1:
                input_tensor = torch.unsqueeze(input_tensor, 0)

            model_results = list()
            for (waic_model_name, waic_model), config in zip(waic_models.items(), configs):
                if config.normalization == "standardize":
                    mean, std = config.data.mean_b, config.data.std_b
                    normalized_input_tensor = (input_tensor - mean) / std
                else:
                    normalized_input_tensor = input_tensor

                waic_values = maximum_likelihood_loss(*waic_model(normalized_input_tensor)).detach().cpu().numpy()
                model_results.append([sample_value for sample_value in waic_values])

            model_results = np.array(model_results)
            waic_results[model_name]["waic_results"].extend(np.var(model_results, axis=0) + np.mean(model_results, axis=0))

    for model_name, waic_result in waic_results.items():
        waic_results[model_name]["mean"] = np.mean(waic_result["waic_results"])
        waic_results[model_name]["median"] = np.median(waic_result["waic_results"])
        waic_results[model_name]["std"] = np.std(waic_result["waic_results"])

    ranking = [(model_name, waic_results[model_name]["mean"]) for model_name in waic_results]
    ranking.sort(key=lambda x: x[1])
    print(ranking)


if __name__ == "__main__":
    path = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/gan_cinn_hsi"
    calculate_waic_scores(path)

