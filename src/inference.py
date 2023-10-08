import torch
import torch.backends.cudnn as cudnn
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from src.trainers import get_model
from src.data import get_data_module
from src.utils.config_io import load_config, get_conf_path
from src.utils.parser import DomainAdaptationParser
from src.visualization import col_bar

import matplotlib
matplotlib.use('Agg')

cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")

SAVE_DATA_PATH = os.environ["SAVE_DATA_PATH"]
DATA_BASE_PATH = os.environ["UDT_cINN_PROJECT_PATH"]

pretrained_models = glob.glob(os.path.join(DATA_BASE_PATH, "pretrained_models", "*", "*"))

for pretrained_model in pretrained_models:
    print(pretrained_model)

    if "cINN" in pretrained_model:
        EXPERIMENT_NAME = "gan_cinn"
    else:
        EXPERIMENT_NAME = "unit"

    if "HSI" in pretrained_model:
        EXPERIMENT_NAME += "_hsi"

    config_path = os.path.join(pretrained_model, "config.yaml")

    config = load_config(config_path)
    config.inference = True
    config.checkpoint = os.path.join(pretrained_model, "model_checkpoint.ckpt")

    config["save_path"] = pretrained_model
    config["data_base_path"] = os.path.join(DATA_BASE_PATH, "simulated_data")

    parser = DomainAdaptationParser(config=config)
    config = parser.get_new_config()

    model = get_model(experiment_name=EXPERIMENT_NAME)

    if "hsi" in EXPERIMENT_NAME:
        model = model.load_from_checkpoint(checkpoint_path=config.checkpoint, experiment_config=config).cuda()
        data_module, test_data_manager = get_data_module(experiment_name=EXPERIMENT_NAME)
        config.batch_size = 1000
        data_module = data_module(experiment_config=config)

        generated_spectrum_data_path = os.path.join(config.save_path, "generated_spectra_data")
        os.makedirs(generated_spectrum_data_path, exist_ok=True)

        with test_data_manager(data_module):
            for batch_idx, batch in enumerate(data_module.test_dataloader()):
                print(batch_idx)
                spectra_a, spectra_b = model.get_spectra(batch)

                spectra_ab = model.translate_spectrum(spectra_a, input_domain="a")

                spectra_a = spectra_a[0].detach().cpu().numpy() if isinstance(spectra_a, tuple) else spectra_a.detach().cpu().numpy()
                spectra_ab = spectra_ab[0].detach().cpu().numpy() if isinstance(spectra_ab, tuple) else spectra_ab.detach().cpu().numpy()

                spectra_a = spectra_a * config.data.std_a + config.data.mean_a
                spectra_ab = spectra_ab * config.data.std_b + config.data.mean_b

                # Uncomment this for saving the generated images
                # np.savez(os.path.join(generated_spectrum_data_path, f"test_file_{batch_idx}"),
                #          spectra_a=spectra_a,
                #          spectra_ab=spectra_ab,
                #          seg_a=batch["seg_a"].cpu().numpy(),
                #          subjects_a=batch["subjects_a"],
                #          image_ids_a=batch["image_ids_a"],
                #          )

                organ_label_a = batch["mapping"][str(int(batch["seg_a"][0].cpu()))]
                plt.figure(figsize=(6, 6))
                plt.plot(spectra_a[0], color="green", linestyle="solid", label=f"{organ_label_a} simulated spectrum")
                plt.plot(spectra_ab[0], color="blue", linestyle="dashed", label=f"{organ_label_a} Sim2Real spectrum")
                plt.legend()
                plt.savefig(os.path.join(generated_spectrum_data_path, f"test_file_{batch_idx}.png"))
                plt.close()
    else:
        config.data.dimensions = [16, 128, 256]
        model = model.load_from_checkpoint(checkpoint_path=config.checkpoint, experiment_config=config).cuda()
        files_path = os.path.join(DATA_BASE_PATH, "simulated_data", "PAT_Data", "good_simulations", "test", "*.npz")
        file_list = glob.glob(files_path)
        generated_image_data_path = os.path.join(config.save_path, "generated_image_data")
        os.makedirs(generated_image_data_path, exist_ok=True)
        for file_idx, file in enumerate(file_list):
            print(file_idx)
            data = np.load(file, allow_pickle=True)
            img_a = torch.unsqueeze(torch.from_numpy(data["reconstruction"]), 0).type(torch.float32)
            seg_a = torch.unsqueeze(torch.from_numpy(data["segmentation"]), 0).type(torch.float32)

            images_a = (img_a - config.data["mean_a"]) / config.data["std_a"]
            if config.condition == "segmentation":
                images_a = (images_a.cuda(), seg_a)
            else:
                images_a = images_a.cuda()
            images_ab = model.translate_image(images_a, input_domain="a")

            images_a = images_a[0].detach().cpu().numpy() if isinstance(images_a, tuple) else images_a.detach().cpu().numpy()
            images_ab = images_ab[0].detach().cpu().numpy() if isinstance(images_ab, tuple) else images_ab.detach().cpu().numpy()

            images_a = images_a * config.data.std_a + config.data.mean_a
            images_ab = images_ab * config.data.std_b + config.data.mean_b

            # Uncomment this for saving the generated images
            # np.savez(os.path.join(generated_image_data_path, f"test_file_{file_idx}"),
            #          images_a=images_a,
            #          images_ab=images_ab,
            #          seg_a=data["segmentation"],
            #          )

            plt.figure(figsize=(6, 6))
            plt.subplot(2, 1, 1)
            plt.title("Simulated image at 800 nm")
            img_a = plt.imshow(images_a[0, 10, :, :])
            col_bar(img_a)
            plt.subplot(2, 1, 2)
            plt.title("Sim2Real Image at 800 nm")
            img_ab = plt.imshow(images_ab[0, 10, :, :])
            col_bar(img_ab)
            plt.savefig(os.path.join(generated_image_data_path, f"test_image_{file_idx}.png"))
            plt.close()
