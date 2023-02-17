from src.trainers import GanCondinitionalDomainAdaptationINN
from src.utils.config_io import load_config
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.visualization import col_bar

config_path = "/home/kris/Work/Data/DA_results/gan_cinn/2023_01_23_22_47_44/version_0/hparams.yaml"
config = load_config(config_path)
chkpt = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/gan_cinn/2023_01_31_18_34_48/version_0/checkpoints/epoch=299-step=213000.ckpt"

comparison_path = "/home/kris/Work/Data/domain_adaptation_simulations/min_max_preprocessed_data_sqrt_ms/comparison.npz"

# model = GanCondinitionalDomainAdaptationINN(experiment_config=config)
model = GanCondinitionalDomainAdaptationINN.load_from_checkpoint(chkpt, experiment_config=config)
model = model.cuda()

comparison_data = np.load(comparison_path)

simulated_image = comparison_data["sim"]
simulated_image = (simulated_image - config.data.mean_a)/config.data.std_a
simulated_image = torch.from_numpy(simulated_image).type(torch.float32).unsqueeze(0)

simulated_seg = comparison_data["sim_seg"]
simulated_seg = torch.from_numpy(simulated_seg).type(torch.float32).unsqueeze(0)

input = (simulated_image.cuda(), simulated_seg)

translated_image = model.translate_image(input)
translated_image = np.squeeze(translated_image[0].detach().cpu().numpy())
translated_image = translated_image * config.data.std_b + config.data.mean_b

fig = plt.figure(figsize=(12, 7))
plt.subplot(1, 3, 1)
plt.title("Real image")
im = plt.imshow(comparison_data["real_im"][0, :, :])
col_bar(im)
plt.subplot(1, 3, 2)
plt.title("Simulated image")
im = plt.imshow(comparison_data["sim"][0, :, :])
col_bar(im)
plt.subplot(1, 3, 3)
plt.title("Translated image")
im = plt.imshow(translated_image[0, :, :])
col_bar(im)
plt.show()






