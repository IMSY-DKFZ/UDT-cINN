import os

from torch.utils.data import Dataset
import glob
import torch
import numpy as np
from typing import Iterable, Sized
from omegaconf import DictConfig


class DomainAdaptationDataset(Dataset):
    def __init__(self, root_a: str, root_b: str, experiment_config: DictConfig,
                 noise_aug: bool = False, noise_std: float = 0.3):

        self.image_list_a = sorted(glob.glob(root_a))
        self.image_list_b = sorted(glob.glob(root_b))

        self.image_list_a_length = len(self.image_list_a)
        self.image_list_b_length = len(self.image_list_b)

        self.exp_config = experiment_config

        self.noise_aug = noise_aug
        self.noise_std = noise_std

        self.used_channels = experiment_config.data.used_channels

        if experiment_config.normalization not in ["None", "none"]:
            self.normalization = experiment_config.normalization
        else:
            self.normalization = False

    def __getitem__(self, index):
        path_a = self.image_list_a[index % self.image_list_a_length]
        if os.path.splitext(path_a)[1] == ".npz":
            data = np.load(path_a, allow_pickle=True)
            img_a = torch.from_numpy(data["reconstruction"])
            seg_a = data["segmentation"] if "segmentation" in data else 0
            oxy_a = data["oxygenation"] if "oxygenation" in data else 0
        else:
            img_a = torch.from_numpy(np.load(path_a))
            oxy_a, seg_a = 0, 0
        if len(img_a.size()) < 3:
            img_a = img_a.unsqueeze(0)

        path_b = self.image_list_b[index % self.image_list_b_length]
        if os.path.splitext(path_b)[1] == ".npz":
            data = np.load(path_b, allow_pickle=True)
            img_b = torch.from_numpy(data["reconstruction"])
            seg_b = data["segmentation"] if "segmentation" in data else 0
            oxy_b = data["oxygenation"] if "oxygenation" in data else 0
        else:
            img_b = torch.from_numpy(np.load(path_b))
            oxy_b, seg_b = 0, 0
        if len(img_b.size()) < 3:
            img_b = img_b.unsqueeze(0)

        if isinstance(self.used_channels, (slice, int)):
            img_a = img_a[self.used_channels, :, :].unsqueeze(0)
            img_b = img_b[self.used_channels, :, :].unsqueeze(0)
        elif isinstance(self.used_channels, (Iterable, Sized)):
            num_channels = len(self.used_channels)
            img_dims = img_a.size()
            new_img_a = torch.ones(num_channels, img_dims[1], img_dims[2])
            new_img_b = torch.ones(num_channels, img_dims[1], img_dims[2])

            for channel_idx, used_channel in enumerate(self.used_channels):
                new_img_a[channel_idx, :, :] = img_a[used_channel, :, :]
                new_img_b[channel_idx, :, :] = img_b[used_channel, :, :]

            img_a = new_img_a
            img_b = new_img_b
        else:
            raise ValueError("attribute used_channels must be of the type slice, int iterable or Sized.")

        # normalization
        if self.normalization:
            if self.normalization == "standardize":
                img_a = (img_a - self.exp_config.data["mean_a"]) / self.exp_config.data["std_a"]
                img_b = (img_b - self.exp_config.data["mean_b"]) / self.exp_config.data["std_b"]

        # image flipping
        if torch.rand(1).item() < 0.5:
            img_a = torch.flip(img_a, [2])
            param_dict = {"seg": seg_a, "oxy": oxy_a}
            for key, value in param_dict.items():
                if not isinstance(value, int):
                    if len(np.shape(value)) < 3:
                        value = np.expand_dims(value, 0)
                    param_dict[key] = np.flip(value, [2])

            seg_a, oxy_a = param_dict["seg"], param_dict["oxy"]
        if torch.rand(1).item() < 0.5:
            img_b = torch.flip(img_b, [2])
            param_dict = {"seg": seg_b, "oxy": oxy_b}
            for key, value in param_dict.items():
                if not isinstance(value, int):
                    if len(np.shape(value)) < 3:
                        value = np.expand_dims(value, 0)
                    param_dict[key] = np.flip(value, [2])

            seg_b, oxy_b = param_dict["seg"], param_dict["oxy"]

        if self.noise_aug:
            img_a += torch.normal(0.0, self.noise_std, size=img_a.shape)
            img_b += torch.normal(0.0, self.noise_std, size=img_b.shape)

        return {"image_a": img_a.type(torch.float32), "image_b": img_b.type(torch.float32),
                "seg_a": seg_a.squeeze(), "seg_b": seg_b,
                "oxy_a": oxy_a.squeeze(), "oxy_b": oxy_b}

    def __len__(self):
        return min(self.image_list_a_length, self.image_list_b_length)
