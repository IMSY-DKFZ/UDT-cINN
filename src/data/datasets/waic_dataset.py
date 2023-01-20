from torch.utils.data import Dataset
import glob
import torch
import numpy as np
from typing import Union, Iterable, Sized
import os


class WAICDataset(Dataset):
    def __init__(self, root, noise_aug: bool = False, noise_std: float = 0.3, flip_aug: bool = True,
                 used_channels: Union[Iterable, Sized, int] = np.s_[:]
                 ):

        self.image_list = sorted(glob.glob(root))

        self.image_list_length = len(self.image_list)

        self.noise_aug = noise_aug
        self.noise_std = noise_std

        self.flip_aug = flip_aug

        self.used_channels = used_channels

    def __getitem__(self, index):
        path = self.image_list[index % self.image_list_length]
        if os.path.splitext(path)[1] == ".npz":
            data = np.load(path)
            img = torch.from_numpy(data["reconstruction"])
        else:
            img = torch.from_numpy(np.load(path))

        if len(img.size()) < 3:
            img = img.unsqueeze(0)

        if isinstance(self.used_channels, (slice, int)):
            img = img[self.used_channels, :, :].unsqueeze(0)
        elif isinstance(self.used_channels, (Iterable, Sized)):
            num_channels = len(self.used_channels)
            img_dims = img.size()
            new_img = torch.ones(num_channels, img_dims[1], img_dims[2])

            for channel_idx, used_channel in enumerate(self.used_channels):
                new_img[channel_idx, :, :] = img[used_channel, :, :]

            img = new_img

        if self.flip_aug:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, [2])

        # if self.noise_aug:
        #     img += torch.normal(0.0, self.noise_std, size=img.shape)

        return {"image": img.type(torch.float32)}

    def __len__(self):
        return self.image_list_length
