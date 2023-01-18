import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import os
from torch.utils.data import DataLoader
from src.data.datasets import WAICDataset
from src.utils.collate_function import collate
from typing import Iterable, Sized


class WAICDataModule(pl.LightningDataModule):
    def __init__(self, experiment_config: DictConfig):
        super().__init__()

        self.exp_config = experiment_config
        self.data_dir = self.exp_config.data.data_dir_b

        self.data_config = OmegaConf.load(os.path.join(self.data_dir, "data_config.yaml"))

        self.used_channels = self.exp_config.data.used_channels
        orig_dims = self.data_config.dims
        if isinstance(self.used_channels, int):
            self.channels = 1
            orig_dims[0] = self.channels
        elif isinstance(self.used_channels, (Iterable, Sized)):
            self.channels = len(self.used_channels)
            orig_dims[0] = self.channels
        else:
            self.channels = orig_dims[0]

        self.dimensions = orig_dims

        self.batch_size = self.exp_config.batch_size
        self.training_data, self.validation_data, self.test_data = None, None, None

        self.adjust_experiment_config()

    def setup(self, stage: str = None):
        self.training_data = WAICDataset(root=os.path.join(self.data_dir, "training/*.np*"),
                                         used_channels=self.used_channels)

        self.validation_data = WAICDataset(root=os.path.join(self.data_dir, "validation/*.np*"),
                                           used_channels=self.used_channels)

        self.test_data = WAICDataset(root=os.path.join(self.data_dir, "test/*.np*"),
                                     used_channels=self.used_channels)

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size,
                          shuffle=self.exp_config.shuffle, num_workers=self.exp_config.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.exp_config.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1,
                          num_workers=self.exp_config.num_workers, pin_memory=True)

    def adjust_experiment_config(self):
        self.exp_config.data.dimensions = self.dimensions
