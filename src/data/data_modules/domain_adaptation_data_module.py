import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import os
from torch.utils.data import DataLoader
from src.data.datasets import DomainAdaptationDataset
from src.utils.collate_function import collate
from typing import Iterable, Sized


class DomainAdaptationDataModule(pl.LightningDataModule):
    def __init__(self, experiment_config: DictConfig):
        super().__init__()

        self.exp_config = experiment_config
        self.data_dir_a = self.exp_config.data.data_dir_a
        self.data_dir_b = self.exp_config.data.data_dir_b

        self.data_config_a = OmegaConf.load(os.path.join(self.data_dir_a, "data_config.yaml"))
        self.data_config_b = OmegaConf.load(os.path.join(self.data_dir_b, "data_config.yaml"))

        self.prepare_data()

        self.used_channels = self.exp_config.data.used_channels
        orig_dims = self.data_config_a.dims
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

    def prepare_data(self):
        if not self.check_data_consistency():
            raise ValueError("The two datasets for the domain adaptation are not consistent in their dimensions!")

    def check_data_consistency(self):
        dimension_consistency = (self.data_config_a.dims == self.data_config_b.dims)

        return True  # dimension_consistency removed check because of inconsistencies in pl versions

    def setup(self, stage: str = None):
        self.training_data = DomainAdaptationDataset(root_a=os.path.join(self.data_dir_a, "training/*.np*"),
                                                     root_b=os.path.join(self.data_dir_b, "training/*.np*"),
                                                     experiment_config=self.exp_config,
                                                     )
        self.validation_data = DomainAdaptationDataset(root_a=os.path.join(self.data_dir_a, "validation/*.np*"),
                                                       root_b=os.path.join(self.data_dir_b, "validation/*.np*"),
                                                       experiment_config=self.exp_config,
                                                       )
        self.test_data = DomainAdaptationDataset(root_a=os.path.join(self.data_dir_a, "test/*.np*"),
                                                 root_b=os.path.join(self.data_dir_b, "test/*.np*"),
                                                 experiment_config=self.exp_config,
                                                 )

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size,
                          shuffle=self.exp_config.shuffle, num_workers=self.exp_config.num_workers, pin_memory=True,
                          drop_last=True,
                          collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.exp_config.num_workers, pin_memory=True,
                          drop_last=True,
                          collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1,
                          num_workers=self.exp_config.num_workers, pin_memory=True,
                          drop_last=True,
                          collate_fn=collate)

    def adjust_experiment_config(self):
        self.exp_config.data.dimensions = self.dimensions
        self.exp_config.data.mean_a = self.data_config_a.mean
        self.exp_config.data.mean_b = self.data_config_b.mean
        self.exp_config.data.std_a = self.data_config_a.std
        self.exp_config.data.std_b = self.data_config_b.std
        self.exp_config.data.n_classes = self.data_config_b.n_classes
        self.exp_config.data.class_prevalences = self.data_config_b.class_prevalences

