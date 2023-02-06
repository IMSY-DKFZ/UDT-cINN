import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from omegaconf import DictConfig


class DomainAdaptationDatasetHSI(Dataset):
    def __init__(self, root_a, root_b, experiment_config: DictConfig):

        self.exp_config = experiment_config

        self.df_a = pd.read_csv(root_a, header=[0, 1], index_col=None)
        self.df_b = pd.read_csv(root_b, header=[0, 1], index_col=None)

        if self.exp_config.shuffle:
            self.df_a = self.df_a.sample(frac=1).reset_index(drop=True)
            self.df_b = self.df_b.sample(frac=1).reset_index(drop=True)

        self.refl_a = torch.tensor(self.df_a.reflectances.values)
        self.refl_b = torch.tensor(self.df_b.reflectances.values)

        self.oxy_a = torch.tensor(self.df_a.layer0.sao2.values)
        self.oxy_b = torch.tensor(self.df_b.layer0.sao2.values)

        self.bvf_a = torch.tensor(self.df_a.layer0.vhb.values)
        self.bvf_b = torch.tensor(self.df_b.layer0.vhb.values)

        if experiment_config.normalization not in ["None", "none"]:
            self.normalization = experiment_config.normalization
        else:
            self.normalization = False

    def __getitem__(self, index):
        refl_a = self.refl_a[index]
        refl_b = self.refl_b[index]


        # normalization
        if self.normalization:
            if self.normalization == "standardize":
                refl_a = (refl_a - self.exp_config.data["mean_a"]) / self.exp_config.data["std_a"]
                refl_b = (refl_b - self.exp_config.data["mean_b"]) / self.exp_config.data["std_b"]

        return {"spectra_a": refl_a.type(torch.float32), "spectra_b": refl_b.type(torch.float32),
                "bvf_a": self.bvf_a[index], "bvf_b": self.bvf_b[index],
                "oxy_a": self.oxy_a[index], "oxy_b": self.oxy_b[index],
                }

    def __len__(self):
        return min(self.df_a.shape[0], self.df_b.shape[0])
