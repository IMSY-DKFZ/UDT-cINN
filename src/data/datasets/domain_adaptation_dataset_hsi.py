import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np


class DomainAdaptationDatasetHSI(Dataset):
    def __init__(self, root_a, root_b):

        self.df_a = pd.read_csv(root_a, header=[0, 1], index_col=0)
        self.df_b = pd.read_csv(root_b, header=[0, 1], index_col=0)

        self.refl_a = torch.tensor(self.df_a.reflectances.values)
        self.refl_b = torch.tensor(self.df_b.reflectances.values)

        # self.oxy_a = torch.tensor(self.df_a.layer0.sao2.values)
        # self.oxy_b = torch.tensor(self.df_b.layer0.sao2.values)

        self.bvf_a = torch.tensor(self.df_a.layer0.vhb.values)
        self.bvf_b = torch.tensor(self.df_b.layer0.vhb.values)

    def __getitem__(self, index):

        return {"spectra_a": self.refl_a[index].type(torch.float32), "spectra_b": self.refl_b[index].type(torch.float32),
                "bvf_a": self.bvf_a[index], "bvf_b": self.bvf_b[index],
                # "oxy_a": self.oxy_a[index], "oxy_b": self.oxy_b[index],
                }

    def __len__(self):
        return min(self.df_a.shape[0], self.df_b.shape[0])
