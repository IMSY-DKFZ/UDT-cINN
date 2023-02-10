import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import re
from torch.linalg import norm
from typing import List

from src import settings


class SemanticDataset(Dataset):
    def __init__(self,
                 root_a: Path,
                 root_b: Path,
                 exp_config: DictConfig,
                 noise_aug: bool = False,
                 noise_std: float = 0.3,
                 ignore_classes: list = None,
                 test_set: bool = False
                 ):
        super(SemanticDataset, self).__init__()
        self.root_a = root_a
        self.root_b = root_b
        if exp_config.normalization not in ["None", "none"]:
            self.normalization = exp_config.normalization
        else:
            self.normalization = False
        self.noise_aug = noise_aug
        self.noise_std = noise_std
        self.exp_config = exp_config
        self.image_list_a = [f for f in sorted(root_a.glob('*.npy')) if '_ind.npy' not in f.name]
        self.image_list_b = [f for f in sorted(root_b.glob('*.npy')) if '_ind.npy' not in f.name]
        self.segmentation_path = settings.intermediates_dir / 'semantic' / 'segmentation'
        seg_file_names_a = self._strip_names([f.name for f in self.image_list_a])
        seg_file_names_b = self._strip_names([f.name for f in self.image_list_b])
        self.seg_list_a = [self.segmentation_path / f for f in seg_file_names_a]
        self.seg_list_b = [self.segmentation_path / f for f in seg_file_names_b]
        self.data_a, self.seg_data_a, self.data_b, self.seg_data_b = self.load_data()
        self.mapping: dict = settings.mapping
        self.mapping_inv = {v: i for i, v in self.mapping.items()}
        self.ignore_classes = ignore_classes
        self.filter_dataset()
        self.data_a_size = self.data_a.shape[0]
        self.data_b_size = self.data_b.shape[0]
        self.organs = [o for o in settings.organ_labels if o not in self.ignore_classes]
        self.order = {int(self.mapping_inv[o]): i for i, o in enumerate(self.organs) if o not in self.ignore_classes}
        self.test_set = test_set

    def filter_dataset(self):
        if self.ignore_classes:
            assert np.all([o in self.mapping.values() for o in self.ignore_classes])
            ignored_indices = [int(i) for i, k in self.mapping.items() if k in self.ignore_classes]
            masks_a = [(self.seg_data_a != i).numpy() for i in ignored_indices]
            masks_b = [(self.seg_data_b != i).numpy() for i in ignored_indices]
            if masks_a:
                mask = np.any(masks_a, axis=0)
                assert mask.shape == self.seg_data_a.shape
                self.data_a = self.data_a[mask]
                self.seg_data_a = self.seg_data_a[mask]
            if masks_b:
                mask = np.any(masks_b, axis=0)
                assert mask.shape == self.seg_data_b.shape
                self.data_b = self.data_b[mask]
                self.seg_data_b = self.seg_data_b[mask]

    @staticmethod
    def _strip_names(files: List[str]):
        patterns = [re.findall('_KNN_\d', f) or '' for f in files]
        patterns = [p[0] if p else '' for p in patterns]
        files_clean = [f.replace(p, '') for f, p in zip(files, patterns)]
        return files_clean

    def load_data(self):
        arrays_a = [np.load(str(f), allow_pickle=True) for f in self.image_list_a]
        arrays_b = [np.load(str(f), allow_pickle=True) for f in self.image_list_b]
        seg_maps_a = [np.load(str(f), allow_pickle=True) for f in self.seg_list_a]
        seg_maps_b = [np.load(str(f), allow_pickle=True) for f in self.seg_list_b]
        data_a = torch.tensor(np.concatenate(arrays_a))
        data_b = torch.tensor(np.concatenate(arrays_b))
        seg_data_a = torch.tensor(np.concatenate(seg_maps_a))
        seg_data_b = torch.tensor(np.concatenate(seg_maps_b))
        ind = torch.randperm(data_a.shape[0])
        data_a = data_a[ind, :]
        seg_data_a = seg_data_a[ind]
        return data_a, seg_data_a, data_b, seg_data_b

    @staticmethod
    def normalize(x: torch.Tensor):
        return x / norm(x, ord=2)

    def __getitem__(self, index) -> dict:
        spectra_a = self.data_a[index % self.data_a_size, ...]
        spectra_b = self.data_b[index % self.data_b_size, ...]
        seg_a = self.seg_data_a[index % self.data_a_size]
        seg_b = self.seg_data_b[index % self.data_b_size]
        # normalization and noise augmentation
        if self.normalization and self.normalization == "standardize":
            spectra_a = (self.normalize(spectra_a) - self.exp_config.data["mean_a"]) / self.exp_config.data["std_a"]
            spectra_b = (self.normalize(spectra_b) - self.exp_config.data["mean_b"]) / self.exp_config.data["std_b"]
        if self.noise_aug:
            spectra_a += torch.normal(0.0, self.noise_std, size=spectra_a.shape)
            spectra_b += torch.normal(0.0, self.noise_std, size=spectra_b.shape)
        return {
            "spectra_a": spectra_a.type(torch.float32),
            "spectra_b": spectra_b.type(torch.float32),
            "seg_a": seg_a.type(torch.float32),
            "seg_b": seg_b.type(torch.float32),
            "mapping": self.mapping,
            "order": self.order}

    def __len__(self):
        if self.test_set:
            size = min(self.data_a_size, self.data_b_size)
        else:
            size = max(self.data_a_size, self.data_b_size)
        return size
