import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
import re

from src import settings


class SemanticDataset(Dataset):
    def __init__(self, root: Path, exp_config: DictConfig, noise_aug: bool = False, noise_std: float = 0.3):
        self.root = root
        if exp_config.normalization not in ["None", "none"]:
            self.normalization = exp_config.normalization
        else:
            self.normalization = False
        self.noise_aug = noise_aug
        self.noise_std = noise_std
        self.exp_config = exp_config
        self.image_list = [f for f in sorted(root.glob('*.npy')) if '_ind.npy' not in f.name]
        self.segmentation_path = settings.intermediates_dir / 'semantic' / 'segmentation'
        seg_file_names = self._strip_names([f.name for f in self.image_list])
        self.seg_list = [self.segmentation_path / f for f in seg_file_names]
        self.data, self.seg_data = self.load_data()
        self.mapping: dict = settings.mapping
        self.ignore_classes = ['gallbladder']
        self.filter_dataset()

    def filter_dataset(self):
        if self.ignore_classes:
            assert np.all([o in self.mapping.values() for o in self.ignore_classes])
            ignored_indices = [int(i) for i, k in self.mapping.items() if k in self.ignore_classes]
            masks = [(self.seg_data != i).numpy() for i in ignored_indices]
            if masks:
                mask = np.any(masks, axis=0)
                assert mask.shape == self.seg_data.shape
                self.data = self.data[mask]
                self.seg_data = self.seg_data[mask]

    @staticmethod
    def _strip_names(files: list[str]):
        patterns = [re.findall('_KNN_\d', f) or '' for f in files]
        patterns = [p[0] if p else '' for p in patterns]
        files_clean = [f.replace(p, '') for f, p in zip(files, patterns)]
        return files_clean

    def load_data(self):
        maps = [np.load(str(f), allow_pickle=True) for f in self.image_list]
        seg_maps = [np.load(str(f), allow_pickle=True) for f in self.seg_list]
        data = torch.tensor(np.concatenate(maps))
        seg_data = torch.tensor(np.concatenate(seg_maps))
        return data, seg_data

    def __getitem__(self, index) -> dict:
        img = self.data[index, ...]
        seg = self.seg_data[index]
        # normalization
        if self.normalization and self.normalization == "standardize":
            img = (img - self.exp_config.data["mean"]) / self.exp_config.data["std"]
        if self.noise_aug:
            img += torch.normal(0.0, self.noise_std, size=img.shape)
        return {"image": img.type(torch.float32),
                "seg": seg.type(torch.float32),
                "mapping": self.mapping}

    def __len__(self):
        return self.data.shape[0]
