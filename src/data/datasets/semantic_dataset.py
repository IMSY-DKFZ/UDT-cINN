import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from torch.linalg import norm
from torch.utils.data import Dataset

from src import settings


class SemanticDataset(Dataset):
    """
    A dataset that is used to load the real and simulated dataset used for domain adaptation.
    """
    def __init__(self,
                 root_a: Path,
                 root_b: Path,
                 exp_config: DictConfig,
                 noise_aug: bool = False,
                 noise_std: float = 0.3,
                 ignore_classes: list = None,
                 test_set: bool = False
                 ):
        """

        :param root_a: path to data in the simulation domain. It should contain files ending with `_ind.npy` and
            `_seg.npy`
        :param root_b: path to data in the real domain
        :param exp_config: class containing the configuration necessary for the experiment: normalization,
            mean and standard deviation from both domains a and b.
        :param noise_aug: whether to do noise augmentation during loading
        :param noise_std: how much noise augmentation to do
        :param ignore_classes: list of classes to ignore during data loading
        :param test_set: whether to enable the test set or not
        """
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
        self.image_list_a = [f for f in sorted(root_a.glob('*.npy')) if '_ind.npy' not in f.name and '_seg.npy' not in f.name]
        self.image_list_b = [f for f in sorted(root_b.glob('*.npy')) if '_ind.npy' not in f.name and '_seg.npy' not in f.name]
        seg_file_names_a = self._strip_names([f"{str(f.name).split('.')[0]}_seg.npy" for f in self.image_list_a])
        seg_file_names_b = self._strip_names([f"{str(f.name).split('.')[0]}_seg.npy" for f in self.image_list_b])
        self.seg_list_a = [root_a / f for f in seg_file_names_a]
        self.seg_list_b = [root_b / f for f in seg_file_names_b]
        self.data_a, self.seg_data_a, self.data_b, self.seg_data_b, self.subjects_a, self.subjects_b, self.image_ids_a, self.image_ids_b = self.load_data()
        self.mapping: dict = settings.mapping
        self.mapping_inv = {v: i for i, v in self.mapping.items()}
        self.ignore_classes = ignore_classes
        self.filter_dataset()
        self.data_a_size = self.data_a.shape[0]
        self.data_b_size = self.data_b.shape[0]
        if self.ignore_classes:
            self.organs = [o for o in settings.organ_labels if o not in self.ignore_classes]
            self.order = {int(self.mapping_inv[o]): i for i, o in enumerate(self.organs) if
                          o not in self.ignore_classes}
        else:
            self.organs = settings.organ_labels
            self.order = {int(self.mapping_inv[o]): i for i, o in enumerate(self.organs)}
        self.test_set = test_set
        self.balance_batch_classes = True

    def filter_dataset(self):
        if self.ignore_classes:
            assert np.all([o in self.mapping.values() for o in self.ignore_classes])
            ignored_indices = [int(i) for i, k in self.mapping.items() if k in self.ignore_classes]
            masks_a = [(self.seg_data_a != i).numpy() for i in ignored_indices]
            masks_b = [(self.seg_data_b != i).numpy() for i in ignored_indices]
            if masks_a:
                mask = np.all(masks_a, axis=0)
                assert mask.shape == self.seg_data_a.shape
                self.data_a = self.data_a[mask]
                self.seg_data_a = self.seg_data_a[mask]
                self.subjects_a = self.subjects_a[mask]
                self.image_ids_a = self.image_ids_a[mask]
                assert np.all([i not in self.seg_data_a for i in ignored_indices])
                assert len(self.data_a) == len(self.seg_data_a) == len(self.subjects_a) == len(self.image_ids_a)
            if masks_b:
                mask = np.all(masks_b, axis=0)
                assert mask.shape == self.seg_data_b.shape
                self.data_b = self.data_b[mask]
                self.seg_data_b = self.seg_data_b[mask]
                self.subjects_b = self.subjects_b[mask]
                self.image_ids_b = self.image_ids_b[mask]
                assert np.all([i not in self.seg_data_b for i in ignored_indices])
                assert len(self.data_b) == len(self.seg_data_b) == len(self.subjects_b) == len(self.image_ids_b)

    @staticmethod
    def _strip_names(files: List[str]) -> List[str]:
        patterns = [re.findall('_KNN_\d', f) or '' for f in files]
        patterns = [p[0] if p else '' for p in patterns]
        files_clean = [f.replace(p, '') for f, p in zip(files, patterns)]
        return files_clean

    def get_image_ids(self, files, images):
        files = [str(f.name) for f in files]
        files = self._strip_names(files)
        image_ids = [[str(f).split('#')[-1].split('.')[0] for _ in range(a.shape[0])] for f, a in zip(files, images)]
        image_ids = np.concatenate(image_ids)
        return image_ids

    @staticmethod
    def get_subject_ids(files, images):
        subject_ids = [[str(f.name).split('#')[0] for _ in range(a.shape[0])] for f, a in zip(files, images)]
        subject_ids = np.concatenate(subject_ids)
        return subject_ids

    def load_data(self):
        arrays_a = [np.load(str(f), allow_pickle=True) for f in self.image_list_a]
        subjects_a = self.get_subject_ids(self.image_list_a, arrays_a)
        image_ids_a = self.get_image_ids(self.image_list_a, arrays_a)
        arrays_b = [np.load(str(f), allow_pickle=True) for f in self.image_list_b]
        subjects_b = self.get_subject_ids(self.image_list_b, arrays_b)
        image_ids_b = self.get_image_ids(self.image_list_b, arrays_b)
        seg_maps_a = [np.load(str(f), allow_pickle=True) for f in self.seg_list_a]
        seg_maps_b = [np.load(str(f), allow_pickle=True) for f in self.seg_list_b]
        data_a = torch.tensor(np.concatenate(arrays_a))
        data_b = torch.tensor(np.concatenate(arrays_b))
        seg_data_a = torch.tensor(np.concatenate(seg_maps_a))
        seg_data_b = torch.tensor(np.concatenate(seg_maps_b))
        ind = torch.randperm(data_a.shape[0])
        data_a = data_a[ind, :]
        seg_data_a = seg_data_a[ind]
        subjects_a = subjects_a[ind]
        image_ids_a = image_ids_a[ind]
        return data_a, seg_data_a, data_b, seg_data_b, subjects_a, subjects_b, image_ids_a, image_ids_b

    @staticmethod
    def normalize(x: torch.Tensor):
        if len(x.shape) == 2:
            return x / norm(x, ord=2, dim=1).unsqueeze(dim=1)
        elif len(x.shape) == 1:
            return x / norm(x, ord=2)

    def __getitem__(self, index) -> dict:
        """
        while iterating the dataset, it returns a dictionary with the keys:
        1. spectra_a: spectral from the simulation domain
        2. spectra_b: spectra from the real domain
        3. seg_a: segmentation of domain a data (organ labels)
        4. seg_b: segmentation of domain b data (organ labels)
        5. subjects_a: IDs of the subjects from which the data from spectra_a is extracted
        6. subjects_b: IDs of the subjects from which the data from spectra_b is extracted
        7. image_ids_a: IDs of the images from which the data from spectra_a is extracted
        8. image_ids_b: IDs of the images from which the data from spectra_b is extracted
        9. mapping: dictionary mapping from int values to organ labels
        10. order: order of organ labels that is always the same on each iteration. Can be used to sort the data that is
            returned

        :param index:
        :return:
        """
        if isinstance(index, list):
            index = torch.tensor(index)
        spectra_a = self.data_a[index % self.data_a_size, ...]
        spectra_b = self.data_b[index % self.data_b_size, ...]
        seg_a = self.seg_data_a[index % self.data_a_size]
        seg_b = self.seg_data_b[index % self.data_b_size]
        subjects_a = self.subjects_a[index % self.data_a_size]
        subjects_b = self.subjects_b[index % self.data_b_size]
        image_ids_a = self.image_ids_a[index % self.data_a_size]
        image_ids_b = self.image_ids_b[index % self.data_b_size]
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
            "subjects_a": subjects_a,
            "subjects_b": subjects_b,
            "image_ids_a": image_ids_a,
            "image_ids_b": image_ids_b,
            "mapping": self.mapping,
            "order": self.order}

    def __getitems__(self, indexes):
        if isinstance(indexes, tuple):
            data = self.__getitem__(indexes[0])
            data_b = self.__getitem__(indexes[1])
            keys_b = [k for k in data.keys() if k.endswith('_b')]
            data.update({k: data_b[k] for k in keys_b})
        else:
            data = self.__getitem__(indexes)
        return data

    def __len__(self):
        if self.test_set:
            # we always define the length based on the synthetic data during testing because we want to iterate over
            # the entire test set in order to train later on the classifier
            size = self.data_a_size
        else:
            # during training, we define the length as the maximum because we want to iterate over all the real and all
            # the synthetic data
            size = max(self.data_a_size, self.data_b_size)
        return size
