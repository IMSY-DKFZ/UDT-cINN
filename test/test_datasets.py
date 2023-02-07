import unittest
from pathlib import Path
from tqdm import tqdm
import torch
from omegaconf import DictConfig
import numpy as np
import json

from src import settings
from src.data.datasets.semantic_dataset import SemanticDataset
from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData

this_path = Path(__file__)


class TestAdaptToCamera(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_semantic_dataset_splits(self):
        train_folder = settings.intermediates_dir / 'semantic' / 'train'
        val_folder = settings.intermediates_dir / 'semantic' / 'val'
        test_folder = settings.intermediates_dir / 'semantic' / 'test'
        train_synthetic_folder = settings.intermediates_dir / 'semantic' / 'train_synthetic_adapted'
        val_synthetic_folder = settings.intermediates_dir / 'semantic' / 'val_synthetic_adapted'
        test_synthetic_folder = settings.intermediates_dir / 'semantic' / 'test_synthetic_adapted'

        train_files = list(train_folder.glob('*.npy'))
        val_files = list(val_folder.glob('*.npy'))
        test_files = list(test_folder.glob('*.npy'))
        train_synthetic_files = list(train_synthetic_folder.glob('*.npy'))
        val_synthetic_files = list(val_synthetic_folder.glob('*.npy'))
        test_synthetic_files = list(test_synthetic_folder.glob('*.npy'))

        train_subjects = set([str(f.name).split('#')[0] for f in train_files])
        val_subjects = set([str(f.name).split('#')[0] for f in val_files])
        test_subjects = set([str(f.name).split('#')[0] for f in test_files])
        train_synthetic_subjects = set([str(f.name).split('#')[0] for f in train_synthetic_files])
        val_synthetic_subjects = set([str(f.name).split('#')[0] for f in val_synthetic_files])
        test_synthetic_subjects = set([str(f.name).split('#')[0] for f in test_synthetic_files])

        self.assertFalse(set.intersection(*map(set, [train_subjects, val_subjects, test_subjects,
                                                     train_synthetic_subjects, val_synthetic_subjects, test_synthetic_subjects])))


class TestSemanticDataset(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_loading(self):
        config = DictConfig({'data': {'mean': 0.1, 'std': 0.1}, 'normalization': 'standardize'})
        ds = SemanticDataset(root_a=settings.intermediates_dir / 'semantic' / 'train',
                             root_b=settings.intermediates_dir / 'semantic' / 'train_synthetic_sampled',
                             exp_config=config,
                             noise_aug=True,
                             noise_std=0.3)
        data = ds[0]
        self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_a').size()) == 1)
        self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
        self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_b').size()) == 1)
        self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
        self.assertTrue(isinstance(data.get('mapping'), dict))

    def test_segmentations(self):
        folder = settings.intermediates_dir / 'semantic'
        mapping_file = folder / 'mapping.json'
        semantic_labels_file = this_path.parent.parent / 'src' / 'data' / 'semantic_organ_labels.json'
        with open(str(semantic_labels_file), 'rb') as handle:
            labels = json.load(handle)['organ_labels']
        organs = []
        with open(str(mapping_file), 'rb') as handle:
            mapping = json.load(handle)
        for f in tqdm(list((folder/'segmentation').glob('*.npy'))):
            x = np.load(f, allow_pickle=True)
            ind = np.unique(x)
            organs += [mapping[str(i)] for i in ind]
        self.assertTrue(set(labels) == set(organs), f"organs in labels != organs in segmentations "
                                                    f"{list(set(labels))}!={list(set(organs))}")


class TestSemanticDataModule(unittest.TestCase):
    def setUp(self) -> None:
        conf = dict(batch_size=100,
                    shuffle=False,
                    num_workers=1,
                    normalization='standardize',
                    data=dict(mean_a=None, mean_b=None, std=None, std_b=None),
                    target='real'
                    )
        conf = DictConfig(conf)
        self.dl = SemanticDataModule(experiment_config=conf)
        self.dl.setup(stage='setup')

    def test_train_dl(self):
        train_dl = self.dl.train_dataloader()
        data = next(iter(train_dl))
        self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_a').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
        self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_b').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
        self.assertTrue(isinstance(data.get('mapping'), dict))

    def test_val_dl(self):
        dl = self.dl.val_dataloader()
        data = next(iter(dl))
        self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_a').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
        self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_b').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
        self.assertTrue(isinstance(data.get('mapping'), dict))

    @unittest.skipIf(False, "loading all data is slow, this test should be run manually")
    def test_dl_loading(self):
        loaders = [self.dl.val_dataloader(), self.dl.train_dataloader()]
        for loader in loaders:
            ignore_classes = loader.dataset.ignore_classes
            ignore_indices = [int(i) for i, k in settings.mapping.items() if k in ignore_classes]
            for data in tqdm(loader):
                self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_a').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                self.assertFalse(np.any([i in data.get('seg_a') for i in ignore_indices]))
                self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_b').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                self.assertFalse(np.any([i in data.get('seg_b') for i in ignore_indices]))
                self.assertTrue(isinstance(data.get('mapping'), dict))

    @unittest.skipIf(False, "loading all data is slow, this test should be run manually")
    def test_dl_loading_synthetic(self):
        conf = dict(batch_size=100,
                    shuffle=False,
                    num_workers=1,
                    normalization='standardize',
                    data=dict(mean=0.1, std=0.1),
                    target='synthetic'
                    )
        conf = DictConfig(conf)
        dl = SemanticDataModule(experiment_config=conf)
        dl.setup(stage='train')
        loaders = [dl.val_dataloader(), dl.train_dataloader()]
        for loader in loaders:
            for data in tqdm(loader):
                self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_a').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_b').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                self.assertTrue(isinstance(data.get('mapping'), dict))

    def test_dl_test_context_manager(self):
        with EnableTestData(self.dl):
            loaders = [self.dl.test_dataloader()]
            for loader in loaders:
                for data in tqdm(loader):
                    self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_a').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                    self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_b').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                    self.assertTrue(isinstance(data.get('mapping'), dict))
        self.assertTrue(self.dl.test_dataloader is None)


if __name__ == '__main__':
    unittest.main()
