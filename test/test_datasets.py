import unittest
import torch
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize

from src import settings
from src.data.datasets.semantic_dataset import SemanticDataset
from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData, BalancedBatchSampler

this_path = Path(__file__)


def find_unique_rows(x: torch.Tensor, desc: str = "") -> torch.Tensor:
    target = x[0]
    unique_rows = []
    pbar = tqdm(desc=desc, total=len(x))
    while x.numel():
        diff = x - target
        index_inv = torch.where(~torch.all(diff == 0, dim=1))
        del diff
        x = x[index_inv]
        torch.cuda.empty_cache()
        unique_rows.append(target.cpu())
        if x.numel():
            target = x[0]
            pbar.update(1)
            pbar.total = len(x)
            pbar.refresh()
    unique = torch.vstack(unique_rows)
    return unique


class TestSamplers(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 9
        self.n_samples = 1000
        self.classes = torch.randint(0, 7, (self.n_samples,)).cpu().numpy()
        self.class_size = self.batch_size // len(np.unique(self.classes))
        self.data_source = np.random.rand(self.n_samples)
        self.batch_sampler = BalancedBatchSampler(data_source=self.data_source,
                                                  batch_size=self.batch_size,
                                                  drop_last=True,
                                                  classes=self.classes,
                                                  )

    def test_batch_sampler(self):
        prevalence_tolerance = self.batch_size % len(np.unique(self.classes))
        for idx in self.batch_sampler:
            self.assertTrue(len(idx[0]) == self.batch_size)
            labels = self.classes[idx[0]]
            for label in np.unique(labels):
                np.testing.assert_allclose(len(labels[labels == label]), self.class_size, atol=prevalence_tolerance)


class TestSemanticUnique(unittest.TestCase):
    def setUp(self) -> None:
        self.base_folder = settings.intermediates_dir / 'semantic_unique'
        self.config = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, normalization="standardize",
                                      data=dict(mean_a=None, mean_b=None, std=None, std_b=None, balance_classes=True,
                                                dataset_version='semantic_v2', choose_spectra='unique'),
                                      noise_aug=False, noise_aug_level=None))
        self.dm = SemanticDataModule(experiment_config=self.config)
        self.dm.setup(stage='train')

    def test_train_dl(self):
        dl = self.dm.train_dataloader()
        x = dl.dataset.data_a
        y = dl.dataset.seg_data_a
        for i in np.unique(y):
            print(i, y[y == i].size())

        x_unique_true = find_unique_rows(x.cuda())
        assert x_unique_true.shape == x.shape

    def test_val_dl(self):
        dl = self.dm.val_dataloader()
        x = dl.dataset.data_a
        y = dl.dataset.seg_data_a
        for i in np.unique(y):
            print(i, y[y == i].size())

        x_unique_true = find_unique_rows(x.cuda())
        assert x_unique_true.shape == x.shape

    def test_test_dl(self):
        with EnableTestData(self.dm):
            dl = self.dm.test_dataloader()
        x = dl.dataset.data_a
        y = dl.dataset.seg_data_a
        for i in np.unique(y):
            print(i, y[y == i].size())

        x_unique_true = find_unique_rows(x.cuda())
        assert x_unique_true.shape == x.shape


class TestSemanticUniqueRealSource(unittest.TestCase):
    def setUp(self) -> None:
        self.base_folder = settings.intermediates_dir / 'semantic_unique'
        self.config = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, normalization="standardize",
                                      data=dict(mean_a=None, mean_b=None, std=None, std_b=None, balance_classes=True,
                                                dataset_version='semantic_v2', choose_spectra='unique'),
                                      noise_aug=False, noise_aug_level=None))
        self.dm = SemanticDataModule(experiment_config=self.config)
        self.dm.setup(stage='train')

        self.config_source = self.config.copy()
        self.config_source.data.choose_spectra = 'real_source_unique'
        self.dm_source = SemanticDataModule(experiment_config=self.config_source)
        self.dm_source.setup(stage='train')

    def test_labels(self):
        loaders = [self.dm.train_dataloader(), self.dm.val_dataloader()]
        loaders_source = [self.dm_source.train_dataloader(), self.dm_source.val_dataloader()]
        for loader, loader_source in zip(loaders, loaders_source):
            for label in torch.unique(loader.dataset.seg_data_a):
                n_label = len(loader.dataset.seg_data_a[loader.dataset.seg_data_a == label])
                n_label_source = len(loader_source.dataset.seg_data_a[loader_source.dataset.seg_data_a == label])
                assert n_label == n_label_source, f"{n_label}!={n_label_source} for {label}"


class TestDataSplits(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_semantic_dataset_splits(self):
        train_folder = settings.intermediates_dir / 'semantic_v2' / 'train'
        val_folder = settings.intermediates_dir / 'semantic_v2' / 'val'
        test_folder = settings.intermediates_dir / 'semantic_v2' / 'test'
        train_synthetic_folder = settings.intermediates_dir / 'semantic_v2' / 'train_synthetic_sampled'
        val_synthetic_folder = settings.intermediates_dir / 'semantic_v2' / 'val_synthetic_sampled'
        test_synthetic_folder = settings.intermediates_dir / 'semantic_v2' / 'test_synthetic_sampled'

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
                                                     train_synthetic_subjects, val_synthetic_subjects,
                                                     test_synthetic_subjects])))


class TestSemanticDataset(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_loading(self):
        config = DictConfig({'data': {'mean_a': 0.1, 'std_a': 0.1, 'mean_b': 0.1, 'std_b': 0.1, 'balance_classes': True},
                             'normalization': 'standardize'})
        ds = SemanticDataset(root_a=settings.intermediates_dir / 'semantic_v2' / 'train',
                             root_b=settings.intermediates_dir / 'semantic_v2' / 'train_synthetic_sampled',
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
        self.assertTrue(isinstance(data.get('subjects_a'), str))
        self.assertTrue(isinstance(data.get('subjects_b'), str))
        self.assertTrue(isinstance(data.get('image_ids_a'), str))
        self.assertTrue(isinstance(data.get('image_ids_b'), str))

    def test_segmentations(self):
        folder = settings.intermediates_dir / 'semantic_v2'
        labels = settings.organ_labels
        mapping = settings.mapping
        organs = []
        splits = [f for f in list(folder.glob('*')) if f.is_dir()]
        for split in splits:
            files = list(split.glob('*_seg.npy'))
            for f in tqdm(files, desc=split.name):
                x = np.load(f, allow_pickle=True)
                ind = np.unique(x)
                organs += [mapping[str(i)] for i in ind]
            self.assertTrue(set(organs).issubset(labels), f"organs in labels != organs in segmentations "
                                                          f"{list(set(organs))} is not subset of {list(set(labels))}")


class TestSemanticDataModule(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 10000
        self.balance_classes = False
        conf = dict(batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=1,
                    normalization='standardize',
                    data=dict(mean_a=None, mean_b=None, std=None, std_b=None, balance_classes=self.balance_classes,
                              dataset_version='semantic_v2', choose_spectra='unique'),
                    noise_aug=True,
                    noise_aug_level=0.1
                    )
        conf = DictConfig(conf)
        self.dm = SemanticDataModule(experiment_config=conf)
        self.dm.setup(stage='setup')

    def test_normalization(self):
        # test normalization of 1D vector
        x = torch.rand(100)
        true_normalization = normalize(x.numpy()[np.newaxis, ...], axis=1, norm='l2').squeeze()
        dm_norm = self.dm.train_dataloader().dataset.normalize(x)
        np.testing.assert_allclose(true_normalization, dm_norm.numpy(), atol=1e-6)

        # test normalization of 2D array
        x = torch.rand(10, 100)
        true_normalization = normalize(x.numpy(), axis=1, norm='l2').squeeze()
        dm_norm = self.dm.train_dataloader().dataset.normalize(x)
        np.testing.assert_allclose(true_normalization, dm_norm.numpy(), atol=1e-6)

    def test_train_dl(self):
        train_dl = self.dm.train_dataloader()
        data = next(iter(train_dl))
        self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_a').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
        self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_b').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
        self.assertTrue(isinstance(data.get('mapping'), dict))
        self.assertTrue(isinstance(data.get('subjects_a'), np.ndarray))
        self.assertTrue(isinstance(data.get('subjects_b'), np.ndarray))
        self.assertTrue(isinstance(data.get('image_ids_a'), np.ndarray))
        self.assertTrue(isinstance(data.get('image_ids_b'), np.ndarray))
        self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                        == len(data.get('seg_b')) == len(data.get('subjects_a')) == len(data.get('subjects_b'))
                        == len(data.get('image_ids_a')) == len(data.get('image_ids_b')))
        if self.balance_classes:
            labels = data.get('seg_a')
            class_size = self.batch_size // len(torch.unique(labels))
            prevalence_tolerance = self.batch_size % len(torch.unique(labels))
            for label in torch.unique(labels):
                np.testing.assert_allclose(len(labels[labels == label]), class_size, atol=prevalence_tolerance)

    def test_val_dl(self):
        dl = self.dm.val_dataloader()
        data = next(iter(dl))
        self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_a').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
        self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
        self.assertTrue(len(data.get('spectra_b').size()) == 2)
        self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
        self.assertTrue(isinstance(data.get('mapping'), dict))
        self.assertTrue(isinstance(data.get('subjects_a'), np.ndarray))
        self.assertTrue(isinstance(data.get('subjects_b'), np.ndarray))
        self.assertTrue(isinstance(data.get('image_ids_a'), np.ndarray))
        self.assertTrue(isinstance(data.get('image_ids_b'), np.ndarray))
        self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                        == len(data.get('seg_b')))

    @unittest.skipIf(False, "loading all data is slow, this test should be run manually")
    def test_dl_loading(self):
        loaders = [self.dm.train_dataloader(), self.dm.val_dataloader()]
        for loader in loaders:
            batch_size = loader.batch_sampler.batch_size
            ignore_classes = loader.dataset.ignore_classes
            pbar = tqdm(total=len(loader))
            ignore_indices = [int(i) for i, k in settings.mapping.items() if k in ignore_classes]
            loader_iter = iter(loader)
            while True:
                try:
                    data = next(loader_iter)
                    self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_a').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                    self.assertTrue(np.all([i not in data.get('seg_a') for i in ignore_indices]))
                    self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_b').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                    self.assertTrue(np.any([i not in data.get('seg_b') for i in ignore_indices]))
                    self.assertTrue(isinstance(data.get('mapping'), dict))
                    self.assertTrue(np.all([i in np.arange(len(data.get('order'))) for i in data.get('order').values()]))
                    if batch_size == 1:
                        self.assertTrue(isinstance(data.get('subjects_a'), str))
                        self.assertTrue(isinstance(data.get('subjects_b'), str))
                        self.assertTrue(isinstance(data.get('image_ids_a'), str))
                        self.assertTrue(isinstance(data.get('image_ids_b'), str))
                        self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                                        == len(data.get('seg_b')))
                    else:
                        self.assertTrue(isinstance(data.get('subjects_a'), np.ndarray))
                        self.assertTrue(isinstance(data.get('subjects_b'), np.ndarray))
                        self.assertTrue(isinstance(data.get('image_ids_a'), np.ndarray))
                        self.assertTrue(isinstance(data.get('image_ids_b'), np.ndarray))
                        self.assertTrue(
                            len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                            == len(data.get('seg_b')) == len(data.get('subjects_a')) == len(data.get('subjects_b'))
                            == len(data.get('image_ids_a')) == len(data.get('image_ids_b')))
                        labels = data.get('seg_a')
                        class_size = self.batch_size // len(torch.unique(labels))
                        prevalence_tolerance = self.batch_size % len(torch.unique(labels))
                        if self.balance_classes:
                            for label in torch.unique(labels):
                                np.testing.assert_allclose(len(labels[labels == label]), class_size,
                                                           atol=prevalence_tolerance)
                        else:
                            for label in torch.unique(labels):
                                self.assertFalse(
                                    np.allclose(len(labels[labels == label]), class_size, atol=prevalence_tolerance)
                                )
                    pbar.update(1)
                except (StopIteration, KeyboardInterrupt):
                    break

    @unittest.skipIf(False, "loading all data is slow, this test should be run manually")
    def test_dl_loading_synthetic(self):
        self.batch_size = 10000
        balance_classes = True
        conf = dict(batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=1,
                    normalization='standardize',
                    data=dict(mean=None, std=None, balance_classes=balance_classes,
                              dataset_version='semantic_v2', choose_spectra='unique'),
                    noise_aug=False,
                    noise_aug_level=0.1
                    )
        conf = DictConfig(conf)
        dm = SemanticDataModule(experiment_config=conf)
        dm.setup(stage='train')
        loaders = [dm.train_dataloader(), dm.val_dataloader()]
        for loader in loaders:
            batch_size = loader.batch_sampler.batch_size
            for data in tqdm(loader):
                self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_a').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                self.assertTrue(len(data.get('spectra_b').size()) == 2)
                self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                self.assertTrue(isinstance(data.get('mapping'), dict))
                self.assertTrue(np.all([i in np.arange(len(data.get('order'))) for i in data.get('order').values()]))
                if batch_size == 1:
                    self.assertTrue(isinstance(data.get('subjects_a'), str))
                    self.assertTrue(isinstance(data.get('subjects_b'), str))
                    self.assertTrue(isinstance(data.get('image_ids_a'), str))
                    self.assertTrue(isinstance(data.get('image_ids_b'), str))
                    self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                                    == len(data.get('seg_b')))
                else:
                    self.assertTrue(isinstance(data.get('subjects_a'), np.ndarray))
                    self.assertTrue(isinstance(data.get('subjects_b'), np.ndarray))
                    self.assertTrue(isinstance(data.get('image_ids_a'), np.ndarray))
                    self.assertTrue(isinstance(data.get('image_ids_b'), np.ndarray))
                    self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                                    == len(data.get('seg_b')) == len(data.get('subjects_a')) == len(data.get('subjects_b'))
                                    == len(data.get('image_ids_a')) == len(data.get('image_ids_b')))
                    labels = data.get('seg_a')
                    class_size = self.batch_size // len(torch.unique(labels))
                    prevalence_tolerance = self.batch_size % len(torch.unique(labels))
                    if balance_classes:
                        for label in torch.unique(labels):
                            np.testing.assert_allclose(len(labels[labels == label]), class_size,
                                                       atol=prevalence_tolerance)
                    else:
                        for label in torch.unique(labels):
                            self.assertFalse(
                                np.allclose(len(labels[labels == label]), class_size, atol=prevalence_tolerance)
                            )

    def test_dl_test_context_manager(self):
        with EnableTestData(self.dm):
            loaders = [self.dm.test_dataloader()]
            for loader in loaders:
                for data in tqdm(loader):
                    self.assertTrue(isinstance(data.get('spectra_a'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_a').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_a'), torch.Tensor))
                    self.assertTrue(isinstance(data.get('spectra_b'), torch.Tensor))
                    self.assertTrue(len(data.get('spectra_b').size()) == 2)
                    self.assertTrue(isinstance(data.get('seg_b'), torch.Tensor))
                    self.assertTrue(isinstance(data.get('mapping'), dict))
                    self.assertTrue(
                        np.all([i in np.arange(len(data.get('order'))) for i in data.get('order').values()]))
                    self.assertTrue(isinstance(data.get('subjects_a'), np.ndarray))
                    self.assertTrue(isinstance(data.get('subjects_b'), np.ndarray))
                    self.assertTrue(isinstance(data.get('image_ids_a'), np.ndarray))
                    self.assertTrue(isinstance(data.get('image_ids_b'), np.ndarray))
                    self.assertTrue(len(data.get('spectra_a')) == len(data.get('spectra_b')) == len(data.get('seg_a'))
                                    == len(data.get('seg_b')) == len(data.get('subjects_a')) == len(
                        data.get('subjects_b'))
                                    == len(data.get('image_ids_a')) == len(data.get('image_ids_b')))
        self.assertTrue(self.dm.test_dataloader is None)


if __name__ == '__main__':
    unittest.main()
