import numpy as np
import pytorch_lightning as pl
import json
import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, SequentialSampler
from omegaconf import DictConfig
from typing import Sized, List, Iterator, Iterable, Sequence
from itertools import cycle

from src.data.datasets.semantic_dataset import SemanticDataset
from src import settings
from src.utils.collate_function import collate_hsi


class SemanticDataModule(pl.LightningDataModule):
    def __init__(self, experiment_config: DictConfig, target='sampled', target_dataset='semantic_v2'):
        """
        PyTorchLightning data loader. Each data loader feed the data as a dictionary containing the reflectances, the
        labels and a dictionary with mappings from labels (int) -> organ names (str).
        The shapes of the reflectances are `nr_samples * nr_channels` while the labels have a shape of
        `nr_samples`. Each data split was generated by splitting all pigs in the dataset such that each data split
        contains a unique set o f pigs and all organs are represented in each data split.
        By default, the test data set is hidden to avoind data leackage during training. During testing, it can be
        enabled through the context manager `EnableTestData`, and example of this is given below.

        >>> cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, normalization="standardize"))
        >>> dm = SemanticDataModule(cfg)
        >>> dm.setup()
        >>> dl = dm.train_dataloader()
        >>> with EnableTestData(dm):
        >>>     test_dl = dm.test_dataloader()

        :param experiment_config: configuration containing loader parameters such as batch size, number of workers, etc.
            The minimum parameters expected are `shuffle, num_workers, batch_size, target`. The target should be `real`
            or `synthetic`. The target `real` represents the raw reflectances from different organs of pigs while the
            target `synthetic` represents simulated data that was generated by assigning the nearest neighbor to each
            pixel from real images.
        :param target: suffix of data set of domain_a to be loaded. The data belonging to domain_b is always the same.
            The folder for each data split is defined as (e.g. for training set):
            `settings.intermediates_dir / self.target_dataset / f'train_synthetic_{self.target}'`
        :param target_dataset: folder name inside `settings.intermediates_dir` from which the dataset is loaded
        """
        super(SemanticDataModule, self).__init__()
        self.exp_config = experiment_config
        self.shuffle = experiment_config.shuffle
        self.num_workers = experiment_config.num_workers
        self.batch_size = experiment_config.batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.target = target
        self.target_dataset = target_dataset
        self.dimensions = 100
        self.data_stats = self.load_data_stats()
        self.ignore_classes = ['gallbladder']
        self.organs = [o for o in settings.organ_labels if o not in self.ignore_classes]
        self.mapping_inv = {v: int(k) for k, v in settings.mapping.items() if k in self.organs}
        self.organ_ids = list(self.mapping_inv.keys())
        self.adjust_experiment_config()
        self.root_path = settings.intermediates_dir / self.target_dataset
        self.batch_samplers = {}

    def load_data_stats(self):
        with open(str(settings.intermediates_dir / self.target_dataset / 'data_stats.json'), 'rb') as handle:
            content = json.load(handle)
        return content

    def setup(self, stage: str) -> None:
        self.train_dataset = SemanticDataset(
            settings.intermediates_dir / self.target_dataset / f'train_synthetic_{self.target}',
            settings.intermediates_dir / self.target_dataset / f'train',
            exp_config=self.exp_config,
            ignore_classes=self.ignore_classes,
            noise_aug=self.exp_config.noise_aug,
            noise_std=self.exp_config.noise_aug_level,
        )
        self.batch_samplers['train'] = BalancedBatchSampler(data_source=self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            drop_last=True,
                                                            classes=self.train_dataset.seg_data_a.cpu().numpy(),
                                                            )
        self.val_dataset = SemanticDataset(
            settings.intermediates_dir / self.target_dataset / f'val_synthetic_{self.target}',
            settings.intermediates_dir / self.target_dataset / f'val',
            exp_config=self.exp_config,
            ignore_classes=self.ignore_classes)

        self.batch_samplers['val'] = BalancedBatchSampler(data_source=self.val_dataset,
                                                          batch_size=1,
                                                          drop_last=True,
                                                          classes=self.val_dataset.seg_data_a.cpu().numpy(),
                                                          )

    def train_dataloader(self) -> DataLoader:
        if self.exp_config.data.balance_classes:
            dl = DataLoader(self.train_dataset,
                            num_workers=self.exp_config.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            batch_sampler=self.batch_samplers['train']
                            )
        else:
            dl = DataLoader(self.train_dataset,
                            num_workers=self.exp_config.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            drop_last=True,
                            batch_size=self.batch_size,
                            )
        return dl

    def val_dataloader(self) -> DataLoader:
        if self.exp_config.data.balance_classes:
            dl = DataLoader(self.val_dataset,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            batch_sampler=self.batch_samplers['val']
                            )
        else:
            dl = DataLoader(self.val_dataset,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            drop_last=True,
                            batch_size=1,
                            )
        return dl

    def test_dataloader(self) -> DataLoader:
        if self.exp_config.data.balance_classes:
            dl = DataLoader(self.val_dataset,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            batch_sampler=self.batch_samplers['val']
                            )
        else:
            dl = DataLoader(self.val_dataset,
                            num_workers=self.num_workers,
                            pin_memory=True,
                            collate_fn=collate_hsi,
                            drop_last=True,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle
                            )
        return dl

    def adjust_experiment_config(self):
        self.exp_config.data.dimensions = self.dimensions
        self.exp_config.data.mean_a = self.data_stats.get(f'train_synthetic_{self.target}').get('mean')
        self.exp_config.data.mean_b = self.data_stats.get(f'train').get('mean')
        self.exp_config.data.std_a = self.data_stats.get(f'train_synthetic_{self.target}').get('std')
        self.exp_config.data.std_b = self.data_stats.get(f'train').get('std')
        self.exp_config.data.n_classes = len(self.organs)


class EnableTestData:
    def __init__(self, dm: SemanticDataModule):
        self.dm = dm

    def __enter__(self):
        self.dm.test_dataset = SemanticDataset(
            settings.intermediates_dir / self.dm.target_dataset / f'test_synthetic_{self.dm.target}',
            settings.intermediates_dir / self.dm.target_dataset / f'test',
            exp_config=self.dm.exp_config,
            ignore_classes=self.dm.ignore_classes,
            test_set=True)
        self.dm.batch_samplers['test'] = BalancedBatchSampler(data_source=self.dm.test_dataset,
                                                              batch_size=self.dm.batch_size,
                                                              drop_last=True,
                                                              classes=self.dm.test_dataset.seg_data_a.cpu().numpy(),
                                                              )

        def test_data_loader():
            if self.dm.exp_config.data.balance_classes:
                dl = DataLoader(self.dm.test_dataset,
                                num_workers=self.dm.num_workers,
                                pin_memory=True,
                                collate_fn=collate_hsi,
                                batch_sampler=self.dm.batch_samplers['test']
                                )
            else:
                dl = DataLoader(self.dm.test_dataset,
                                num_workers=self.dm.num_workers,
                                pin_memory=True,
                                collate_fn=collate_hsi,
                                shuffle=True,
                                drop_last=False,
                                batch_size=self.dm.batch_size,
                                )
            return dl

        self.dm.__setattr__('test_dataloader', test_data_loader)
        return self.dm

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dm.__setattr__('test_dataloader', None)
        self.dm.batch_samplers['test'] = None


class BalancedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 data_source: Sized,
                 batch_size: int,
                 drop_last: bool,
                 classes: Sequence[int]) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.classes = classes
        self.unique_classes = np.unique(classes)
        self.unique_classes.sort()

    def __iter__(self) -> Iterator[List[int]]:
        data_size = len(self.data_source)
        sample_ind = np.random.permutation(range(data_size))
        class_index = {k: np.random.permutation(np.where(self.classes == k)[0]) for k in self.unique_classes}
        for k, v in class_index.items():
            class_index[k] = cycle(torch.tensor(v, dtype=torch.int64))
        balanced_ind = []
        for k in cycle(self.unique_classes):
            balanced_ind.append(int(next(class_index[k])))
            if len(balanced_ind) == data_size:
                break
        sampler_iter = iter(sample_ind)
        balanced_iter = iter(balanced_ind)
        while True:
            try:
                batch = [next(sampler_iter) for _ in range(self.batch_size)]
                batch_balanced = [next(balanced_iter) for _ in range(self.batch_size)]
                yield batch_balanced, batch
            except StopIteration:
                break

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.data_source) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
