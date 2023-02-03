from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.data.datasets.semantic_dataset import SemanticDataset
from src import settings


class SemanticDataModule(pl.LightningDataModule):
    def __init__(self, experiment_config: DictConfig):
        """
        PyTorchLightning data loader

        :param experiment_config: configuration containing loader parameters such as batch size, number of workers, etc.
        """
        super().__init__()
        self.exp_config = experiment_config
        self.shuffle = experiment_config.shuffle
        self.num_workers = experiment_config.num_workers
        self.batch_size = experiment_config.batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.target = experiment_config.target
        self._suffix = None

    def setup(self, stage: str) -> None:
        if self.target == 'synthetic':
            self.suffix = '_synthetic_sampled'
        elif self.target == 'real':
            self.suffix = ''
        else:
            raise ValueError(f"target {self.target} unknown")
        self.train_dataset = SemanticDataset(settings.intermediates_dir / 'semantic' / f'train{self.suffix}',
                                             exp_config=self.exp_config)
        self.val_dataset = SemanticDataset(settings.intermediates_dir / 'semantic' / f'val{self.suffix}',
                                           exp_config=self.exp_config)

    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.train_dataset,
                        batch_size=self.batch_size,
                        shuffle=self.exp_config.shuffle,
                        num_workers=self.exp_config.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        )
        return dl

    def val_dataloader(self) -> DataLoader:
        dl = DataLoader(self.val_dataset,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        )
        return dl

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError


class EnableTestData:
    def __init__(self, dl: SemanticDataModule):
        self.dl = dl

    def __enter__(self):
        self.dl.test_dataset = SemanticDataset(settings.intermediates_dir / 'semantic' / f'test{self.dl.suffix}',
                                               exp_config=self.dl.exp_config)

        def test_data_loader():
            dl = DataLoader(self.dl.test_dataset,
                            batch_size=self.dl.batch_size,
                            shuffle=self.dl.shuffle,
                            num_workers=self.dl.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            )
            return dl
        self.dl.__setattr__('test_dataloader', test_data_loader)
        return self.dl

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dl.__setattr__('test_dataloader', None)