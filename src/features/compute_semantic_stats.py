import click
import numpy as np
import json
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src import settings
from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData


def compute_running_stats(data: list):
    """
    compute online mean and variance based on Welford's method
    https://www.johndcook.com/blog/standard_deviation/

    :param data:
    :return:
    """

    n = 0
    mean = 0
    m2 = 0
    for item in tqdm(data, desc="computing running stats"):
        item = item.flatten()
        for x in item:
            n += 1
            delta = x - mean
            mean = mean + delta / n
            m2 = m2 + delta * (x - mean)
    variance = m2 / (n - 1)
    std = np.sqrt(variance)
    return dict(mean=mean, variance=variance, n=n, std=std)


def _compute_stats(data: np.ndarray):
    x = normalize(data, norm='l2', axis=1)
    x = x.flatten()
    return dict(mean=float(x.mean()), std=float(x.std()), n=int(x.size), variance=float(x.std() ** 2))


def load_data(dm: SemanticDataModule, target: str = 'real_source'):
    splits = ['train', 'val', 'test']
    data = {}
    for split in tqdm(splits):
        folder = dm.root_path / f'{split}_synthetic_{target}'
        files = list(folder.glob('*.npy'))
        files = [f for f in files if '_ind.npy' not in f.name and '_seg.npy' not in f.name]
        x = [np.load(f) for f in files]
        x = np.vstack(x)
        x = torch.tensor(x)
        data[split] = x
    return data


def get_stats():
    real_cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, normalization="standardize",
                               data=dict(mean_a=None, mean_b=None, std=None, std_b=None),
                               noise_aug=False, noise_aug_level=None))
    adapted_cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, target="adapted", normalization="standardize",
                                  data=dict(mean_a=None, mean_b=None, std=None, std_b=None),
                                  noise_aug=False, noise_aug_level=None))
    dm = SemanticDataModule(experiment_config=real_cfg, target_dataset='semantic_v2')
    dm.setup(stage='train')
    adapted_dl = SemanticDataModule(experiment_config=adapted_cfg)
    adapted_dl.setup(stage='train')
    synthetic_real_source = load_data(dm=dm, target='real_source')
    synthetic_unique = load_data(dm=dm, target='unique')
    splits = {
        'train': dm.train_dataloader().dataset.data_b,
        'val': dm.val_dataloader().dataset.data_b,
        'train_synthetic_sampled': dm.train_dataloader().dataset.data_a,
        'val_synthetic_sampled': dm.val_dataloader().dataset.data_a,
        'train_synthetic_adapted': dm.train_dataloader().dataset.data_a,
        'val_synthetic_adapted': adapted_dl.val_dataloader().dataset.data_a,
        'train_synthetic_real_source': synthetic_real_source['train'],
        'val_synthetic_real_source': synthetic_real_source['val'],
        'test_synthetic_real_source': synthetic_real_source['test'],
        'train_synthetic_unique': synthetic_unique['train'],
        'val_synthetic_unique': synthetic_unique['val'],
        'test_synthetic_unique': synthetic_unique['test'],
    }
    with EnableTestData(dm):
        splits['test'] = dm.test_dataloader().dataset.data_b
        splits['test_synthetic_sampled'] = dm.test_dataloader().dataset.data_a
    with EnableTestData(adapted_dl):
        splits['test_synthetic_adapted'] = adapted_dl.test_dataloader().dataset.data_a
    results = {}
    for split, data in splits.items():
        stats = _compute_stats(data.numpy())
        results[split] = stats
    return results


def build_stats():
    stats = get_stats()
    results_file = settings.intermediates_dir / 'semantic_v2' / 'data_stats.json'
    with open(str(results_file), 'w') as handle:
        json.dump(stats, handle, indent=4)


@click.command()
@click.option('--stats', is_flag=True, help="compute stats used for normalization during model training")
def main(stats):
    if stats:
        build_stats()
    pass


if __name__ == '__main__':
    main()
