import click
import numpy as np
from tqdm import tqdm
import json
from omegaconf import DictConfig

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
    x = data.flatten()
    return dict(mean=float(x.mean()), std=float(x.std()), n=int(x.size), variance=float(x.std() ** 2))


def get_stats():
    real_cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, target="real", normalization="standardize"))
    synthetic_cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, target="synthetic", normalization="standardize"))
    synthetic_adapted_cfg = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, target="synthetic_adapted", normalization="standardize"))
    real_dl = SemanticDataModule(experiment_config=real_cfg)
    real_dl.setup(stage='train')
    synthetic_dl = SemanticDataModule(experiment_config=synthetic_cfg)
    synthetic_dl.setup(stage='train')
    synthetic_adapted_dl = SemanticDataModule(experiment_config=synthetic_adapted_cfg)
    synthetic_adapted_dl.setup(stage='train')
    splits = {
        'train': real_dl.train_dataloader().dataset.data,
        'val': real_dl.val_dataloader().dataset.data,
        'train_synthetic_sampled': synthetic_dl.train_dataloader().dataset.data,
        'val_synthetic_sampled': synthetic_dl.val_dataloader().dataset.data,
        'train_synthetic_adapted': synthetic_adapted_dl.train_dataloader().dataset.data,
        'val_synthetic_adapted': synthetic_adapted_dl.val_dataloader().dataset.data,
    }
    with EnableTestData(real_dl):
        splits['test'] = real_dl.test_dataloader().dataset.data
    with EnableTestData(synthetic_dl):
        splits['test_synthetic_sampled'] = synthetic_dl.test_dataloader().dataset.data
    with EnableTestData(synthetic_adapted_dl):
        splits['test_synthetic_adapted'] = synthetic_adapted_dl.test_dataloader().dataset.data
    results = {}
    for split, data in splits.items():
        stats = _compute_stats(data.numpy())
        results[split] = stats
    return results


def build_stats():
    stats = get_stats()
    results_file = settings.intermediates_dir / 'semantic' / 'data_stats.json'
    with open(str(results_file), 'w') as handle:
        json.dump(stats, handle)


@click.command()
@click.option('--stats', is_flag=True, help="compute stats used for normalization during model training")
def main(stats):
    if stats:
        build_stats()
    pass


if __name__ == '__main__':
    main()
