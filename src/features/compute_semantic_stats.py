import click
import numpy as np
from tqdm import tqdm
import json

from src import settings


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


def compute_stats(data: list):
    x = np.concatenate(data).flatten()
    return dict(mean=float(x.mean()), std=float(x.std()), n=int(x.size), variance=float(x.std()**2))


def load_data():
    splits = ['train',
              'val',
              'test',
              'train_synthetic_adapted',
              'train_synthetic_sampled',
              'val_synthetic_adapted',
              'val_synthetic_sampled',
              'test_synthetic_adapted',
              'test_synthetic_sampled'
              ]
    results = {}
    for split in splits:
        folder = settings.intermediates_dir / 'semantic' / split
        files = folder.glob('*.npy')
        files = [f for f in files if '_ind' not in f.name]
        mmaps = []
        for f in tqdm(files, desc=f"loading mmaps {split}"):
            data = np.load(f, allow_pickle=True, mmap_mode='r')
            mmaps.append(data)
        stats = compute_stats(mmaps)
        results[split] = stats
    return results


def build_stats():
    stats = load_data()
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
