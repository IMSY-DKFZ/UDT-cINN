import click
import pandas as pd
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import torch

from src.data.data_modules import SemanticDataModule
from src.utils.susi import ExperimentResults
from src.data.utils import get_label_mapping
from src import settings

torch.manual_seed(100)
np.random.seed(100)


def calculate_redundancy():
    config = DictConfig({'data': {'mean_a': 0.1, 'std_a': 0.1, 'mean_b': 0.1, 'std_b': 0.1,
                                  'dataset_version': 'semantic_v2', 'choose_spectra': 'unique'},
                         'normalization': 'standardize', 'shuffle': False, 'num_workers': 5,
                         'batch_size': 1000, 'noise_aug': False, 'noise_aug_level': None},
                        )
    dm = SemanticDataModule(experiment_config=config)
    dm.setup(stage='train')
    x = dm.train_dataloader().dataset.data_a.cuda().type(torch.float32)
    y = dm.train_dataloader().dataset.seg_data_a.cuda().type(torch.float32)
    del dm

    unique_labels = torch.unique(y).cpu().numpy()
    results = {k: {k: 0 for k in unique_labels} for k in unique_labels}
    results_unique = {k: 0 for k in unique_labels}
    target = x[0]
    true_label = int(y[0])
    pbar = tqdm(desc="searching KNN duplicates", total=len(x))
    while x.numel():
        diff = x - target
        index = torch.where(torch.all(diff == 0, dim=1))
        index_inv = torch.where(~torch.all(diff == 0, dim=1))
        del diff
        assigned_labels: torch.Tensor = y[index]
        unique_assigned_labels = assigned_labels.unique()
        if len(unique_assigned_labels) == 1:
            results_unique[true_label] += 1
        for k in unique_labels:
            results[true_label][k] += int((assigned_labels == k).sum())

        # remove arrays that have been found
        x = x[index_inv]
        y = y[index_inv]
        if x.numel():
            target = x[0]
            true_label = int(y[0])
            pbar.update(1)
            pbar.total = len(x)
            pbar.refresh()
    results_agg = ExperimentResults()
    mapping = get_label_mapping()
    results_unique = {mapping[str(k)]: i for k, i in results_unique.items()}
    results_unique_df = pd.DataFrame(results_unique, index=[0])
    for source in results:
        for target in results[source]:
            results_agg.append(name='source', value=mapping[str(source)])
            results_agg.append(name='target', value=mapping[str(target)])
            results_agg.append(name='amount', value=results[source][target])
    df = results_agg.get_df()
    df.to_csv(settings.results_dir / 'knn' / 'redundancy.csv', index=False)
    results_unique_df.to_csv(settings.results_dir / 'knn' / 'uniqueness_per_organ.csv', index=False)


def count_unique_rows(x: torch.Tensor, label: str) -> (int, list):
    target = x[0]
    n_unique = 0
    n_unique_per_sample = []
    pbar = tqdm(desc=str(label), total=len(x))
    while x.numel():
        diff = x - target
        n_unique_per_sample.append(int(torch.all(diff == 0, dim=1).sum()))
        index_inv = torch.where(~torch.all(diff == 0, dim=1))
        x = x[index_inv]
        n_unique += 1
        if x.numel():
            target = x[0]
            pbar.update(1)
            pbar.total = len(x)
            pbar.refresh()
    return n_unique, n_unique_per_sample


def calculate_unique():
    config = DictConfig({'data': {'mean_a': 0.1, 'std_a': 0.1, 'mean_b': 0.1, 'std_b': 0.1,
                                  'dataset_version': 'semantic_v2', 'choose_spectra': 'unique'},
                         'normalization': 'standardize', 'shuffle': False, 'num_workers': 5,
                         'batch_size': 1000, 'noise_aug': False, 'noise_aug_level': None},
                        )
    dm = SemanticDataModule(experiment_config=config)
    dm.setup(stage='train')
    x = dm.train_dataloader().dataset.data_a.cuda().type(torch.float32)
    y = dm.train_dataloader().dataset.seg_data_a.cuda()
    del dm

    mapping = get_label_mapping()
    unique_labels = torch.unique(y).cpu().numpy()
    results = {k: 0 for k in unique_labels}
    results_per_sample = {k: [] for k in unique_labels}
    for label in unique_labels:
        tmp = x[y == label]
        if tmp.numel():
            n_unique, n_unique_per_sample = count_unique_rows(x=tmp, label=mapping[str(label)])
            results[label] = n_unique
            results_per_sample[label] = n_unique_per_sample

    sample_repetitions = ExperimentResults()
    for k, v in results_per_sample.items():
        sample_repetitions.append(name='organ', value=[mapping[str(k)] for _ in v])
        sample_repetitions.append(name='repetitions', value=v)
    sample_df = sample_repetitions.get_df()
    results = {mapping[str(k)]: i for k, i in results.items()}
    df = pd.DataFrame(results, index=[0])
    for k in sample_df.organ.unique():
        assert df.loc[0, k] == len(sample_df[sample_df.organ == k].repetitions)
    df.to_csv(settings.results_dir / 'knn' / 'uniqueness.csv', index=False)
    sample_df.to_csv(settings.results_dir / 'knn' / 'repetitions_per_sample.csv', index=False)


@click.command()
@click.option('--redundancy', is_flag=True, help="calculate how often each unique spectra is assigned to different "
                                                 "organs by KNN")
@click.option('--unique', is_flag=True, help="calculate the number of unique spectra that exist in one organ, and not"
                                             "in any other organ")
def main(redundancy: bool, unique: bool):
    if redundancy:
        calculate_redundancy()
    if unique:
        calculate_unique()


if __name__ == '__main__':
    main()
