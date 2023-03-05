import click
import numpy as np
import torch
import cuml
from tqdm import tqdm
from omegaconf import DictConfig

from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData

torch.manual_seed(100)
np.random.seed(100)


def fit_knn(x, **kwargs):
    nn = cuml.NearestNeighbors(**kwargs)
    nn.fit(x)
    return nn


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


def make_unique_knn_dataset():
    config = DictConfig({'data': {'mean_a': 0.1,
                                  'std_a': 0.1,
                                  'mean_b': 0.1,
                                  'std_b': 0.1,
                                  'dataset_version': 'semantic_v2',
                                  'choose_spectra': 'sampled',
                                  # simulated sampled data is used to search for the unique spectra
                                  'balance_classes': False},
                         'normalization': 'standardize',
                         'shuffle': False,
                         'num_workers': 5,
                         'batch_size': 1000,
                         'noise_aug': False,
                         'noise_aug_level': None},
                        )
    dm = SemanticDataModule(experiment_config=config)
    results_folder = config.data.dataset_version
    dm.setup(stage='train')
    stages = {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader()
    }
    with EnableTestData(dm):
        stages['test'] = dm.test_dataloader()
    del dm

    config.data.choose_spectra = 'real_source'  # unique spectra labeled based on data used to generate simulations
    dm_synthetic_real_source = SemanticDataModule(experiment_config=config)
    # setting target=real_source loads real data associated with synthetic data in *_a variables of the class
    dm_synthetic_real_source.setup('train')
    synthetic_real_source_stages = {
        'train': dm_synthetic_real_source.train_dataloader(),
        'val': dm_synthetic_real_source.val_dataloader(),
    }
    with EnableTestData(dm_synthetic_real_source):
        synthetic_real_source_stages['test'] = dm_synthetic_real_source.test_dataloader()
    del dm_synthetic_real_source

    for stage, dl in stages.items():
        x_a = dl.dataset.data_a.cuda(non_blocking=True).type(torch.float32)
        image_ids_unique_a = dl.dataset.image_ids_a
        subject_ids_unique_a = dl.dataset.subjects_a
        x_unique = find_unique_rows(x=x_a, desc=stage).cpu().numpy()
        x_synthetic_real_source = synthetic_real_source_stages[stage].dataset.data_a.cpu().numpy()
        y_synthetic_real_source = synthetic_real_source_stages[stage].dataset.seg_data_a.cpu().numpy()

        knn = fit_knn(x=x_synthetic_real_source, n_neighbors=1)
        idx = knn.kneighbors(x_unique, return_distance=False)
        y_unique = y_synthetic_real_source[idx[:, 0]]
        image_ids_unique_a = image_ids_unique_a[idx[:, 0]]
        subject_ids_unique_a = subject_ids_unique_a[idx[:, 0]]

        save_folder = results_folder / f'{stage}_synthetic_unique'
        save_folder_unique_source = results_folder / f'{stage}_synthetic_real_source_unique'
        save_folder_unique_source.mkdir(exist_ok=True, parents=True)
        save_folder.mkdir(exist_ok=True, parents=True)
        unique_pairs = list(set(zip(subject_ids_unique_a, image_ids_unique_a)))
        for subject_id, image_id in unique_pairs:
            ind = (subject_ids_unique_a == subject_id) & (image_ids_unique_a == image_id)
            x_tmp = x_unique[ind]
            y_tmp = y_unique[ind]
            file_name = f"{subject_id}#{image_id}.npy"
            seg_file_name = f"{subject_id}#{image_id}_seg.npy"
            np.save(file=save_folder / file_name, arr=x_tmp)
            np.save(file=save_folder / seg_file_name, arr=y_tmp)

            # save the real data used to label the unique samples
            x_tmp = x_synthetic_real_source[idx[:, 0][ind]]
            y_tmp = y_synthetic_real_source[idx[:, 0][ind]]
            np.save(file=save_folder_unique_source / file_name, arr=x_tmp)
            np.save(file=save_folder_unique_source / seg_file_name, arr=y_tmp)


@click.command()
@click.option('--unique', is_flag=True, help="generate a set of unique simulations with organ labels based on:\n"
                                             "1. take real images and assign for ech pixel the closes neighbour"
                                             "2. find all unique spectra that were assigned in step 1"
                                             "3. label the unique spectra generated in step 2 by finding the closest"
                                             "spectra from the real data and assigning the label that corresponds to "
                                             "that closest real pixel")
def main(unique):
    if unique:
        make_unique_knn_dataset()


if __name__ == '__main__':
    main()
