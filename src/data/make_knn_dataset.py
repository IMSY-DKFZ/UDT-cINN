import click
import numpy as np
import cuml
import pandas as pd
import torch
from htc import DataPath, LabelMapping
from pathlib import Path
from htc.tivita.DataPathMultiorgan import DataPathMultiorgan
from typing import *
import json
from tqdm import tqdm

from src import settings
from src.data.multi_layer_loader import SimulationDataLoader
from src.data.utils import get_organ_labels

here = Path(__file__).parent

np.random.seed(100)
torch.manual_seed(100)


def fit_knn(x: np.ndarray, **kwargs):
    nn = cuml.NearestNeighbors(**kwargs)
    nn.fit(x)
    return nn


def get_dataset_iterator():
    iterator = DataPath.iterate(settings.tivita_semantic)
    return iterator


def split_dataset(iterator):
    """
    Splits the dataset into train, validation, and test sets based on the provided iterator.

    :param iterator: An iterator containing `DataPathMultiorgan` objects.
    :return: A dictionary with keys 'train', 'val', 'test', 'train_synthetic', 'val_synthetic', 'test_synthetic', and
             values as lists of DataPathMultiorgan objects.
    """
    paths: List[DataPathMultiorgan] = list(iterator)
    with open(str(here / "semantic_data_splits.json"), "rb") as handle:
        splits = json.load(handle)
    for p in splits['train']:
        assert p not in splits['test'], "found ID of subject in both train and test sets"
    for p in splits['val']:
        assert p not in splits['test'], "found ID of subject in both val and test sets"
    paths_splits = {
        "train": [p for p in paths if p.subject_name in splits["train"]],
        "val": [p for p in paths if p.subject_name in splits["val"]],
        "test": [p for p in paths if p.subject_name in splits["test"]],
        "train_synthetic": [p for p in paths if p.subject_name in splits["train_synthetic"]],
        "val_synthetic": [p for p in paths if p.subject_name in splits["val_synthetic"]],
        "test_synthetic": [p for p in paths if p.subject_name in splits["test_synthetic"]],
    }
    return paths_splits


def load_adapted_layered_dataset() -> pd.DataFrame:
    """
    This method is used to load an adapted layered dataset for training.

    :return: A DataFrame object representing the loaded dataset.
    """
    loader = SimulationDataLoader()
    df = loader.get_database(simulation='generic_depth_adapted', splits=['train'])
    return df


def load_sampled_layered_dataset():
    """
    Loads a layered dataset from a simulation using a SimulationDataLoader.

    :return: A `pandas` DataFrame containing the loaded dataset.
    """
    loader = SimulationDataLoader()
    df = loader.get_database(simulation='generic_depth_sampled', splits=['train'])
    return df


def get_knn_models(nr_neighbours: int):
    """
    Trains knn models for transformed simulations based on `sampling wavelengths` and `adapting to the optical system`

    :param nr_neighbours: The number of nearest neighbors to consider when fitting the k-nearest neighbors model.
    :return: A dictionary containing the k-nearest neighbors models and their respective data. The keys 'sampled' and
        'adapted' correspond to the models fitted on the sampled and adapted datasets, respectively. Each model is
        represented as a dictionary with keys 'model' and 'data', where 'model' refers to the fitted k-nearest neighbors
        model and 'data' refers to the input data used for fitting the model.
    """
    df_sampled = load_sampled_layered_dataset()
    x_sampled = df_sampled.reflectances.values
    knn_sampled = fit_knn(x=x_sampled, **dict(n_neighbors=nr_neighbours))
    df_adapted = load_adapted_layered_dataset()
    x_adapted = df_adapted.reflectances.values
    knn_adapted = fit_knn(x=x_adapted, **dict(n_neighbors=nr_neighbours))
    return {'sampled': {'model': knn_sampled, 'data': x_sampled}, 'adapted': {'model': knn_adapted, 'data': x_adapted}}


def get_nearest_neighbors(im, models):
    results = {}
    for k, item in models.items():
        model = item['model']
        x = item['data']
        idx = model.kneighbors(im, return_distance=False)
        results[k] = {i: x[idx[:, i]] for i in range(idx.shape[1])}
    return results


def get_organ_data(p: DataPathMultiorgan,
                   nr_pixels: int,
                   labels: dict,
                   target_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    reads image from the provided path and 

    :param p: The DataPath object representing the data source.
    :param nr_pixels: The desired number of pixels for each organ.
    :param labels: The dictionary containing the labels of the organs.
    :param target_folder: The name of the target folder.
    :return: A tuple containing the image data, segmented data, and mask.
    """
    mapping = LabelMapping.from_path(p)
    with open(str(settings.intermediates_dir / target_folder / 'mapping.json'), 'w') as handle:
        json.dump(mapping.mapping_index_name, handle, indent=4)
    im = p.read_cube()
    seg = p.read_segmentation()
    organ_indexes = [i for i in np.unique(seg) if mapping.index_to_name(i) in labels['organ_labels']]
    if organ_indexes:
        mask = np.zeros_like(seg)
        for i in np.unique(seg):
            if i not in organ_indexes:
                continue
            organ_location = np.where(seg == i)
            nr_organ_pixels = organ_location[0].size
            if not organ_location:
                continue
            location_index = np.arange(organ_location[0].size)
            location_sample = np.random.choice(location_index, min(nr_organ_pixels, nr_pixels), replace=False)
            organ_location_sampled = (organ_location[0][location_sample], organ_location[1][location_sample])
            mask[organ_location_sampled] = 1
    else:
        return
    mask = mask.astype(bool)
    seg = seg[mask]
    assert seg.size <= nr_pixels * len(organ_indexes)  # organs can have less than nr_pixels
    return im, seg, mask


def generate_dataset(nr_neighbours: int, target_folder: str = 'semantic_v2'):
    dataset_iterator = get_dataset_iterator()
    labels = get_organ_labels()
    nr_pixels = labels['n_pixels']
    splits = split_dataset(iterator=dataset_iterator)
    # get fitted KNN models
    models = get_knn_models(nr_neighbours=nr_neighbours)
    # iterate over data splits
    for k, paths in splits.items():
        for p in tqdm(paths):
            organ_data = get_organ_data(p=p, nr_pixels=nr_pixels, labels=labels, target_folder=target_folder)
            if organ_data is None:
                continue
            else:
                im, seg, mask = organ_data
            if k in ["train", "test", "val"]:
                results_folder = settings.intermediates_dir / target_folder / k
                results_folder.mkdir(exist_ok=True)
                data = im[mask]
                assert data.shape[-1] == 100
                ind = np.array(np.where(mask))
                np.save(file=results_folder / f"{p.image_name()}.npy", arr=data)
                np.save(file=results_folder / f"{p.image_name()}_ind.npy", arr=ind)
                np.save(file=results_folder / f'{p.image_name()}_seg.npy', arr=seg)
            if k in ["train_synthetic", "val_synthetic", "test_synthetic"]:
                data = im[mask]
                ind = np.array(np.where(mask))
                results_folder = settings.intermediates_dir / target_folder / f'{k}_real_source'
                results_folder.mkdir(exist_ok=True, parents=True)
                np.save(file=results_folder / f"{p.image_name()}.npy", arr=data)
                np.save(file=results_folder / f"{p.image_name()}_ind.npy", arr=data)
                np.save(file=results_folder / f'{p.image_name()}_seg.npy', arr=seg)
                neighbours = get_nearest_neighbors(im=data, models=models)
                for name, nn_images in neighbours.items():
                    results_folder = settings.intermediates_dir / target_folder / f'{k}_{name}'
                    results_folder.mkdir(exist_ok=True)
                    np.save(file=results_folder / f'{p.image_name()}_seg.npy', arr=seg)
                    for i, nn_array in nn_images.items():
                        np.save(file=results_folder / f"{p.image_name()}_KNN_{i}.npy", arr=nn_array)
                        np.save(file=results_folder / f"{p.image_name()}_KNN_{i}_ind.npy", arr=ind)


@click.command()
@click.option('--knn', is_flag=True, help="generate dataset using KNN")
@click.option('--nr_neighbours', default=1, help="number of nearest neighbors")
def main(knn: bool, nr_neighbours: int):
    if knn:
        generate_dataset(nr_neighbours=nr_neighbours)


if __name__ == '__main__':
    main()
