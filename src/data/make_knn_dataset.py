import click
import numpy as np
import cuml
from htc import DataPath, LabelMapping
from htc.tivita.DataPathMultiorgan import DataPathMultiorgan
from typing import *
import json
from tqdm import tqdm

from src import settings
from src.data.multi_layer_loader import SimulationDataLoader


import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])


def fit_knn(x: np.ndarray, **kwargs):
    nn = cuml.NearestNeighbors(**kwargs)
    nn.fit(x)
    return nn


def get_dataset_iterator():
    iterator = DataPath.iterate(settings.tivita_semantic)
    return iterator


def split_dataset(iterator):
    paths: List[DataPathMultiorgan] = list(iterator)
    with open("./semantic_data_splits.json", "rb") as handle:
        splits = json.load(handle)
    for p in splits['train']:
        assert p not in splits['test'], "found ID of subject in both train and test sets"
    paths_splits = {"train": [p for p in paths if p.subject_name in splits["train"]],
                    "test": [p for p in paths if p.subject_name in splits["test"]],
                    "train_synthetic": [p for p in paths if p.subject_name in splits["train_synthetic"]]}
    return paths_splits


def load_adapted_layered_dataset():
    loader = SimulationDataLoader()
    df = loader.get_database(simulation='generic_depth_adapted', splits=['train'])
    return df


def load_sampled_layered_dataset():
    loader = SimulationDataLoader()
    df = loader.get_database(simulation='generic_depth_sampled', splits=['train'])
    return df


def get_knn_models(nr_neighbours: int):
    df_sampled = load_sampled_layered_dataset()
    x_sampled = df_sampled.reflectances.values
    knn_sampled = fit_knn(x=x_sampled, **dict(n_neighbors=nr_neighbours))
    df_adapted = load_adapted_layered_dataset()
    x_adapted = df_adapted.reflectances.values
    knn_adapted = fit_knn(x=x_adapted, **dict(n_neighbors=nr_neighbours))
    return {'sampled': {'model': knn_sampled, 'data': x_sampled}, 'adapted': {'model': knn_adapted, 'data': x_adapted}}


def get_nearest_neighbors(im, models):
    im_shape = im.shape
    im = im.reshape((im_shape[0]*im_shape[1], im_shape[2]))
    results = {}
    for k, item in models.items():
        model = item['model']
        x = item['data']
        idx = model.kneighbors(im, return_distance=False)
        results[k] = {i: x[idx[:, i]].reshape(im_shape) for i in range(idx.shape[1])}
    return results


def get_organ_labels():
    with open('./semantic_organ_labels.json', 'rb') as handle:
        labels = json.load(handle)
    return labels


def generate_dataset(nr_neighbours: int):
    dataset_iterator = get_dataset_iterator()
    labels = get_organ_labels()
    splits = split_dataset(iterator=dataset_iterator)
    # get fitted KNN models
    models = get_knn_models(nr_neighbours=nr_neighbours)
    # iterate over data splits
    for k, paths in splits.items():
        mapping = LabelMapping.from_path(paths[0])
        for p in tqdm(paths):
            im = p.read_cube()
            seg = p.read_segmentation()
            non_organ_indexes = [i for i in np.unique(seg) if mapping.index_to_name(i) not in labels['organ_labels']]
            if non_organ_indexes:
                mask = np.any(np.array([seg == i for i in non_organ_indexes]), axis=0)
            else:
                mask = np.zeros_like(seg).astype(bool)
            mask = np.repeat(mask[..., np.newaxis], im.shape[-1], axis=2)
            if k in ["train", "test"]:
                results_folder = settings.intermediates_dir / 'semantic' / k
                results_folder.mkdir(exist_ok=True)
                im_masked = np.ma.array(im, mask=mask)
                im_masked.dump(file=results_folder / f"{p.image_name()}.npy")
            if k == "train_synthetic":
                neighbours = get_nearest_neighbors(im=im, models=models)
                for name, nn_images in neighbours.items():
                    results_folder = settings.intermediates_dir / 'semantic' / f'{k}_{name}'
                    results_folder.mkdir(exist_ok=True)
                    for i, nn_array in nn_images.items():
                        nn_array_masked = np.ma.array(nn_array, mask=mask)
                        nn_array_masked.dump(results_folder / f"{p.image_name()}_KNN_{i}.npy")


@click.command()
@click.option('--knn', type=bool, help="generate dataset using KNN")
@click.option('--nr_neighbours', default=9, help="number of nearest neighbors")
def main(knn: bool, nr_neighbours: int):
    if knn:
        generate_dataset(nr_neighbours=nr_neighbours)


if __name__ == '__main__':
    main()
