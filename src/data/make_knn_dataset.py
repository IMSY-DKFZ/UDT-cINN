import click
import numpy as np
import cuml
from htc import DataPath, LabelMapping
from pathlib import Path
from htc.tivita.DataPathMultiorgan import DataPathMultiorgan
from typing import *
import json
from tqdm import tqdm

from src import settings
from src.data.multi_layer_loader import SimulationDataLoader

here = Path(__file__).parent


def fit_knn(x: np.ndarray, **kwargs):
    nn = cuml.NearestNeighbors(**kwargs)
    nn.fit(x)
    return nn


def get_dataset_iterator():
    iterator = DataPath.iterate(settings.tivita_semantic)
    return iterator


def split_dataset(iterator):
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
    results = {}
    for k, item in models.items():
        model = item['model']
        x = item['data']
        idx = model.kneighbors(im, return_distance=False)
        results[k] = {i: x[idx[:, i]] for i in range(idx.shape[1])}
    return results


def get_organ_labels():
    with open(str(here / 'semantic_organ_labels.json'), 'rb') as handle:
        labels = json.load(handle)
    return labels


def generate_dataset(nr_neighbours: int):
    dataset_iterator = get_dataset_iterator()
    labels = get_organ_labels()
    nr_pixels = labels['nr_pixels']
    splits = split_dataset(iterator=dataset_iterator)
    # get fitted KNN models
    models = get_knn_models(nr_neighbours=nr_neighbours)
    # iterate over data splits
    seg_folder = settings.intermediates_dir / 'semantic' / 'segmentation'
    seg_folder.mkdir(exist_ok=True)
    for k, paths in splits.items():
        for p in tqdm(paths):
            mapping = LabelMapping.from_path(p)
            with open(str(settings.intermediates_dir / 'semantic' / 'mapping.json'), 'w') as handle:
                json.dump(mapping.mapping_index_name, handle)
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
                continue
            mask = mask.astype(bool)
            seg = seg[mask]
            assert seg.size <= nr_pixels * len(organ_indexes)  # organs can have less than nr_pixels
            np.save(file=str(seg_folder / f'{p.image_name()}.npy'), arr=seg)
            if k in ["train", "test", "val"]:
                results_folder = settings.intermediates_dir / 'semantic' / k
                results_folder.mkdir(exist_ok=True)
                data = im[mask]
                assert data.shape[-1] == 100
                ind = np.array(np.where(mask))
                np.save(file=results_folder / f"{p.image_name()}.npy", arr=data)
                np.save(file=results_folder / f"{p.image_name()}_ind.npy", arr=ind)
            if k in ["train_synthetic", "val_synthetic", "test_synthetic"]:
                data = im[mask]
                ind = np.array(np.where(mask))
                neighbours = get_nearest_neighbors(im=data, models=models)
                for name, nn_images in neighbours.items():
                    results_folder = settings.intermediates_dir / 'semantic' / f'{k}_{name}'
                    results_folder.mkdir(exist_ok=True)
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
