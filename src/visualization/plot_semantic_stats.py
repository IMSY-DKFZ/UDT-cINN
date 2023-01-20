import click
import json
from htc import DataPath, LabelMapping
from htc.tivita.DataPathMultiorgan import DataPathMultiorgan
import pandas as pd
import seaborn as sns
from typing import *
from tqdm import tqdm
import numpy as np
import plotly.express as px

from src import settings


def get_organ_labels():
    with open('../data/semantic_organ_labels.json', 'rb') as handle:
        labels = json.load(handle)
    return labels


def get_dataset_iterator():
    iterator = DataPath.iterate(settings.tivita_semantic)
    return iterator


def split_dataset(iterator):
    paths: List[DataPathMultiorgan] = list(iterator)
    with open("../data/semantic_data_splits.json", "rb") as handle:
        splits = json.load(handle)
    for p in splits['train']:
        assert p not in splits['test'], "found ID of subject in both train and test sets"
    paths_splits = {"train": [p for p in paths if p.subject_name in splits["train"]],
                    "test": [p for p in paths if p.subject_name in splits["test"]],
                    "train_synthetic": [p for p in paths if p.subject_name in splits["train_synthetic"]]}
    return paths_splits


def plot_organ_statistics():
    dataset_iterator = get_dataset_iterator()
    labels = get_organ_labels()
    splits = split_dataset(iterator=dataset_iterator)
    results = {'organ': [], 'subject': [], 'split': []}
    for k, paths in splits.items():
        mapping = LabelMapping.from_path(paths[0])
        for p in tqdm(paths):
            seg = p.read_segmentation()
            organs = [mapping.index_to_name(i) for i in np.unique(seg)]
            organs = [o for o in organs if o in labels['organ_labels']]
            results['organ'] += organs
            results['subject'] += [p.subject_name for _ in organs]
            results['split'] += [k for _ in organs]
    results = pd.DataFrame(results)
    results_count = results.value_counts().reset_index().rename({0: 'count'}, axis=1)
    sns.set_context('talk')
    fig = px.bar(data_frame=results_count, x='split', y='count', color='organ', barmode='group', hover_data=['subject'])
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'semantic_data_splits.html')


@click.command()
@click.option('--describe', type=bool, help="plot the organ distribution for each pig on each datatest split")
def main(describe: bool):
    if describe:
        plot_organ_statistics()


if __name__ == '__main__':
    main()
