import click
import json
from htc import DataPath, LabelMapping
import pandas as pd
from tqdm import tqdm
import numpy as np
import plotly.express as px

from src import settings
from src.data.make_knn_dataset import split_dataset


def get_organ_labels():
    with open('../data/semantic_organ_labels.json', 'rb') as handle:
        labels = json.load(handle)
    return labels


def get_dataset_iterator():
    iterator = DataPath.iterate(settings.tivita_semantic)
    return iterator


def plot_organ_statistics():
    dataset_iterator = get_dataset_iterator()
    labels = get_organ_labels()
    splits = split_dataset(iterator=dataset_iterator)
    results = {'organ': [], 'subject': [], 'split': [], 'spectra_count': [], 'name': []}
    for k, paths in splits.items():
        mapping = settings.mapping
        for p in tqdm(paths, desc=k):
            seg = p.read_segmentation()
            organ_id = [i for i in np.unique(seg) if mapping[str(i)] in labels["organ_labels"]]
            organs = [mapping[str(i)] for i in organ_id]
            spectra_count = [(seg == i).sum() for i in organ_id]
            results['organ'] += organs
            results['spectra_count'] += spectra_count
            results['subject'] += [p.subject_name for _ in organs]
            results['split'] += [k for _ in organs]
            results['name'] += [p.image_name() for _ in organs]
    results = pd.DataFrame(results)
    results_count = results.groupby(['organ', 'subject', 'split', 'name'], as_index=False).sum()
    results_count['spectra_trimmed'] = [min(i, labels['n_pixels']) for i in results_count.spectra_count]
    fig = px.bar(data_frame=results_count,
                 x='split',
                 y='spectra_trimmed',
                 color='organ',
                 barmode='group',
                 hover_data=['subject'],
                 category_orders=dict(split=[
                     "train", "train_synthetic",
                     "val", "val_synthetic",
                     "test", "test_synthetic"
                 ])
                 )
    fig.update_layout(font_size=16, font_family='Whitney Book', margin=dict(l=20, r=20, t=20, b=20))
    fig.write_html(settings.figures_dir / 'semantic_data_splits.html')
    fig.write_image(settings.figures_dir / 'semantic_data_splits.pdf')
    results_count.to_csv(settings.figures_dir / 'semantic_data_splits.csv')


@click.command()
@click.option('--describe', is_flag=True, help="plot the organ distribution for each pig on each datatest split")
def main(describe: bool):
    if describe:
        plot_organ_statistics()


if __name__ == '__main__':
    main()
