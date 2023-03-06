import click
import json
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
from htc import DataPath
from omegaconf import DictConfig

from src import settings
from src.data.make_knn_dataset import split_dataset
from src.data.data_modules.semantic_module import SemanticDataModule, EnableTestData
from src.utils.susi import ExperimentResults


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
            organ_id = [i for i in np.unique(seg) if str(i) in mapping and mapping[str(i)] in labels["organ_labels"]]
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
    fig.write_image(settings.figures_dir / 'semantic_data_splits.png')
    results_count.to_csv(settings.figures_dir / 'semantic_data_splits.csv')


def plot_organ_unique_statistics():
    config = DictConfig(dict(shuffle=True, num_workers=2, batch_size=100, normalization="standardize",
                             data=dict(mean_a=None, mean_b=None, std=None, std_b=None),
                             noise_aug=False, noise_aug_level=None))
    dm = SemanticDataModule(experiment_config=config, target_dataset='semantic_v2', target='unique')
    dm.setup(stage='train')
    stages = {
        'train': dm.train_dataloader(),
        'val': dm.val_dataloader()
    }
    with EnableTestData(dm):
        stages['test'] = dm.test_dataloader()
    results = ExperimentResults()
    mapping = settings.mapping
    for stage, dl in stages.items():
        y = dl.dataset.seg_data_a.cpu().numpy()
        for label in np.unique(y):
            results.append(name="stage", value=stage)
            results.append(name="organ", value=mapping[str(label)])
            results.append(name="value", value=len(y[y == label]))
    df = results.get_df()

    fig = px.bar(data_frame=df,
                 x="organ",
                 y="value",
                 color="organ",
                 template="plotly_white",
                 category_orders=dict(stage=["train", "val", "test"]),
                 facet_col="stage")
    fig.update_layout(font_size=16, font_family='Liberatinus Serif', margin=dict(l=20, r=20, t=20, b=20))
    fig.write_html(settings.figures_dir / 'semantic_unique_data_splits.html')
    fig.write_image(settings.figures_dir / 'semantic_unique_data_splits.pdf')
    fig.write_image(settings.figures_dir / 'semantic_unique_data_splits.png')


@click.command()
@click.option('--describe', is_flag=True, help="plot the organ distribution for each pig on each datatest split")
@click.option('--describe_unique', is_flag=True, help="plot the organ distribution for each pig on each datatest split")
def main(describe: bool, describe_unique: bool):
    if describe:
        plot_organ_statistics()
    if describe_unique:
        plot_organ_unique_statistics()


if __name__ == '__main__':
    main()
