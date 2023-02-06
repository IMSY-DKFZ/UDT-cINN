import click
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import plotly.express as px
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt

from src import settings


def load_data(splits: list, norm: bool = True):
    mapping = settings.mapping
    data = {s: [] for s in splits}
    for split in splits:
        folder = settings.intermediates_dir / 'semantic' / split
        files = [f for f in folder.glob('*.npy') if '_ind.npy' not in f.name]

        seg_folder = settings.intermediates_dir / 'semantic' / 'segmentation'
        for f in tqdm(files):
            seg_name = f.name
            if '_KNN_' in str(f) and '_KNN_0.npy' not in str(f):
                continue
            if '_KNN_' in str(f):
                seg_name = re.sub('_KNN_\d.npy', '.npy', str(f.name))
            seg = np.load(seg_folder / seg_name, allow_pickle=True)
            img = np.load(f, allow_pickle=True)
            subject_id = f.name.split('#')[0]
            image_agg_data = []
            for organ_id in np.unique(seg):
                if norm:
                    organ_agg = np.median(normalize(img[seg == organ_id], norm='l1', axis=1), axis=0)
                else:
                    organ_agg = np.median(img[seg == organ_id], axis=0)
                assert organ_agg.size == 100, "wrong number of channels in agg data"
                tmp = {'subject_id': [subject_id for _ in organ_agg],
                       'organ_id': [organ_id for _ in organ_agg],
                       'reflectance': organ_agg,
                       'wavelength': np.arange(500, 1000, 5)}
                image_agg_data.append(tmp)
            data[split] += image_agg_data
    results = {k: {} for k in data}
    for k, content in data.items():
        results[k]['subject_id'] = np.concatenate([image_dict['subject_id'] for image_dict in content])
        results[k]['organ_id'] = np.concatenate([image_dict['organ_id'] for image_dict in content])
        results[k]['reflectance'] = np.concatenate([image_dict['reflectance'] for image_dict in content])
        results[k]['wavelength'] = np.concatenate([image_dict['wavelength'] for image_dict in content])
        tmp = pd.DataFrame(results[k])
        tmp['organ'] = [mapping[str(i)] for i in tmp.organ_id]
        results[k] = tmp
    return results


def plot_semantic_spectra():
    data = load_data(splits=['train', 'train_synthetic_adapted', 'train_synthetic_sampled'])
    train_agg = data.get('train').copy().groupby(['organ', 'wavelength', 'subject_id'],
                                                 as_index=False).reflectance.median()
    train_sampled_agg = data.get('train_synthetic_sampled').copy().groupby(['organ', 'wavelength', 'subject_id'],
                                                                           as_index=False).reflectance.median()
    train_adapted_agg = data.get('train_synthetic_adapted').copy().groupby(['organ', 'wavelength', 'subject_id'],
                                                                           as_index=False).reflectance.median()
    train_agg['dataset'] = 'real'
    train_sampled_agg['dataset'] = 'simulated_sampled'
    train_adapted_agg['dataset'] = 'simulated_adapted'

    df = pd.concat([train_agg, train_sampled_agg, train_adapted_agg], ignore_index=True, axis=0)

    sns.set_context('talk')
    n_classes = len(df.organ.unique())
    g = sns.relplot(data=df,
                    x="wavelength",
                    y="reflectance",
                    hue="dataset",
                    col="organ",
                    errorbar="sd",
                    col_wrap=min(n_classes, 5),
                    kind='line')
    g.tight_layout()
    plt.savefig(settings.figures_dir / 'semantic_reflectance.pdf')
    plt.clf()
    df.to_csv(settings.figures_dir / 'semantic_reflectance.csv', index=False)


def plot_knn_difference():
    data = load_data(splits=['train', 'train_synthetic_adapted', 'train_synthetic_sampled'])
    train_agg = data.get('train').copy().groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    train_sampled_agg = data.get('train_synthetic_sampled').copy().groupby(['organ', 'wavelength'],
                                                                           as_index=True).reflectance.median()
    train_adapted_agg = data.get('train_synthetic_adapted').copy().groupby(['organ', 'wavelength'],
                                                                           as_index=True).reflectance.median()
    diff_sampled = ((train_agg - train_sampled_agg).abs() / train_agg).reset_index().dropna()
    diff_sampled.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_adapted = ((train_agg - train_adapted_agg).abs() / train_agg).reset_index().dropna()
    diff_adapted.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    n_classes = len(diff_sampled.organ.unique())
    fig = px.line(data_frame=diff_sampled,
                  x="wavelength",
                  y="difference [%]",
                  color="organ",
                  facet_col="organ",
                  facet_col_wrap=min(n_classes, 5)
                  )
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'knn_diff_sampled.html')
    fig.write_image(settings.figures_dir / 'knn_diff_sampled.pdf')

    fig = px.line(data_frame=diff_adapted,
                  x="wavelength",
                  y="difference [%]",
                  color="organ",
                  facet_col="organ",
                  facet_col_wrap=min(n_classes, 5)
                  )
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'knn_diff_adapted.html')
    fig.write_image(settings.figures_dir / 'knn_diff_adapted.pdf')

    diff_sampled.to_csv(settings.figures_dir / 'knn_diff_sampled.csv', index=False)
    diff_adapted.to_csv(settings.figures_dir / 'knn_diff_adapted.csv', index=False)


@click.command()
@click.option('--diff', is_flag=True, help="plot difference between real data and KNN simulations")
@click.option('--spectra', is_flag=True, help="plot spectra of the semantic dataset")
def main(diff: bool, spectra: bool):
    if diff:
        plot_knn_difference()
    if spectra:
        plot_semantic_spectra()


if __name__ == '__main__':
    main()
