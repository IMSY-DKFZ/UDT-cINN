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
            subject_id, image_id = f.name.split('#')
            image_id = '.'.join(image_id.split('.')[:-1])
            image_agg_data = []
            for organ_id in np.unique(seg):
                if norm:
                    organ_agg = np.median(normalize(img[seg == organ_id], norm='l1', axis=1), axis=0)
                else:
                    organ_agg = np.median(img[seg == organ_id], axis=0)
                assert organ_agg.size == 100, "wrong number of channels in agg data"
                tmp = {'subject_id': [subject_id for _ in organ_agg],
                       'image_id': [image_id for _ in organ_agg],
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
        results[k]['image_id'] = np.concatenate([image_dict['image_id'] for image_dict in content])
        tmp = pd.DataFrame(results[k])
        tmp['organ'] = [mapping[str(i)] for i in tmp.organ_id]
        results[k] = tmp
    return results


def load_inn_results() -> pd.DataFrame:
    folder = settings.results_dir / 'dummy_data'
    files = list(folder.glob('*.npz'))
    data = []
    seg = []
    subject_ids = []
    image_ids = []
    for file in files:
        tmp_data = np.load(file, allow_pickle=True)
        x = tmp_data['spectra_ab']
        # spectra adapted from synthetic to real should be normalized with the statistics of the real data set
        x = normalize(x, axis=1, norm='l1')
        y = tmp_data['seg_a']
        image_ids.append(tmp_data.get('image_ids_a'))
        subject_ids.append(tmp_data.get('subjects_a'))
        data.append(x)
        seg.append(y)
    data = np.concatenate(data, axis=0)
    seg = np.concatenate(seg)
    image_ids = np.concatenate(image_ids)
    subject_ids = np.concatenate(subject_ids)
    df = pd.DataFrame(data)
    df.columns = np.arange(500, 1000, 5)
    df['dataset'] = 'inn_adapted'
    df['organ'] = [settings.mapping[str(int(i))] for i in seg]
    df['image_id'] = image_ids
    df['subject_id'] = subject_ids
    df = df.melt(id_vars=['organ', 'dataset', 'image_id', 'subject_id'], value_name="reflectance", var_name="wavelength")
    return df


def agg_data(df: pd.DataFrame):
    data = df.copy().groupby(['organ', 'wavelength', 'subject_id', 'image_id'], as_index=False).reflectance.median()
    data = data.groupby(['organ', 'wavelength', 'subject_id'], as_index=False).reflectance.median()
    data = filter_data(data)
    return data


def filter_data(df: pd.DataFrame):
    data = df[df.organ.isin(settings.organ_labels) & (df.organ != 'gallbladder')]
    return data


def plot_semantic_spectra():
    inn_results = load_inn_results()
    inn_agg = agg_data(inn_results)
    data = load_data(splits=['train', 'train_synthetic_sampled'])
    real_agg = agg_data(data.get('train'))
    simulated_agg = agg_data(data.get('train_synthetic_sampled'))
    real_agg['dataset'] = 'real'
    simulated_agg['dataset'] = 'simulated'
    inn_agg['dataset'] = 'inn'

    df = pd.concat([real_agg, simulated_agg, inn_agg], ignore_index=True, sort=True, axis=0)

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
    # load and aggregate data
    inn_results = load_inn_results()
    inn_agg = agg_data(inn_results).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    data = load_data(splits=['train', 'train_synthetic_sampled'])
    real_agg = agg_data(data.get('train')).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    simulated_agg = agg_data(data.get('train_synthetic_sampled')).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    assert inn_agg.shape == real_agg.shape == simulated_agg.shape
    # compute differences
    diff_simulated = ((real_agg - simulated_agg).abs() / real_agg).reset_index().dropna()
    diff_simulated.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_simulated['source'] = 'real - simulated'
    diff_inn = ((real_agg - inn_agg).abs() / real_agg).reset_index().dropna()
    diff_inn.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_inn['source'] = "real - inn"
    n_classes = len(diff_simulated.organ.unique())
    df = pd.concat([diff_inn, diff_simulated], sort=True, ignore_index=True, axis=0)
    # plot data
    fig = px.line(data_frame=df,
                  x="wavelength",
                  y="difference [%]",
                  color="source",
                  facet_col="organ",
                  facet_col_wrap=min(n_classes, 5)
                  )
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'knn_diff.html')
    fig.write_image(settings.figures_dir / 'knn_diff.pdf')


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
