import click
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import plotly.express as px
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.decomposition import PCA

from src import settings
from src.visualization.plot import line
from src.visualization.templates import cmap_qualitative, cmap_qualitative_diff
from src.utils.susi import ExperimentResults


def load_data(splits: list, norm: bool = True):
    mapping = settings.mapping
    data = {s: [] for s in splits}
    for split in splits:
        folder = settings.intermediates_dir / 'semantic' / split
        files = [f for f in folder.glob('*.npy') if '_ind.npy' not in f.name]

        seg_folder = settings.intermediates_dir / 'semantic' / 'segmentation'
        i = 0
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
                       'sample_id': [i for _ in organ_agg],
                       'reflectance': organ_agg,
                       'wavelength': np.arange(500, 1000, 5)}
                image_agg_data.append(tmp)
                i += 1
            data[split] += image_agg_data
    results = {k: {} for k in data}
    for k, content in data.items():
        results[k]['subject_id'] = np.concatenate([image_dict['subject_id'] for image_dict in content])
        results[k]['organ_id'] = np.concatenate([image_dict['organ_id'] for image_dict in content])
        results[k]['reflectance'] = np.concatenate([image_dict['reflectance'] for image_dict in content])
        results[k]['wavelength'] = np.concatenate([image_dict['wavelength'] for image_dict in content])
        results[k]['image_id'] = np.concatenate([image_dict['image_id'] for image_dict in content])
        results[k]['sample_id'] = np.concatenate([image_dict['sample_id'] for image_dict in content])
        del content
        tmp = pd.DataFrame(results[k])
        tmp['organ'] = [mapping[str(i)] for i in tmp.organ_id]
        results[k] = tmp
    return results


def load_inn_results(folder: str) -> pd.DataFrame:
    folder = settings.results_dir / folder
    files = list(folder.glob('*.npz'))
    data = []
    seg = []
    subject_ids = []
    image_ids = []
    for file in tqdm(files):
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
    df['sample_id'] = np.arange(df.shape[0])
    df = df.melt(id_vars=[
        'organ',
        'dataset',
        'image_id',
        'subject_id',
        'sample_id'
    ],
        value_name="reflectance",
        var_name="wavelength")
    return df


def agg_data(df: pd.DataFrame):
    data = df.copy()
    data = data.groupby(['organ', 'wavelength', 'subject_id', 'image_id'], as_index=False).reflectance.median()
    if 'sample_id' in df.columns:
        data = data.groupby(['wavelength'], as_index=False, group_keys=False).apply(add_sample_id)
    data = filter_data(data)
    return data


def add_sample_id(df: pd.DataFrame):
    df['sample_id'] = np.arange(df.shape[0])
    return df


def filter_data(df: pd.DataFrame):
    data = df[df.organ.isin(settings.organ_labels) & (df.organ != 'gallbladder')]
    return data


def plot_semantic_spectra():
    inn_results = load_inn_results(folder='gan_cinn_hsi/2023_02_24_13_43_17/testing/generated_spectra_data')
    inn_agg = agg_data(inn_results)
    del inn_results
    unit_results = load_inn_results(folder='unit/generated_spectra_data')
    unit_agg = agg_data(unit_results)
    del unit_results

    data = load_data(splits=['val', 'val_synthetic_sampled'])
    real_agg = filter_data(data.get('val'))
    simulated_agg = filter_data(data.get('val_synthetic_sampled'))
    real_agg['dataset'] = 'real'
    simulated_agg['dataset'] = 'simulated'
    inn_agg['dataset'] = 'inn'
    unit_agg['dataset'] = 'unit'

    df = pd.concat([real_agg, simulated_agg, inn_agg, unit_agg], ignore_index=True, sort=True, axis=0)

    sns.set_context('talk')
    n_classes = len(df.organ.unique())
    # plot spectra
    fig, _ = line(data_frame=df,
                  x="wavelength",
                  y="reflectance",
                  color="dataset",
                  facet_col="organ",
                  facet_col_wrap=min(n_classes, 5),
                  color_discrete_map=cmap_qualitative
                  )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'semantic_reflectance.html')
    fig.write_image(settings.figures_dir / 'semantic_reflectance.pdf')
    fig.write_image(settings.figures_dir / 'semantic_reflectance.png')
    df.to_csv(settings.figures_dir / 'semantic_reflectance.csv', index=False)


def plot_knn_difference():
    # load and aggregate data
    inn_results = load_inn_results(folder='gan_cinn_hsi/2023_02_24_13_43_17/testing/generated_spectra_data')
    inn_agg = agg_data(inn_results).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    del inn_results
    data = load_data(splits=['val', 'val_synthetic_sampled'])
    real_agg = agg_data(data.get('val')).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    simulated_agg = agg_data(data.get('val_synthetic_sampled')).groupby(['organ', 'wavelength'],
                                                                        as_index=True).reflectance.median()
    unit_results = load_inn_results(folder='unit/generated_spectra_data')
    unit_agg = agg_data(unit_results).groupby(['organ', 'wavelength'], as_index=True).reflectance.median()
    del unit_results
    assert inn_agg.shape == real_agg.shape == simulated_agg.shape == unit_agg.shape
    # compute differences
    diff_simulated = ((real_agg - simulated_agg).abs() / real_agg).reset_index().dropna()
    diff_simulated.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_simulated['source'] = 'real - simulated'
    diff_inn = ((real_agg - inn_agg).abs() / real_agg).reset_index().dropna()
    diff_inn.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_inn['source'] = "real - inn"
    diff_unit = ((real_agg - unit_agg).abs() / real_agg).reset_index().dropna()
    diff_unit.rename({'reflectance': 'difference [%]'}, axis=1, inplace=True)
    diff_unit['source'] = "real - unit"

    n_classes = len(diff_simulated.organ.unique())
    df = pd.concat([diff_inn, diff_simulated, diff_unit], sort=True, ignore_index=True, axis=0)
    # plot data
    fig = px.line(data_frame=df,
                  x="wavelength",
                  y="difference [%]",
                  color="source",
                  facet_col="organ",
                  facet_col_wrap=min(n_classes, 5),
                  color_discrete_map=cmap_qualitative_diff
                  )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'knn_diff.html')
    fig.write_image(settings.figures_dir / 'knn_diff.pdf')
    fig.write_image(settings.figures_dir / 'knn_diff.png')


def plot_pca():
    inn_results = load_inn_results(folder='gan_cinn_hsi/2023_02_24_13_43_17/testing/generated_spectra_data')
    inn_agg = agg_data(inn_results)
    del inn_results
    unit_results = load_inn_results(folder='unit/generated_spectra_data')
    unit_agg = agg_data(unit_results)
    del unit_results
    data = load_data(splits=['val', 'val_synthetic_sampled'])
    real_agg = filter_data(data.get('val'))
    simulated_agg = filter_data(data.get('val_synthetic_sampled'))
    del data
    real_agg['dataset'] = 'real'
    simulated_agg['dataset'] = 'simulated'
    inn_agg['dataset'] = 'inn'
    unit_agg['dataset'] = 'unit'
    results = ExperimentResults()
    # train PCA on entire real data
    pca = PCA(n_components=2)
    r_df = real_agg.pivot(columns='wavelength', values='reflectance', index='sample_id')
    r_real = r_df.values
    r_norm = normalize(r_real, norm="l2", axis=1)
    pca.fit(r_norm)
    for organ in tqdm(real_agg.organ.unique()):
        # transform spectra
        tmp = real_agg[real_agg.organ == organ]
        compute_pcs(pca=pca, df=tmp, results=results, ids=(('dataset', 'real'), ('organ', organ)))

        tmp = simulated_agg[simulated_agg.organ == organ]
        compute_pcs(pca=pca, df=tmp, results=results, ids=(('dataset', 'simulated'), ('organ', organ)))

        tmp = unit_agg[unit_agg.organ == organ]
        compute_pcs(pca=pca, df=tmp, results=results, ids=(('dataset', 'unit'), ('organ', organ)))

        tmp = inn_agg[inn_agg.organ == organ]
        compute_pcs(pca=pca, df=tmp, results=results, ids=(('dataset', 'inn'), ('organ', organ)))

    pc_df = results.get_df()
    explained_variance = (round(pca.explained_variance_ratio_[0] * 100), round(pca.explained_variance_ratio_[1] * 100))
    pc_df = pc_df.rename({'pc_1': f"PC 1 [{explained_variance[0]} %]",
                          'pc_2': f"PC 2 [{explained_variance[1]} %]"},
                         axis=1)

    n_classes = len(real_agg.organ.unique())
    fig = px.scatter(data_frame=pc_df,
                     x=f"PC 1 [{explained_variance[0]} %]",
                     y=f"PC 2 [{explained_variance[1]} %]",
                     color="dataset",
                     facet_col="organ",
                     facet_col_wrap=min(n_classes, 5),
                     color_discrete_map=cmap_qualitative
                     )
    fig.update_traces(opacity=0.5)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(ncontours=10, zmin=0.05, zmax=0.95, line=dict(width=2), selector=trace_selector_2d_contour)
    fig.update_traces(histnorm='probability', nbinsx=100, selector=trace_selector_histogram)
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / 'semantic_pca.html')
    fig.write_image(settings.figures_dir / 'semantic_pca.pdf')
    fig.write_image(settings.figures_dir / 'semantic_pca.png')


def trace_selector_2d_contour(tr):
    return hasattr(tr, 'ncontours')


def trace_selector_histogram(tr):
    return hasattr(tr, 'cumulative')


def compute_pcs(pca: PCA, df: pd.DataFrame, results: ExperimentResults, ids: tuple):
    r_norm = normalize(df.pivot(columns='wavelength', values='reflectance', index='sample_id').values,
                       norm='l2',
                       axis=1)
    r_pc = pca.transform(r_norm)
    results.append('pc_1', r_pc[:, 0])
    results.append('pc_2', r_pc[:, 1])
    for k, name in ids:
        results.append(k, [name for _ in r_pc])


@click.command()
@click.option('--diff', is_flag=True, help="plot difference between real data and KNN simulations")
@click.option('--spectra', is_flag=True, help="plot spectra of the semantic dataset")
@click.option('--pca', is_flag=True, help="plot spectra principal components (PCA)")
def main(diff: bool, spectra: bool, pca: bool):
    if diff:
        plot_knn_difference()
    if spectra:
        plot_semantic_spectra()
    if pca:
        plot_pca()


if __name__ == '__main__':
    main()
