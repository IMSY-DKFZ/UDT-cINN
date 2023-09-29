import click
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src import settings
from src.data.multi_layer_loader import SimulationDataLoader


def load_data(dataset: str, splits: list):
    dl = SimulationDataLoader()
    df = dl.get_database(dataset, splits=splits)
    split_c = df.split.values
    df = df.loc[:, [c for c in df.columns.get_level_values(0).unique() if 'layer' in c]]
    df['split'] = split_c
    frames = []
    for c in df.columns.get_level_values(0).unique():
        if c == 'split':
            continue
        tmp = df.loc[:, c].copy()
        tmp.loc[:, 'split'] = df.split.values
        tmp = tmp.drop('a_ray', axis=1)
        tmp = pd.melt(tmp, value_name='value', var_name='property', id_vars='split')
        tmp['layer'] = c
        frames.append(tmp)
    results = pd.concat(frames, axis=0, ignore_index=True)
    return results


def plot_sims_stats(dataset: str, splits: list):
    df = load_data(dataset=dataset, splits=splits)
    fig = px.violin(data_frame=df, x='value', y='split', facet_row='layer', facet_col='property', color='property')
    fig.update_traces(scalegroup=False, alignmentgroup='', width=1)
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / f'sims_dist_{dataset}.html')
    fig.write_image(settings.figures_dir / f'sims_dist_{dataset}.pdf')


def plot_spectra(dataset: str, splits: list):
    dl = SimulationDataLoader()
    df = dl.get_database(dataset, splits=splits)
    df = df.sample(n=1000)
    df_ref = df.reflectances.copy()
    df_ref['sao2'] = df.layer0.sao2.values
    df_ref['vhb'] = df.layer0.vhb.values
    df_ref['id'] = np.arange(0, df_ref.shape[0])
    df_ref['split'] = df['split'].values
    df_plot = df_ref.melt(value_name='reflectance', var_name='wavelength', id_vars=['sao2', 'vhb', 'id', 'split'])
    df_plot['wavelength'] = (df_plot['wavelength'].values.astype(float) / 10**-9).astype(int)
    fig = px.scatter(data_frame=df_plot, x="wavelength", y="reflectance", hover_data=['sao2', 'vhb'], color='sao2',
                     facet_col='split')
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.update_traces(opacity=0.2)
    fig.write_html(settings.figures_dir / f'sims_dist_{dataset}_ref.html')
    fig.write_image(settings.figures_dir / f'sims_dist_{dataset}_ref.pdf')


def plot_pca(dataset: str, splits: list):
    dl = SimulationDataLoader()
    df = dl.get_database(dataset, splits=splits)
    r = df.reflectances.values
    r_norm = normalize(r, norm='l1', axis=1)
    pca = PCA(n_components=2)
    r_pcs = pca.fit_transform(r_norm)
    ev = pca.explained_variance_ratio_ * 100
    df_pc = pd.DataFrame({
        f'PC 1 [{int(ev[0])}%]': r_pcs[:, 0],
        f'PC 2 [{int(ev[1])}%]': r_pcs[:, 1]
    })
    df_pc['sao2'] = df.layer0.sao2.values
    df_pc['split'] = df['split'].values
    fig = px.scatter(data_frame=df_pc, x=f'PC 1 [{int(ev[0])}%]', y=f'PC 2 [{int(ev[1])}%]', color='sao2', facet_col='split')
    fig.update_layout(font_size=16, font_family='Whitney Book')
    fig.write_html(settings.figures_dir / f'sims_dist_{dataset}_pcs.html')
    fig.write_image(settings.figures_dir / f'sims_dist_{dataset}_pcs.pdf')


@click.command()
@click.option('--stats', is_flag=True, help="plot distribution of optical properties of dataset")
@click.option('--dataset', is_flag=True, default='generic_depth_adapted', help="id of dataset to plot")
@click.option('--pca', is_flag=True, help="plots PCA visualization of corresponding dataset and splits")
@click.option('--ref', is_flag=True, help="plots reflectance corresponding to dataset in splits")
@click.option('-s', '--splits', multiple=True, default=['train', 'test'])
def main(stats: bool, dataset: str, splits: list, pca: bool, ref: bool):
    if stats:
        plot_sims_stats(dataset=dataset, splits=splits)
    if pca:
        plot_pca(dataset=dataset, splits=splits)
    if ref:
        plot_spectra(dataset=dataset, splits=splits)


if __name__ == '__main__':
    main()
