import click
import pandas as pd
import plotly.express as px

from src.data.multi_layer_loader import SimulationDataLoader
from src import settings


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
        tmp = df.loc[:, c]
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


@click.command()
@click.option('--stats', type=bool, help="plot distribution of optical properties of dataset")
@click.option('--dataset', type=str, default='generic_depth_adapted', help="id of dataset to plot")
@click.option('-s', '--splits', multiple=True, default=['train', 'test'])
def main(stats: bool, dataset: str, splits: list):
    if stats:
        plot_sims_stats(dataset=dataset, splits=splits)


if __name__ == '__main__':
    main()
