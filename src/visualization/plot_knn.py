import click
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go

from src import settings


def plot_redundancy():
    df = pd.read_csv(settings.results_dir / 'knn' / 'redundancy.csv')
    labels_source = np.array([f'source_{l}' for l in df.source.unique()])
    labels_target = np.array([f'target_{l}' for l in df.target.unique()])
    labels = np.concatenate([labels_source, labels_target])
    colors = plotly.colors.qualitative.Plotly
    colors = np.concatenate([colors, colors])
    indexer = {k: i for i, k in enumerate(labels)}
    source = [indexer[f'source_{k}'] for k in df.source]
    target = [indexer[f'target_{k}'] for k in df.target]
    values = df.amount.values

    fig = go.Figure(
        data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
            )
        )]
    )
    fig.update_layout(font_size=16, font_family='Liberatinus Serif')
    fig.write_html(settings.figures_dir / 'knn_redundancy.html')
    fig.write_image(settings.figures_dir / 'knn_redundancy.pdf')
    fig.write_image(settings.figures_dir / 'knn_redundancy.png', scale=2)


def plot_uniqueness():
    df = pd.read_csv(settings.results_dir / 'knn' / 'uniqueness_per_organ.csv', index_col=None)
    df = df.melt(value_name='unique spectra', var_name='organ')

    fig = px.bar(data_frame=df,
                 x="organ",
                 y='unique spectra',
                 color='organ')
    fig.update_layout(font_size=16, font_family='Liberatinus Serif')
    fig.write_html(settings.figures_dir / 'knn_uniqueness_per_organ.html')
    fig.write_image(settings.figures_dir / 'knn_uniqueness_per_organ.pdf')
    fig.write_image(settings.figures_dir / 'knn_uniqueness_per_organ.png', scale=2)

    df = pd.read_csv(settings.results_dir / 'knn' / 'uniqueness.csv', index_col=None)
    df = df.melt(value_name='unique spectra', var_name='organ')

    fig = px.bar(data_frame=df,
                 x="organ",
                 y='unique spectra',
                 color='organ')
    fig.update_layout(font_size=16, font_family='Liberatinus Serif')
    fig.write_html(settings.figures_dir / 'knn_uniqueness.html')
    fig.write_image(settings.figures_dir / 'knn_uniqueness.pdf')
    fig.write_image(settings.figures_dir / 'knn_uniqueness.png', scale=2)

    df = pd.read_csv(settings.results_dir / 'knn' / 'repetitions_per_sample.csv', index_col=None)
    df = df.groupby('organ', group_keys=False).apply(normalize_repetitions)
    df = df.rename({'repetitions': 'repetitions [%]'}, axis=1)
    fig = px.strip(data_frame=df,
                   x="organ",
                   y='repetitions [%]',
                   color='organ')
    fig.update_layout(font_size=16, font_family='Liberatinus Serif')
    fig.write_html(settings.figures_dir / 'knn_repetitions_per_sample.html')
    fig.write_image(settings.figures_dir / 'knn_repetitions_per_sample.pdf')
    fig.write_image(settings.figures_dir / 'knn_repetitions_per_sample.png', scale=2)


def normalize_repetitions(df: pd.DataFrame):
    df.repetitions = 100 * (df.repetitions / df.repetitions.sum())
    return df


@click.command()
@click.option('--redundancy', is_flag=True, help="plot how often spectra from one organ is also assigned to a "
                                                 "different organ by KNN")
@click.option('--uniqueness', is_flag=True, help="plot how many unique spectra are present on each organ")
def main(redundancy: bool, uniqueness: bool):
    if redundancy:
        plot_redundancy()

    if uniqueness:
        plot_uniqueness()


if __name__ == '__main__':
    main()
