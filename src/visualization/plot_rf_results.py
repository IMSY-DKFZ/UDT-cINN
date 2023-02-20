import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import re
from functools import partial

from src import settings
from src.evaluation.si_classifier import get_label_mapping

METRICS_PROCESSED = []


def plot_dots(tr, data: pd.DataFrame, fig: go.Figure):
    metric = re.findall(f'metric=(.*?)<', tr.hovertemplate)[0]
    if metric in METRICS_PROCESSED:
        return
    METRICS_PROCESSED.append(metric)
    tmp = data[data.metric == metric]
    fig.add_trace(go.Box(
        x=tmp.dataset,
        y=tmp.value,
        hoverinfo="skip",
        showlegend=False,
        yaxis=tr.yaxis,
        xaxis=tr.xaxis,
        offsetgroup='boxes',
        fillcolor='silver',
        line=dict(color='black'),
        opacity=0.5
    ))


def plot_rf_results():
    stages = [
        'real',
        'adapted',
        'sampled',
        'adapted_inn'
    ]
    results = []
    for stage in stages:
        file = settings.results_dir / 'rf' / f"rf_classifier_report_{stage}.csv"
        tmp = pd.read_csv(file, index_col=0, header=0)
        tmp['dataset'] = stage
        tmp = tmp.reset_index(names=['metric'])
        tmp = tmp.melt(id_vars=['dataset', 'metric', 'macro avg', 'weighted avg', 'accuracy'], var_name='organ',
                       value_name='value')
        results.append(tmp)
    results = pd.concat(results, axis=0, sort=True)

    fig = px.strip(data_frame=results,
                   x="dataset",
                   y="value",
                   facet_col_spacing=0.08,
                   facet_col="metric",
                   color='organ'
                   )
    fig.update_yaxes(matches=None)

    fig.for_each_trace(partial(plot_dots, data=results, fig=fig))
    fig.update_layout(autosize=False)
    fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.write_html(settings.figures_dir / 'rf_results.html')
    fig.write_image(settings.figures_dir / 'rf_results.pdf')
    fig.write_image(settings.figures_dir / 'rf_results.png')
    results.to_csv(settings.figures_dir / 'rf_results.csv', index=False)


def plot_confusion_matrix():
    stages = [
        'real',
        'adapted',
        'sampled',
        'adapted_inn'
    ]
    mapping = get_label_mapping()
    for stage in stages:
        file = settings.results_dir / 'rf' / f"rf_classifier_matrix_{stage}.npz"
        data = np.load(file)
        matrix = data['matrix']
        labels = data['labels']
        names = [mapping.get(str(l)) for l in labels]
        fig = px.imshow(matrix, text_auto='.2f', color_continuous_scale='Greens')
        fig.update_layout(font=dict(size=16))
        axis_ticks = dict(
                tickmode='array',
                tickvals=np.arange(0, len(names)),
                ticktext=names
            )
        fig.update_layout(
            xaxis=axis_ticks,
            yaxis=axis_ticks
        )
        fig.write_html(settings.figures_dir / f'rf_confusion_matrix_{stage}.html')
        fig.write_image(settings.figures_dir / f'rf_confusion_matrix_{stage}.pdf')
        fig.write_image(settings.figures_dir / f'rf_confusion_matrix_{stage}.png')


@click.command()
@click.option('--rf', is_flag=True, help="plot random forest classification results")
def main(rf: bool):
    if rf:
        plot_confusion_matrix()
        plot_rf_results()


if __name__ == '__main__':
    main()
