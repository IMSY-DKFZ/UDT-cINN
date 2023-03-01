import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import re


def line(data_frame: pd.DataFrame, x, y, color, facet_col=None, **kwargs):
    categories = [c for c in [x, color, facet_col] if c is not None]
    agg = data_frame.groupby(categories).agg({y: ['mean', 'std']}).reset_index()
    data = agg.copy()
    data.drop(y, axis=1, inplace=True, level=0)
    data[y] = agg[(y, 'mean')]
    data['sd'] = agg[(y, 'std')]
    fig = px.line(data_frame=data, x=x, y=y, color=color, facet_col=facet_col, **kwargs)
    for tr in fig.data:
        tr: go.Scattergl
        c = re.findall(f'{color}=(.*?)<', tr.hovertemplate)[0]
        if facet_col:
            f = re.findall(f'{facet_col}=(.*?)<', tr.hovertemplate)[0]
            sd = data.loc[(data[color] == c) & (data[facet_col] == f), :]['sd'].values
        else:
            sd = data.loc[data[color] == c, :]['sd'].values
        y_high = tr.y + (sd)
        y_low = tr.y - (sd)
        band = go.Scatter(
            name=tr.name,
            legendgroup=tr.legendgroup,
            x=list(tr.x) + list(tr.x[::-1]),  # x, then x reversed
            y=list(y_high) + list(y_low)[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=hex_to_rgba(tr.line.color),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            yaxis=tr.yaxis,
            xaxis=tr.xaxis
        )
        fig.add_trace(band)
    return fig, data


def hex_to_rgba(color: str, transparency=0.2):
    hex = color.lstrip('#')
    hlen = len(hex)
    rgba = tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3)) + (transparency,)
    return f'rgba{str(rgba)}'
