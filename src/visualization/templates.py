import plotly as ply


palette = ply.colors.qualitative.Plotly
cmap_qualitative = {
    'real': palette[3],
    'simulated': '#127475',
    'cINN': palette[2],
    'UNIT': palette[1]
}


cmap_quantitative = {
    'real': 'Purples',
    'simulated': 'Tempo',
    'cINN': 'Greens',
    'UNIT': 'Reds'
}
