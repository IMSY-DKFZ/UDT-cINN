import plotly as ply


palette = ply.colors.qualitative.Plotly
cmap_qualitative = {
    'real': "#4363d8",
    'simulated': "#911eb4",
    'cINN': palette[2],
    'UNIT': "#e6194B",
    'unit': "#e6194B",
    'inn': palette[2]
}

cmap_qualitative_diff = {
    'real - simulated': cmap_qualitative['simulated'],
    'real - inn': cmap_qualitative['inn'],
    'real - unit': cmap_qualitative['unit']
}


cmap_quantitative = {
    'real': 'Purples',
    'simulated': 'Tempo',
    'cINN': 'Greens',
    'UNIT': 'Reds'
}
