import plotly as ply
from matplotlib.colors import LinearSegmentedColormap


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
    'real': LinearSegmentedColormap.from_list("real", [(0, "white"), (1, cmap_qualitative["real"])]),
    'simulated': LinearSegmentedColormap.from_list("simulated", [(0, "white"), (1, cmap_qualitative["simulated"])]),
    'cINN': LinearSegmentedColormap.from_list("cINN", [(0, "white"), (1, cmap_qualitative["cINN"])]),
    'UNIT': LinearSegmentedColormap.from_list("UNIT", [(0, "white"), (1, cmap_qualitative["UNIT"])])
}

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    cmaps = len(cmap_quantitative)
    a = np.array([np.arange(0, 100) for i in range(10)])
    plt.figure(figsize=(9, 6))
    for i, (name, cmap) in enumerate(cmap_quantitative.items()):
        plt.subplot(cmaps, 1, i + 1)
        plt.title(name)
        img = plt.imshow(a, cmap=cmap)
    plt.show()
