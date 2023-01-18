import numpy as np


def calculate_downscale_dimensionality(dim: (list, np.ndarray), downscale_factor: int) -> list:
    feature_dim = int(round(dim[0]*downscale_factor**2))
    imd_dim_1 = int(round(dim[1]/downscale_factor))
    imd_dim_2 = int(round(dim[2]/downscale_factor))
    return [feature_dim, imd_dim_1, imd_dim_2]
