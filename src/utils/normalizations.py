import numpy as np


def range_normalization(data, data_range=(0, 1)):

    data_normalized = min_max_normalization(data)

    data_normalized *= (data_range[1] - data_range[0])
    data_normalized += data_range[0]
    return data_normalized


def min_max_normalization(data):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn
    data_normalized /= old_range

    return data_normalized


def normalize_min_max(data, log=True):
    if log is True:
        data = np.log10(data)
    mx = np.amax(data)
    mn = np.amin(data)
    normalized_data = min_max_normalization(data, eps=1e-8)

    return normalized_data, mx, mn


def normalize_min_max_range(data, data_range=(0, 1), log=True):
    if log is True:
        # data = np.log10(data)
        data = np.sqrt(data)
    mx = np.amax(data)
    mn = np.amin(data)
    normalized_data = range_normalization(data, data_range=data_range)

    return normalized_data, mx, mn


def normalize_min_max_inverse(data, mx=1, mn=0, log=True):
    if log is True:
        data = 1 - data
        data *= -(mx - mn)
        data += mx
        data = 10 ** data
    else:
        data *= mx - mn
        data += mx

    return data


def spectral_min_max_normalizaiton(spectral_data: np.ndarray, log: bool = True):
    if log is True:
        spectral_data = np.log10(spectral_data)
    maximum = np.max(spectral_data)
    minimum = np.min(spectral_data)



def standardize(data, log=True):
    if log is True:
        data = np.log10(data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data, mean, std


def standardize_inverse(data, mean=0, std=1, log=True):
    data = data * std + mean
    if log is True:
        data = 10 ** data
    return data