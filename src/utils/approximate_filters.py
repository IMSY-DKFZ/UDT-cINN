"""
===================================================================

Surgical Spectral Imaging Library (SuSI)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt for details.

===================================================================
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
from typing import Union

from src.utils import susi


def approximate_filters(target_filters: str, source_filters: str, optical_system_parts=None,
                        normalize: bool = True) -> np.ndarray:
    """
    create a transformation matrix from one set of filters to another.
    Mostly used for source: multispectral - target: rgb.
    If the created transformation matrix is multiplied with a
    multispectral pixel this will give a rgb estimate for the pixel.

    :param target_filters: the path to/ and filters which we want to approximate by the source filters
    :param source_filters: the path to/ and filters of the target system (e.g. multispectral filter system)
    :param optical_system_parts: other parts of the source optical system, e.g. irradiance
    :param normalize: normalize transformation
    :return: transformation matrix
    """
    wavelengths = np.arange(320, 780, 1) * 10 ** -9
    source_filters = susi.load_filter_response(source_filters, wavelengths)
    # optionally, the optical system of the source can be specified to be included in the transformation:
    if optical_system_parts is not None:
        for COMPONENT in optical_system_parts:
            component_transmission = susi.load_irradiance(COMPONENT, wavelengths)
            source_filters *= component_transmission
    else:
        warnings.warn("No optical system being used")

    target_filters = susi.load_filter_response(target_filters, wavelengths)
    # manually set the weights for the bgr image calculation
    transform = get_spectral_transform(source_filters, target_filters)

    # TODO: Remove if new solution works. Was used in norm = np.sum(reconstruction_np,...).
    # reconstruction_np = np.dot(transform, source_filters)
    if normalize:
        norm = np.sum(np.dot(transform, source_filters), axis=1)[..., np.newaxis]
        transform_normed = transform / norm
        return transform_normed
    else:
        return transform


def get_spectral_transform(filter_original: Union[str, pd.DataFrame], filter_wanted: Union[str, pd.DataFrame],
                           wavelengths: Union[None, np.ndarray] = None,
                           regressor=LinearRegression(normalize=False, fit_intercept=False)) -> np.ndarray:
    """
    get a transformation matrix from the original filter responses to the
    wanted filter responses. E.g., from multispectral to rgb. You can specify the
    wavelengths that should be parsed for determining the transformation

    :param filter_original: the path to original filters, can be a ximea xml,
        a pandas .csv or a pandas dataframe, in the rows are the filters,
        in the columns are the center wavelengths
    :param filter_wanted: the path to wanted filters, can be a ximea xml,
        a pandas .csv or a pandas dataframe, in the rows are the filters,
        in the columns are the center wavelengths
    :param wavelengths: optional wavelength array ([m]),
        a row specifying the points parsed to determine the solution
    :param regressor: the regressor which is used to establish this relationship
    :return: the linear transformation in matrix form:
        a nr_filters_original x nr_filters_wanted matrix
    """
    if wavelengths is None:
        wavelengths = np.arange(450, 700, 1) * 10**-9

    f_original = susi.load_filter_response(filter_original, wavelengths)
    f_wanted = susi.load_filter_response(filter_wanted, wavelengths)

    regressor.fit(f_original.values.T, f_wanted.values.T)

    return regressor.coef_
