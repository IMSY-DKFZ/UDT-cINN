from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import *


class RandomForestEstimator(BaseEstimator):
    """
    This class implements a random forest regressor with default parameters according to the publication by Wirkert et. al.
    `Robust near real-time estimation of physiological parameters <https://link.springer.com/article/10.1007/s11548-016-1376-5>`_
    """
    def __init__(self,
                 min_samples_leaf: Union[int, float] = 10,
                 n_estimators: int = 10,
                 max_depth: int = 9,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        See documentation of Sklearn random forest for an in depth explanation of each parameter
        `Random Forests <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.

        :param min_samples_leaf: The minimum number of samples required to split an internal node:
            * If int, then consider min_samples_split as the minimum number.
            * If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum
            number of samples for each split.
        :param n_estimators:
        :param max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
            until all leaves contain less than min_samples_split samples.
        :param n_jobs: The number of jobs to run in parallel.
        :param verbose: Controls the verbosity when fitting and predicting.
        """
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x: np.ndarray = None, y: np.ndarray = None, sample_weight=None, **kwargs):
        """

        :param x: The training input samples. Internally, its :code:`dtype` will be converted to :code:`dtype=np.float32`.
            If a sparse matrix is provided, it will be converted into a sparse :code:`csc_matrix`.
        :param y: The target values (class labels in classification, real numbers in regression).
        :param sample_weight: Sample weights. If None, then samples are equally weighted. Splits that would create child
            nodes with net zero or negative weight are ignored while searching for a split in each node.
            In the case of classification, splits are also ignored if they would result in any single class carrying a
            negative weight in either child node.
        :param kwargs: additional argument parsed during instantiation of :class:`RandomForestRegressor` from sklearn
        :return: self
        """
        self.n_features_in_ = x.shape[-1]
        self.regressor = RandomForestRegressor(max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf,
                                               n_jobs=self.n_jobs,
                                               n_estimators=self.n_estimators,
                                               verbose=self.verbose,
                                               **kwargs)
        self.regressor.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        estimates blood volume fraction and oxygenation from samples in :code:`x`. :code:`x` need to be in the shape of
        :code:`(n_samples, n_features)`.

        :param x: samples as numpy array
        :return: predicted values
        """
        self.params = self.regressor.predict(x).squeeze()
        return self.params
