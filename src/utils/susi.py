import cProfile
import functools
import io
import pstats
import time
from typing import List, Union

import numpy as np
import pandas as pd


def profile(_func=None, *, filter_name="src/"):
    def deco_profile(func):
        @functools.wraps(func)
        def profile_wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            value = func(*args, **kwargs)
            profiler.disable()
            s = io.StringIO()
            sort_by = 'cumulative'
            ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(filter_name)
            print(s.getvalue())
            return value
        return profile_wrapper

    if _func is None:
        return deco_profile
    else:
        return deco_profile(_func)


class MeasureTime:
    def __init__(self, print_time=False):
        self.s = None
        self.e = None
        self.print_time = print_time

    def __enter__(self):
        self.s = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.e = time.time()
        if self.print_time:
            print(f"duration: {self.e - self.s}\n")


class ExperimentResults:
    def __init__(self, names: Union[List, None] = None):
        if isinstance(names, str):
            self.names = [names]
            self._data = {n: [] for n in names}
        elif isinstance(names, List):
            self.names = names
            self._data = {n: [] for n in names}
        elif names is None:
            self.names = []
            self._data = {}
        else:
            raise ValueError(f"Unrecognized names: {names}")

    def _init_name_in_data(self, name):
        if name not in self._data:
            self._data[name] = []

    def get_names(self):
        return self.names

    def append(self, name, value):
        if name not in self.names:
            self.names.append(name)
            self._init_name_in_data(name)
        if isinstance(value, list):
            self._data[name] += value
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            self._data[name].append(value)
        elif isinstance(value, np.ndarray):
            self._data[name] += list(value)
        else:
            raise ValueError(f"Can not interpret value: {value}")

    def check_values(self):
        lengths = {k: len(self._data[k]) for k in self._data}
        if not np.all([lengths[k] for k in lengths]):
            raise ValueError(f"Not all elements in self._data have same length: {lengths}")

    def get_df(self):
        if self.names is None:
            return
        results = pd.DataFrame(self._data)
        return results

