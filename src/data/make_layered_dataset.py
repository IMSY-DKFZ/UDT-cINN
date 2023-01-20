import click
import numpy as np
from typing import *

import pandas as pd

from src.data.multi_layer_loader import SimulationDataLoader
from src.util.susi import adapt_to_camera_reflectance
from src import settings


def find_nearest_value(x: np.ndarray, v: Union[float, int], rtol=1e-6):
    diff = np.abs(x - v)
    idx = diff.argmin()
    assert np.any(diff < rtol), "tolerance value not satisfied"
    return x[idx]


def subsample_simulations(df, wavelengths: np.ndarray):
    """

    :param df: dataframe containing simulations
    :param wavelengths: wavelengths in meters
    :return:
    """
    dataset_wavelengths = df.reflectances.columns.values.astype(float)
    columns = np.array([find_nearest_value(dataset_wavelengths, wv) for wv in wavelengths])
    drop_columns = [str(wv) for wv in dataset_wavelengths if wv not in columns]
    df_sample = df.drop(drop_columns, axis=1, inplace=False, level=1)
    return df_sample


def store_results(df: pd.DataFrame, name: str, dataset: str):
    results_folder = settings.intermediates_dir / 'simulations' / 'multi_layer' / f'{dataset}_{name}'
    results_folder.mkdir(exist_ok=True)
    df[df['split'] == 'train'].to_csv(results_folder / 'train.csv', index=False)
    df[df['split'] == 'test'].to_csv(results_folder / 'test.csv', index=False)


def adapt_simulations_to_cameras(dataset: str):
    loader = SimulationDataLoader()
    df = loader.get_database(simulation=dataset, splits=['train', 'test'])
    # subsample wavelengths
    df_sampled = subsample_simulations(df, wavelengths=np.arange(500, 1000, 5)*10**-9)
    store_results(df_sampled, name='sampled', dataset=dataset)
    # adapt to camera filter responses and illumination irradiance
    df_adapted = adapt_to_camera_reflectance(batch=df,
                                             filter_response=settings.tivita_cam_filters_file,
                                             irradiance=settings.tivita_irradiance)
    layer_cols = [c for c in df.columns.get_level_values(0).unique() if 'layer' in c]
    assert np.all(df[layer_cols].values == df_adapted[layer_cols].values), "optical properties do not match"
    store_results(df_adapted, name='adapted', dataset=dataset)


@click.command()
@click.option('--adapt', default=False, type=bool, help="Adapt simulations to cameras")
@click.option('--dataset', default='generic_depth', type=str, help="ID of dataset to use")
def main(adapt: bool, dataset: str):
    if adapt:
        adapt_simulations_to_cameras(dataset=dataset)


if __name__ == '__main__':
    main()
