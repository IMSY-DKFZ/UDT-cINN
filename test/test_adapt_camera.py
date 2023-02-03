import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src import settings
from src.utils.susi import adapt_to_camera_reflectance
from src.data.make_layered_dataset import subsample_simulations

this_path = Path(__file__)


class TestAdaptToCamera(unittest.TestCase):
    def setUp(self) -> None:
        self.filters = settings.intermediates_dir / 'optics' / 'artificial_tivita_camera_normal_20nm.csv'
        self.irr = settings.intermediates_dir / 'optics' / 'tivita_relative_irradiance_2019_04_05.txt'
        self.sims_file = settings.intermediates_dir / 'simulations/multi_layer/generic_1_layer/train/0.csv'

    def test_adapt_camera(self):
        df = pd.read_csv(self.sims_file, index_col=None, header=[0, 1])
        df_adapted = adapt_to_camera_reflectance(batch=df,
                                                 filter_response=self.filters,
                                                 irradiance=self.irr,
                                                 optical_system_parts=None)
        df_sampled = subsample_simulations(df=df, wavelengths=np.arange(500, 1000, 5)*10**-9)
        self.assertTrue(np.all(df_sampled.reflectances.values != df_adapted.reflectances.values), "sampled equal to adapted")


if __name__ == '__main__':
    unittest.main()
