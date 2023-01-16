import logging
from typing import List, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.settings import intermediates_dir

logger = logging.getLogger()


class SimulationDataLoader:
    """Abstract away susi simulation data."""

    def __init__(self,):
        """Construct a susi simulation dataloader.
        """
        base_data_path = intermediates_dir

        self.simulations = ['generic_depth']
        self.splits = ['train', 'validate', 'test']

        self.base_path = (base_data_path / 'simulations' / 'multi_layer')
        self.simulation_paths = {
            'generic_depth': {
                'train': 'generic_depth/train',
                'validate': 'generic_depth/train',
                'test': 'generic_depth/test',
            }
        }

    def get_database(
            self,
            simulation: str,
            splits: Union[str, List[str]] = 'all',
            train_fraction: float = .9,
            seed: int = 28,
    ) -> pd.DataFrame:
        """Build dataframe containing simulation data.

        :simulation: 'thesis', 'visualize', or 'laparoscopic' to load one of
        the corresponding datasets
        :splits: 'train', 'validate', 'test', or 'all'. If 'train' or
        'validate' is specified then the train set of the simulation is split
        using the train_fraction parameter.
        :train_fraction: Fraction of simulation data to use for training (rest
        is validation)
        :seed: Seed for split in train and validation data
        :return: DataFrame with corresponding simulation data. A column called
        'split' is added indicating the split and analogously a column for
        'simulation' is added.
        """

        if splits == 'all':
            splits = self.splits
        if isinstance(splits, str):
            splits = [splits]

        df = []
        for split in splits:

            rel_path = self.simulation_paths[simulation][split]
            full_path = self.base_path / rel_path

            files = sorted(list(full_path.glob('*.csv')))

            local = []
            for fn in tqdm(files, desc='Files:'):
                loc = pd.read_csv(fn, header=[0, 1], index_col=0)
                local.append(loc)
            loc = pd.concat(local, ignore_index=True)

            if split in ['train', 'validate']:
                if train_fraction not in [0., 1.]:
                    train_df, validate_df = train_test_split(
                        loc,
                        train_size=train_fraction,
                        random_state=seed,
                    )
                    loc = train_df if split == 'train' else validate_df
                else:
                    if (split, train_fraction) in [('train', 0.), ('validate', 1.)]:
                        raise ValueError(f'split={split},'
                                         f' train_fraction={train_fraction}'
                                         ' would lead to an empty dataframe.')

            loc['simulation'] = simulation
            loc['split'] = split
            df.append(loc)

        df = pd.concat(df, ignore_index=True)

        df = self._clean_dataframe(df)

        return df

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.

        :df: DataFrame to clean
        :return: Cleaned DataFrame
        """

        with pd.option_context('mode.use_inf_as_null', True):
            df_res = df.dropna()

        fraction_dropped = 1. - len(df_res) / len(df)
        logger.info(f'{fraction_dropped:.2%} of rows dropped because of NaN values.')

        return df_res
