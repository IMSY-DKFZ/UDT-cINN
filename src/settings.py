import json
import os
from pathlib import Path
from typing import Union

import numpy as np
from dotenv import load_dotenv

here = Path(__file__).parent


def unify_path(path: Union[Path, str]) -> Path:
    """
    Tries to bring some consistency to paths:
        - Resolve home directories (~ → /home/username).
        - Make paths absolute.

    :param path: The original path.
    :returns: The unified path.
    """
    if isinstance(path, str):
        path = Path(path)

    if str(path).startswith('~'):
        # resolve() function cannot handle paths starting with ~. This expands ~ to the home path in this case
        path = path.expanduser()

    # Normalize the path (this makes it also absolute)
    return path.resolve()


repo_dir = Path(__file__)
dotenv_path = repo_dir.parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

# Set Path variables
intermediates_dir = unify_path(Path(os.getenv('HSI_DATA_PATH') or '') / 'intermediates')
results_dir = unify_path(Path(os.getenv('SAVE_DATA_PATH') or '') / 'results')
figures_dir = unify_path(Path(os.getenv('SAVE_DATA_PATH') or '') / 'figures')
tivita_cam_filters_file = intermediates_dir / 'optics' / 'artificial_tivita_camera_normal_20nm.csv'
tivita_irradiance = intermediates_dir / 'optics' / 'tivita_relative_irradiance_2019_04_05.txt'
tivita_semantic = unify_path(Path(intermediates_dir) / 'semantic_v2')


# load pre-defined organ mapping
with open(str(here / 'data/mapping.json'), 'rb') as handle:
    mapping = json.load(handle)
with open(str(here / 'data/semantic_organ_labels.json'), 'rb') as handle:
    organ_labels = json.load(handle)['organ_labels']

for directory in [results_dir, figures_dir]:
    os.makedirs(directory, exist_ok=True)

pai_wavelengths = np.arange(700, 855, 10)
