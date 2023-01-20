import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv


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
intermediates_dir = unify_path(os.getenv('PATH_MICCAI_23_INTERMEDIATES'))
figures_dir = unify_path(Path(os.getenv('PATH_MICCAI_23_PROJECT')) / 'figures')
tivita_cam_filters_file = intermediates_dir / 'optics' / 'artificial_tivita_camera_normal.csv'
tivita_irradiance = intermediates_dir / 'optics' / 'tivita_relative_irradiance_2019_04_05.txt'
tivita_semantic = unify_path(os.getenv('PATH_MICCAI_23_SEMANTIC_DATASET'))
