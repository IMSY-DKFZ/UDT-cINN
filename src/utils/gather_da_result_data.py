import numpy as np
import os
import glob


def gather_da_result_data(model_folder_path: str) -> list:
    """Gathers domain adapted arrays produced by any domain adaptation model.

    :param model_folder_path: Path to the folder where all the checkpoints and the generated test data are stored.
    Usually this is a timestamp of the time when the model training was started.
    :param data_type: 'PAI' or 'HSI' as type of the generated images or spectra, respectively.
    :return: list of dictionaries containing the domain adaptation results
    :raises: FileNotFoundError if model didn't train until the end, i.e. no "testing" folder was created.
    """

    file_list = glob.glob(os.path.join(model_folder_path, "testing", "generated_*_data", "test_batch_*.npz"))

    if not file_list:
        raise FileNotFoundError("This doesn't seem to be a valid experiment model!")

    return_list = list()

    for file_idx, file_path in enumerate(file_list):

        data = np.load(file_path)
        return_list.append({key: data[key] for key in data})

    return return_list
