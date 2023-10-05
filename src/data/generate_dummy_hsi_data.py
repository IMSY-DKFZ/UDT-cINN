# SPDX-FileCopyrightText: Intelligent Medical Systems, DKFZ
# see LICENSE for more details

import click
import os
import numpy as np


def generate_dummy_data(output: str):
    assert os.path.isdir(output), "output folder does not exist"
    destination_folders = [
        "train",
        "val",
        "test"
    ]
    template_name = "P{pig_id:03d}#2019_12_12_00_00_{ind:02d}{suffix}.npy"
    data_shape = (10, 100)
    classes = [3, 7, 9, 11, 17, 23, 26, 27, 28, 29, 31]
    for f in destination_folders:
        data_folder = os.path.join(output, f)
        os.makedirs(data_folder, exist_ok=True)
        for i in range(10):
            identifiers_ref = dict(pig_id=i, ind=i, suffix="")
            identifiers_seg = dict(pig_id=i, ind=i, suffix="_seg")
            name_ref = template_name.format(**identifiers_ref)
            name_seg = template_name.format(**identifiers_seg)
            print(name_ref, name_seg)

            x = np.random.rand(*data_shape)
            y = np.random.choice(classes, data_shape[0], replace=True)
            np.save(os.path.join(data_folder, name_ref), x)
            np.save(os.path.join(data_folder, name_seg), y)


@click.command()
@click.option('--generate', is_flag=True, help="Generates a set of dummy files with random noise to replace "
                                               "real data")
@click.option('--output', help="path to folder where dummy data should be stored. The following folders will be "
                               "generated in this directory: ['train', 'val', 'test']")
def main(generate: bool, output: str):
    if generate:
        generate_dummy_data(output=output)


if __name__ == '__main__':
    main()
