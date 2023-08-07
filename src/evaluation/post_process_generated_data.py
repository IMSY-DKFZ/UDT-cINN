import numpy as np
import glob
import os
import click

from src.utils.config_io import load_config


@click.command()
@click.option('--path', type=str, help="path to the output directory of a trained model")
@click.option('--data_set_type', type=str, default='real_images',
              help="target data set for domain transfer (real_images of good_simulation)")
def main(path: str, data_set_type: str):

    config = load_config(os.path.join(path, "version_0", "hparams.yaml"))

    if data_set_type in config.data.data_dir_a:
        source_images = "images_b"
        seg = "seg_b"
        target_images = "images_ba"
        oxy = "oxy_b"
    elif data_set_type in config.data.data_dir_b:
        source_images = "images_a"
        seg = "seg_a"
        target_images = "images_ab"
        oxy = "oxy_a"
    else:
        raise ValueError("Please select a valid data_set_type ('real_images', 'good_simulation')!")

    file_list = glob.glob(os.path.join(path, "testing", "generated_image_data", "*.npz"))

    for file_idx, file in enumerate(file_list):
        data = np.load(file)
        orig_image = np.squeeze(data[source_images])
        recon = np.squeeze(data[target_images])
        segmentation = np.squeeze(data[seg])

        save_path = file.replace("generated_image_data", "training")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if data_set_type == "good_simulations":
            # oxygenation = data["oxy"]
            np.savez(save_path,
                     reconstruction=recon,
                     segmentation=segmentation,
                     # oxygenation=oxygenation,
                     )
        else:
            np.savez(save_path,
                     reconstruction=recon,
                     segmentation=segmentation)


if __name__ == '__main__':
    main()

