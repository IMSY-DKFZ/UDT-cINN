from argparse import ArgumentParser
from omegaconf.omegaconf import DictConfig, ListConfig
from src.utils.config_io import load_config
import os


class DomainAdaptationParser(ArgumentParser):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        for key, value in self.config.items():
            if isinstance(value, DictConfig):
                for sub_key, sub_value in value.items():
                    data_type = type(sub_value) if not isinstance(sub_value, bool) else str
                    sub_value = str(sub_value)
                    self.add_argument(
                        f"--{key}.{sub_key}",
                        type=data_type,
                        default=sub_value,
                        required=False
                    )
            else:
                data_type = type(value) if not isinstance(value, bool) else str
                value = str(value)
                self.add_argument(
                    f"--{key}",
                    type=data_type,
                    default=value,
                    required=False
                )

    def get_new_config(self):
        parsed_args = self.parse_args()
        parsed_dict = vars(parsed_args)
        nested_items = list()
        for key, value in parsed_dict.items():
            if value in ["True", "true", "False", "false"]:
                if value in ["True", "true"]:
                    parsed_dict[key] = True
                if value in ["False", "false"]:
                    parsed_dict[key] = False
            if "." in key:
                nested_items.append((key, value))
        for item in nested_items:
            key, value = item
            main_key, sub_key = key.split(".")
            if main_key not in parsed_dict.keys():
                parsed_dict[main_key] = dict()
            if value in ["True", "true"]:
                parsed_dict[main_key][sub_key] = True
            elif value in ["False", "false"]:
                parsed_dict[main_key][sub_key] = False
            else:
                parsed_dict[main_key][sub_key] = value
            del parsed_dict[key]

        new_config = DictConfig(parsed_dict)

        if new_config.data.used_channels >= 16:
            new_config.data.used_channels = list(range(0, 16))

        if new_config.data.data_dir_a == "test_dir_a" and new_config.data.data_dir_b == "test_dir_b":
            data_set_name = new_config.data.data_set_name
            data_base_path = new_config.data_base_path
            if data_set_name == "mnist_usps" or "mnist" in new_config.experiment_name:
                new_config.data.data_dir_a = os.path.join(data_base_path, "MNIST")
                new_config.data.data_dir_b = os.path.join(data_base_path, "USPS")
            elif data_set_name in ["real_sim", "sim"]:
                log_string = "_log" if new_config.data.log else "_sqrt"
                ms_string = "_ms" if isinstance(new_config.data.used_channels, (ListConfig, list)) else ""
                folder_name = f"min_max_preprocessed_data{log_string}{ms_string}"
                folder_path = os.path.join(data_base_path, folder_name)
                data_dir_b = "bad_simulations" if data_set_name == "sim" else "real_images"
                new_config.data.data_dir_a = os.path.join(folder_path, "good_simulations")
                new_config.data.data_dir_b = os.path.join(folder_path, data_dir_b)
            elif "hsi" in data_set_name:
                new_config.data.data_dir_a = os.path.join(data_base_path, "HSI_Data", "sampled")
                new_config.data.data_dir_b = os.path.join(data_base_path, "HSI_Data", "adapted")
            else:
                raise KeyError("Please select a valid data_set_name out of [mnist_usps, sim, real_sim]!")

        return new_config


if __name__ == "__main__":
    conf = load_config("/home/kris/Work/Repositories/miccai23/configs/gan_cinn_conf.yaml")
    print(conf)
    parser = DomainAdaptationParser(conf)
    new_conf = parser.get_new_config()
    print(new_conf)




