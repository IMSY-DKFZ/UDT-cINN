from omegaconf import OmegaConf, DictConfig
import os


def load_config(path: str) -> DictConfig:
    with open(path, "r") as conf_file:
        config = OmegaConf.load(conf_file)
    return config


def save_config(path: str, config: DictConfig):
    with open(path, "w") as conf_file:
        OmegaConf.save(config=config, f=conf_file)


def get_conf_path(current_dir, experiment_name):
    configs_path = os.path.join(current_dir, "configs")
    all_configs = os.listdir(configs_path)
    exp_config = [config_file for config_file in all_configs if f"{experiment_name}_conf" in config_file.split(".")]
    if not exp_config:
        raise FileNotFoundError(f"No suitable config file could be found in {configs_path}!\n"
                                f"Please choose a valid experiment name!")
    else:
        exp_config = exp_config[0]

    return os.path.join(configs_path, exp_config)
