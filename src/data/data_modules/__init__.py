from src.data.data_modules.domain_adaptation_data_module import DomainAdaptationDataModule
from src.data.data_modules.domain_adaptation_data_module_hsi import DomainAdaptationDataModuleHSI
from src.data.data_modules.waic_data_module import WAICDataModule


def get_data_module(experiment_name: str):
    data_module = DomainAdaptationDataModule
    if "hsi" in experiment_name:
        data_module = DomainAdaptationDataModuleHSI
    if experiment_name == "waic":
        data_module = WAICDataModule

    return data_module
