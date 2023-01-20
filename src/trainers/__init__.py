from src.trainers.domain_adaptation_trainer_base_pa import DomainAdaptationTrainerBasePA
from src.trainers.domain_adaptation_trainer_base_hsi import DomainAdaptationTrainerBaseHSI
from src.trainers.inn_trainer_base import DAInnBase
from src.trainers.inn_trainer_base_hsi import DAInnBaseHSI
from src.trainers.conditional_inn_trainer import CondinitionalDomainAdaptationINN
from src.trainers.gan_inn_trainer import GANDomainAdaptationInn
from src.trainers.unit_trainer import UNIT
from src.trainers.inn_trainer import DomainAdaptationInn
from src.trainers.gan_conditional_inn_trainer import GanCondinitionalDomainAdaptationINN
from src.trainers.gan_conditional_inn_trainer_hsi import GanCondinitionalDomainAdaptationINNHSI
from src.trainers.waic_trainer import WAICTrainer


def get_model(experiment_name: str):
    model_dictionary = {
        "inn": DomainAdaptationInn,
        "gan_inn": GANDomainAdaptationInn,
        "cinn": CondinitionalDomainAdaptationINN,
        "gan_cinn": GanCondinitionalDomainAdaptationINN,
        "gan_cinn_hsi": GanCondinitionalDomainAdaptationINNHSI,
        "unit": UNIT,
        "waic": WAICTrainer,
    }

    try:
        model = model_dictionary[experiment_name]
    except KeyError:
        try:
            model = model_dictionary[experiment_name.replace("_mnist", "")]
        except KeyError:
            raise KeyError("Please specify a valid experiment name")

    return model
