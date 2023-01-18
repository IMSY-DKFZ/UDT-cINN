from domain_adaptation.trainers.domain_adaptation_trainer_base import DomainAdaptationTrainerBase
from domain_adaptation.trainers.inn_trainer_base import DAInnBase
from domain_adaptation.trainers.conditional_inn_trainer import CondinitionalDomainAdaptationINN
from domain_adaptation.trainers.gan_inn_trainer import GANDomainAdaptationInn
from domain_adaptation.trainers.unit_trainer import UNIT
from domain_adaptation.trainers.inn_trainer import DomainAdaptationInn
from domain_adaptation.trainers.gan_conditional_inn_trainer import GanCondinitionalDomainAdaptationINN
from domain_adaptation.trainers.waic_trainer import WAICTrainer
from domain_adaptation.trainers.vae_trainer import VAETrainer
from domain_adaptation.trainers.gan_vae_trainer import GANVAETrainer
from domain_adaptation.trainers.autoencoder_gan_conditional_inn_trainer import VariationalAutoencoderGanConditionalInn


def get_model(experiment_name: str):
    model_dictionary = {
        "inn": DomainAdaptationInn,
        "gan_inn": GANDomainAdaptationInn,
        "cinn": CondinitionalDomainAdaptationINN,
        "gan_cinn": GanCondinitionalDomainAdaptationINN,
        "unit": UNIT,
        "waic": WAICTrainer,
        "vae": VAETrainer,
        "gan_vae": GANVAETrainer,
        "vae_gan_cinn": VariationalAutoencoderGanConditionalInn
    }

    try:
        model = model_dictionary[experiment_name]
    except KeyError:
        try:
            model = model_dictionary[experiment_name.replace("_mnist", "")]
        except KeyError:
            raise KeyError("Please specify a valid experiment name")

    return model
