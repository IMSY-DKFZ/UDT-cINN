import torch
from abc import abstractmethod, ABC
from typing import Tuple, Dict
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os


class DomainAdaptationTrainerBaseHSI(pl.LightningModule, ABC):
    def __init__(self, experiment_config: DictConfig):
        super().__init__()

        self.config = experiment_config
        self.dimensions = self.config.data.dimensions

    def model_name(self):
        return self._get_name()

    @staticmethod
    def aggregate_total_loss(losses_dict: Dict):
        total_loss = 0
        for loss_name, loss_value in losses_dict.items():
            total_loss += loss_value
            losses_dict[loss_name] = loss_value.detach()
        losses_dict["loss"] = total_loss
        return losses_dict

    def log_losses(self, losses_dict: Dict):
        for loss_key, loss_value in losses_dict.items():
            self.log(loss_key, loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.config.batch_size)

    @staticmethod
    def get_spectra(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        spectra_a = batch["spectra_a"]
        spectra_b = batch["spectra_b"]
        spectra_a, spectra_b = spectra_a.cuda(non_blocking=True), spectra_b.cuda(non_blocking=True)

        return spectra_a, spectra_b

    def recon_criterion(self, model_recon, target_recon):
        """

        :param model_recon:
        :param target_recon:
        :return:
        """
        if self.config.recon_criterion == "mse":
            recon_error = torch.mean((model_recon - target_recon)**2)
        elif self.config.recon_criterion == "abs":
            recon_error = torch.mean(torch.abs(model_recon - target_recon))
        else:
            raise KeyError("Please use one of the implemented reconstruction criteria "
                           "'abs' or 'mse' in config.recon_criterion!")
        return recon_error

    @abstractmethod
    def forward(self, inp, mode="a", *args, **kwargs):
        """
        Forward pass of the domain adaptation model.
        Each model must implement this method as it defines how the spectra from each domain are mapped into the latent
        space and then to the other domain

        :param inp:
        :param mode:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def translate_spectrum(self, spectrum, input_domain="a"):
        """

        :param spectrum:
        :param input_domain:
        :return:
        """
        pass

    def test_step(self, batch, batch_idx, *args, **kwargs):
        path = os.path.join(self.config.save_path, "testing")
        generated_spectrum_data_path = os.path.join(path, "generated_spectra_data")
        os.makedirs(generated_spectrum_data_path, exist_ok=True)

        spectra_a, spectra_b = self.get_spectra(batch)

        spectra_ab = self.translate_spectrum(spectra_a, input_domain="a")
        spectra_ba = self.translate_spectrum(spectra_b, input_domain="b")

        spectra_a = spectra_a.cpu().numpy()
        spectra_b = spectra_b.cpu().numpy()
        spectra_ab = spectra_ab.cpu().numpy()
        spectra_ba = spectra_ba.cpu().numpy()

        if self.config.normalization not in ["None", "none"]:
            if self.config.normalization == "standardize":
                spectra_a = spectra_a * self.config.data.std_a + self.config.data.mean_a
                spectra_ba = spectra_ba * self.config.data.std_a + self.config.data.mean_a

                spectra_b = spectra_b * self.config.data.std_b + self.config.data.mean_b
                spectra_ab = spectra_ab * self.config.data.std_b + self.config.data.mean_b

        np.savez(os.path.join(generated_spectrum_data_path, f"test_batch_{batch_idx}"),
                 spectra_a=spectra_a,
                 spectra_b=spectra_b,
                 spectra_ab=spectra_ab,
                 spectra_ba=spectra_ba,
                 bvf_a=batch["bvf_a"],
                 bvf_b=batch["bvf_b"],
                 # oxy_a=batch["oxy_a"],
                 # oxy_b=batch["oxy_b"],
                 )

        if True:
            generated_spectra_path = os.path.join(path, "generated_spectra")
            os.makedirs(generated_spectra_path, exist_ok=True)
            plt.figure(figsize=(6, 6))
            plt.plot(spectra_a[0], color="green", linestyle="solid", label="spectrum domain A")
            plt.plot(spectra_b[0], color="blue", linestyle="solid", label="spectrum domain B")
            plt.plot(spectra_ab[0], color="blue", linestyle="dashed", label="spectrum domain AB")
            plt.plot(spectra_ba[0], color="green", linestyle="dashed", label="spectrum domain BA")
            plt.legend()
            plt.savefig(os.path.join(generated_spectra_path, f"test_batch_{batch_idx}.png"))
            plt.close()
