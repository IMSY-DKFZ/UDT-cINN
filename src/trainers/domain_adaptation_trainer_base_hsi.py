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
    def aggregate_total_loss(losses_dict: Dict, val_run: bool = False):
        if val_run:
            total_loss_key = "val_loss"
        else:
            total_loss_key = "loss"
        total_loss = 0
        for loss_name, loss_value in losses_dict.items():
            total_loss += loss_value
            losses_dict[loss_name] = loss_value.detach()
        losses_dict[total_loss_key] = total_loss
        return losses_dict

    def log_losses(self, losses_dict: Dict):
        for loss_key, loss_value in losses_dict.items():
            self.log(loss_key, loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.config.batch_size)

    def get_spectra(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        expects that domain_a is the simulated data and domain_b is the real data
        """
        spectra_a = batch["spectra_a"]
        spectra_b = batch["spectra_b"]
        spectra_a, spectra_b = spectra_a.cuda(non_blocking=True), spectra_b.cuda(non_blocking=True)
        if self.config.condition == "segmentation":
            seg_a = torch.tensor([batch["order"][int(label)] for label in batch["seg_a"]]).type(torch.float32)
            ret_data = (spectra_a, seg_a), spectra_b
        else:
            ret_data = spectra_a, spectra_b

        return ret_data

    def get_label_conditions(self, labels, n_labels: int, labels_size=None):
        if isinstance(labels, int) and labels == 0:
            one_hot_seg_shape = [labels_size[0], n_labels]

            if self.config.real_labels == "constant":
                labels = torch.ones(size=one_hot_seg_shape) / n_labels
            elif self.config.real_labels == "noise":
                labels = torch.rand(size=one_hot_seg_shape)

            # random_segmentation = np.random.choice(range(n_labels), size=one_hot_seg_shape)
            #
            # labels = torch.from_numpy(random_segmentation).type(torch.float32)

            one_hot_seg = labels.type(torch.float32)

        else:
            if self.config.label_noise:
                one_hot_seg = torch.stack(
                    [(labels == label) + torch.rand_like(labels) * self.config.label_noise_level for label in
                     range(n_labels)],
                    dim=1
                )
            else:
                one_hot_seg = torch.stack([(labels == label) for label in range(n_labels)], dim=1)

        one_hot_seg /= torch.linalg.norm(one_hot_seg, dim=1, keepdim=True, ord=1)

        assert one_hot_seg.size()[1] == n_labels

        return one_hot_seg

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

        spectra_a = spectra_a[0].cpu().numpy() if isinstance(spectra_a, tuple) else spectra_a.cpu().numpy()
        spectra_b = spectra_b[0].cpu().numpy() if isinstance(spectra_b, tuple) else spectra_b.cpu().numpy()
        spectra_ab = spectra_ab[0].cpu().numpy() if isinstance(spectra_ab, tuple) else spectra_ab.cpu().numpy()
        spectra_ba = spectra_ba[0].cpu().numpy() if isinstance(spectra_ba, tuple) else spectra_ba.cpu().numpy()

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
                 seg_a=batch["seg_a"].cpu().numpy(),
                 seg_b=batch["seg_b"].cpu().numpy(),
                 subjects_a=batch["subjects_a"],
                 subjects_b=batch["subjects_b"],
                 image_ids_a=batch["image_ids_a"],
                 image_ids_b=batch["image_ids_b"],
                 )

        if True:
            organ_label_a = batch["mapping"][str(int(batch["seg_a"][0].cpu()))]
            organ_label_b = batch["mapping"][str(int(batch["seg_b"][0].cpu()))]
            generated_spectra_path = os.path.join(path, "generated_spectra")
            os.makedirs(generated_spectra_path, exist_ok=True)
            plt.figure(figsize=(6, 6))
            plt.plot(spectra_a[0], color="green", linestyle="solid", label=f"{organ_label_a} spectrum domain A")
            plt.plot(spectra_b[0], color="blue", linestyle="solid", label=f"{organ_label_b} spectrum domain B")
            plt.plot(spectra_ab[0], color="green", linestyle="dashed", label=f"{organ_label_a} spectrum domain AB")
            plt.plot(spectra_ba[0], color="blue", linestyle="dashed", label=f"{organ_label_b} spectrum domain BA")
            plt.legend()
            plt.savefig(os.path.join(generated_spectra_path, f"test_batch_{batch_idx}.png"))
            plt.close()
