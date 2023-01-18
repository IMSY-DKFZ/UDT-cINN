import torch
from abc import abstractmethod, ABC
from typing import Tuple, Dict
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from domain_adaptation.figures import col_bar


class DomainAdaptationTrainerBase(pl.LightningModule, ABC):
    def __init__(self, experiment_config: DictConfig):
        super().__init__()

        self.config = experiment_config

        self.dimensions = self.config.data.dimensions
        self.channels = self.dimensions[0]
        self.dimensionality = np.product(self.dimensions)

    def model_name(self):
        return self._get_name()

    def aggregate_total_loss(self, losses_dict: Dict):
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
    def get_images(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images_a = batch["image_a"]
        images_b = batch["image_b"]
        images_a, images_b = images_a.cuda(non_blocking=True), images_b.cuda(non_blocking=True)

        return images_a, images_b

    def spectral_consistency_loss(self, images_a, images_b, images_ab, images_ba, sc_criterion=0.7):
        """

        :param images_a:
        :param images_b:
        :param images_ab:
        :param images_ba:
        :param sc_criterion:
        :return:
        """

        a_mask_orig = images_a[:, 0, :, :] > sc_criterion
        b_mask_orig = images_b[:, 0, :, :] > sc_criterion

        a_mask = a_mask_orig.unsqueeze(1)
        b_mask = b_mask_orig.unsqueeze(1)

        a_mask = torch.repeat_interleave(a_mask, self.channels, 1)
        b_mask = torch.repeat_interleave(b_mask, self.channels, 1)

        x_a = images_a[a_mask] / torch.norm(images_a, dim=1)[a_mask_orig].repeat(self.channels)
        x_ab = images_ab[a_mask] / torch.norm(images_ab, dim=1)[a_mask_orig].repeat(self.channels)
        x_b = images_b[b_mask] / torch.norm(images_b, dim=1)[b_mask_orig].repeat(self.channels)
        x_ba = images_ba[b_mask] / torch.norm(images_ba, dim=1)[b_mask_orig].repeat(self.channels)
        spectral_consistency_loss = self.recon_criterion(x_a, x_ab) + self.recon_criterion(x_b, x_ba)
        spectral_consistency_loss = torch.log(spectral_consistency_loss)
        spectral_consistency_loss *= self.config.sc_weight
        return spectral_consistency_loss

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
        Each model must implement this method as it defines how the images from each domain are mapped into the latent
        space and then to the other domain

        :param inp:
        :param mode:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def translate_image(self, image, input_domain="a"):
        """

        :param image:
        :param input_domain:
        :return:
        """
        pass

    def test_step(self, batch, batch_idx, *args, **kwargs):
        path = os.path.join(self.config.save_path, "testing")
        generated_image_data_path = os.path.join(path, "generated_image_data")
        os.makedirs(generated_image_data_path, exist_ok=True)

        images_a, images_b = self.get_images(batch)
        if len(images_a) == 5 or len(images_a) == 5:
            images_a, images_b = torch.squeeze(images_a, dim=0), torch.squeeze(images_b, dim=0)

        images_ab = self.translate_image(images_a, input_domain="a")
        images_ba = self.translate_image(images_b, input_domain="b")

        images_a = images_a.cpu().numpy()
        images_b = images_b.cpu().numpy()
        images_ab = images_ab.cpu().numpy()
        images_ba = images_ba.cpu().numpy()

        np.savez(os.path.join(generated_image_data_path, f"test_batch_{batch_idx}"),
                 images_a=images_a,
                 images_b=images_b,
                 images_ab=images_ab,
                 images_ba=images_ba,
                 seg_a=batch["seg_a"],
                 seg_b=batch["seg_b"],
                 oxy_a=batch["oxy_a"],
                 oxy_b=batch["oxy_b"],
                 )

        if True:
            generated_images_path = os.path.join(path, "generated_images")
            os.makedirs(generated_images_path, exist_ok=True)
            plt.figure(figsize=(6, 6))
            plt.subplot(2, 2, 1)
            plt.title("Domain A")
            img_a = plt.imshow(images_a[0, 0, :, :])
            col_bar(img_a)
            plt.subplot(2, 2, 2)
            plt.title("Domain A to Domain B")
            img_ab = plt.imshow(images_ab[0, 0, :, :])
            col_bar(img_ab)
            plt.subplot(2, 2, 3)
            plt.title("Domain B")
            img_b = plt.imshow(images_b[0, 0, :, :])
            col_bar(img_b)
            plt.subplot(2, 2, 4)
            plt.title("Domain B to Domain A")
            img_ba = plt.imshow(images_ba[0, 0, :, :])
            col_bar(img_ba)
            plt.savefig(os.path.join(generated_images_path, f"test_batch_{batch_idx}.png"))
            plt.close()
