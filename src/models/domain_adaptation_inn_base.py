import torch
from abc import abstractmethod, ABC
from typing import Tuple, Dict, overload, Callable

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
from domain_adaptation.models.inn_subnets import subnet_conv, subnet_conv_1x1
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


class DomainAdaptationInn(pl.LightningModule, ABC):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        self.channels = self.config.data.dimensions[0]
        self.dimensionality = np.product(self.config.data.dimensions)

        self.downsampling_levels = self.config.downsampling_levels

        self.spectral_consistency = self.config.spectral_consistency

        self.inn_model = self.build_model()

    def model_name(self):
        return self._get_name()

    def maximum_likelihood_loss(self,
                                z_a: torch.Tensor,
                                jac_a: torch.Tensor,
                                z_b: torch.Tensor,
                                jac_b: torch.Tensor) -> torch.Tensor:

        """
        Computes the maximum likelihood loss.

        :param z_a: Latent space representation of image a.
        :param jac_a: Jacobian of image a.
        :param z_b: Latent space representation of image b.
        :param jac_b: Jacobian of image b.
        :return: Maximum likelihood loss,
        """
        # print(jac_a.size(), "\n")
        p_a = torch.sum(z_a ** 2, dim=1)#[1, 2, 3], keepdim=True)
        # print(p_a.size(), "\n")
        loss_a = 0.5 * p_a - jac_a
        loss_a = loss_a.mean()
        loss_b = 0.5 * torch.sum(z_b ** 2, dim=1) - jac_b#[1, 2, 3], keepdim=True) - jac_b
        loss_b = loss_b.mean()
        total_ml_loss = loss_a + loss_b
        total_ml_loss /= self.dimensionality

        return total_ml_loss

    def maximum_likelihood_training(self, images_a, images_b, batch_dictionary: dict = None):
        """

        :param images_a:
        :param images_b:
        :param batch_dictionary:
        :return:
        """
        if batch_dictionary is None:
            batch_dictionary = dict()

        z_a, jac_a = self.forward(images_a, mode="a")
        z_b, jac_b = self.forward(images_b, mode="b")

        ml_loss = self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z_a, jac_a, z_b, jac_b

    @staticmethod
    def get_images(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images_a = batch["image_a"]
        images_b = batch["image_b"]
        images_a, images_b = images_a.cuda(non_blocking=True), images_b.cuda(non_blocking=True)

        return images_a, images_b

    @overload
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

    def forward(self, inp, mode="a", *args, **kwargs):
        if mode not in ["a", "b"]:
            raise AttributeError("Specify either mode 'a' or 'b'!")

        out, jac = self.inn_model(inp, *args, **kwargs)
        return out, jac

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images_a, images_b = self.get_images(batch)
        if images_a.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        batch_dictionary, z_a, jac_a, z_b, jac_b = self.maximum_likelihood_training(images_a, images_b)

        if self.spectral_consistency:
            batch_dictionary, images_ab, images_ba = self.spectral_consistency_training(
                images_a, images_b, z_a, z_b, batch_dictionary=batch_dictionary)



        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(losses_dict=batch_dictionary)

        return batch_dictionary

    @overload
    @abstractmethod
    def model_loss(self, images_a, images_b, z_a, jac_a, z_b, jac_b):
        """

        :param images_a:
        :param images_b:
        :param z_a:
        :param jac_a:
        :param z_b:
        :param jac_b:
        :return:
        """
        pass

    def model_loss(self, z_a, jac_a, z_b, jac_b):
        return self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)

    def sample_inverted_image(self, image, mode="a", visualize=False):
        mn, mx = image.min(), image.max()
        z, _ = self.forward(image, mode=mode)
        x_inv, _ = self.forward(z, rev=True, mode=mode)
        print(f"Maximum Deviation: {torch.max(torch.abs(x_inv - image))}",
              f"Mean Deviation: {torch.mean(torch.abs(x_inv - image))}")

        orig_image = image.cpu().detach().numpy().squeeze()
        inverted_image = x_inv.cpu().detach().numpy().squeeze()

        if visualize:
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(orig_image[0, :, :], vmin=mn, vmax=mx)
            plt.subplot(1, 3, 2)
            plt.title("Latent space")
            plt.imshow(z.cpu().detach().numpy().squeeze()[0, :, :])
            plt.subplot(1, 3, 3)
            plt.title("Inverse Image")
            plt.imshow(inverted_image[0, :, :], vmin=mn, vmax=mx)
            plt.show()
        else:
            plt.close()

        return orig_image, inverted_image

    def append_all_in_one_block(self, nodes: Ff.SequenceINN, sub_network: Callable,
                                condition=None, condition_shape=None):
        nodes.append(Fm.AllInOneBlock,
                     subnet_constructor=sub_network,
                     cond=condition,
                     cond_shape=condition_shape,
                     affine_clamping=self.config.clamping,
                     gin_block=False,
                     global_affine_init=1.,
                     global_affine_type="SOFTPLUS",
                     permute_soft=False,
                     learned_householder_permutation=0,
                     reverse_permutation=False)
        return nodes

    @overload
    @abstractmethod
    def build_model(self):
        """

        :return:
        """
        pass

    def build_model(self):
        downsampling_block = Fm.HaarDownsampling
        nodes = Ff.SequenceINN(*self.config.data.dimensions)

        nodes = self.append_all_in_one_block(nodes, sub_network=subnet_conv)
        nodes.append(downsampling_block)

        for k in range(2):
            if k % 2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv

            nodes = self.append_all_in_one_block(nodes, sub_network=subnet)

        return nodes

    @overload
    @abstractmethod
    def configure_optimizers(self):
        pass

    def configure_optimizers(self):
        inn_optimizer, scheduler = self.get_inn_optimizer()
        return inn_optimizer, scheduler

    def get_inn_optimizer(self):
        inn_optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer,
                                                               factor=0.2,
                                                               patience=8,
                                                               threshold=0.005,
                                                               cooldown=2)
        return [inn_optimizer], [{"scheduler": scheduler, "monitor": "loss_step"}]

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
            self.log(loss_key, loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def spectral_consistency_loss(self, images_a, images_b, images_ab, images_ba):
        a_mask_orig = images_a[:, 0, :, :] > 0.7
        b_mask_orig = images_b[:, 0, :, :] > 0.7

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

    def spectral_consistency_training(self, images_a, images_b, z_a, z_b, batch_dictionary: dict = None):
        """

        :param images_a:
        :param images_b:
        :param z_a:
        :param z_b:
        :param batch_dictionary:
        :return:
        """
        images_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
        images_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)
        spectral_consistency_loss = self.spectral_consistency_loss(images_a,
                                                                   images_b,
                                                                   images_ab,
                                                                   images_ba)

        batch_dictionary["sc_loss"] = spectral_consistency_loss

        return batch_dictionary, images_ab, images_ba

    def recon_criterion(self, model_recon, target_recon):
        if self.config.recon_criterion == "mse":
            recon_error = torch.mean((model_recon - target_recon)**2)
        elif self.config.recon_criterion == "abs":
            recon_error = torch.mean(torch.abs(model_recon - target_recon))
        else:
            raise KeyError("Please use one of the implemented reconstruction criteria "
                           "'abs' or 'mse' in config.recon_criterion!")
        return recon_error

    def reconstruction_training(self, images_a, images_b, batch_dictionary: dict = None):
        """

        :param images_a:
        :param images_b:
        :param batch_dictionary:
        :return:
        """
        if batch_dictionary is None:
            batch_dictionary = dict()

        z_a, jac_a = self.forward(images_a, mode="a")
        z_b, jac_b = self.forward(images_b, mode="b")

        ml_loss = self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z_a, jac_a, z_b, jac_b

    def validation_step(self, batch, batch_idx):
        # print(batch_idx)
        plt.figure(figsize=(20, 5))
        images_a, images_b = self.get_images(batch)
        z_a, jac_a = self.forward(images_a, mode="a")
        images_ab, _ = self.forward(z_a, mode="b", rev=True)
        z_b, jac_b = self.forward(images_b, mode="b")
        images_aba, _ = self.forward(self.forward(images_ab, mode="b")[0], mode="a", rev=True)
        images_ba, _ = self.forward(z_b, mode="a", rev=True)
        images_bab, _ = self.forward(self.forward(images_ba, mode="a")[0], mode="b", rev=True)
        images_a = images_a.cpu().numpy()
        images_b = images_b.cpu().numpy()
        images_ab = images_ab.cpu().numpy()
        images_ba = images_ba.cpu().numpy()
        images_bab = images_bab.cpu().numpy()
        images_aba = images_aba.cpu().numpy()
        z_a = z_a.cpu().numpy()
        z_b = z_b.cpu().numpy()
        z_a_flat = z_a.flatten()
        z_b_flat = z_b.flatten()
        mean_z_a, std_z_a = norm.fit(z_a_flat)
        mean_z_b, std_z_b = norm.fit(z_b_flat)
        x_space = np.linspace(-3, 3, 500)
        y_z_a = norm.pdf(x_space, mean_z_a, std_z_a)
        y_z_b = norm.pdf(x_space, mean_z_b, std_z_b)
        plt.subplot(2, 4, 1)
        plt.title("Simulated Image")
        plt.imshow(images_a[0, 0, :, :])
        plt.subplot(2, 4, 2)
        plt.title("Cycle reconstruction Sim")
        plt.imshow(images_aba[0, 0, :, :])
        plt.subplot(2, 4, 3)
        plt.title("Latent Sim Dist")
        plt.hist(z_a_flat, density=True, bins=50)
        plt.plot(x_space, y_z_a, label="mean={:1.2f}, std={:1.2f}".format(mean_z_a, std_z_a))
        plt.legend()
        plt.subplot(2, 4, 4)
        plt.title("Simulation to Real Image")
        plt.imshow(images_ab[0, 0, :, :])
        plt.subplot(2, 4, 5)
        plt.title("Real Image")
        plt.imshow(images_b[0, 0, :, :])
        plt.subplot(2, 4, 6)
        plt.title("Cycle Reconstruction Real")
        plt.imshow(images_bab[0, 0, :, :])
        plt.subplot(2, 4, 7)
        plt.title("Latent Real Dist")
        plt.hist(z_b_flat, density=True, bins=50)
        plt.plot(x_space, y_z_b, label="mean={:1.2f}, std={:1.2f}".format(mean_z_b, std_z_b))
        plt.legend()
        plt.subplot(2, 4, 8)
        plt.title("Real to Simulated Image")
        plt.imshow(images_ba[0, 0, :, :])
        plt.savefig(os.path.join(self.config.save_path, f"test_im_{self.current_epoch}.png"))
        plt.close()
