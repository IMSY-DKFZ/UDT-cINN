import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from domain_adaptation.trainers import DomainAdaptationTrainerBase
from domain_adaptation.models.inn_subnets import subnet_conv, subnet_res_net
from domain_adaptation.models.multiscale_invertible_blocks import append_multi_scale_inn_blocks
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os


class WAICTrainer(DomainAdaptationTrainerBase):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.model = self.build_model()

    @staticmethod
    def get_images(batch) -> torch.Tensor:
        images = batch["image"]
        images = images.cuda(non_blocking=True)

        return images

    def forward(self, inp, *args, **kwargs):

        out, jac = self.model(inp, *args, **kwargs)

        return out, jac

    def maximum_likelihood_loss(self,
                                z: torch.Tensor,
                                jac: torch.Tensor) -> torch.Tensor:

        """
        Computes the maximum likelihood loss.

        :param z: Latent space representation of the input image.
        :param jac: Jacobian of the input image.
        :return: Maximum likelihood loss,
        """

        p = torch.sum(z ** 2, dim=1)
        loss = 0.5 * p - jac
        loss = loss.mean()
        ml_loss = loss / self.dimensionality

        return ml_loss

    def maximum_likelihood_training(self, images, batch_dictionary: dict = None):
        """

        :param images:
        :param batch_dictionary:
        :return:
        """

        if batch_dictionary is None:
            batch_dictionary = dict()

        z, jac = self.forward(images)

        ml_loss = self.maximum_likelihood_loss(z=z, jac=jac)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z, jac

    def inn_training_step(self, batch):
        images = self.get_images(batch)
        if images.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        batch_dictionary, z, jac = self.maximum_likelihood_training(images)

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(losses_dict=batch_dictionary)

        return batch_dictionary

    def get_inn_optimizer(self):
        inn_optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer,
                                                               factor=0.2,
                                                               patience=8,
                                                               threshold=0.005,
                                                               cooldown=2)
        return [inn_optimizer], [{"scheduler": scheduler, "monitor": "loss_step"}]

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.inn_training_step(batch)

    def configure_optimizers(self):
        return self.get_inn_optimizer()

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        images = self.get_images(batch)
        z, jac = self.forward(images)
        images_inv, _ = self.forward(z, rev=True)

        images = images.cpu().numpy()
        images_inv = images_inv.cpu().numpy()
        z = z.cpu().numpy()
        z_flat = z.flatten()

        mean_z, std_z = norm.fit(z_flat)

        x_space = np.linspace(-3, 3, 500)
        y_z = norm.pdf(x_space, mean_z, std_z)

        plt.subplot(1, 3, 1)
        plt.title("Real Image")
        plt.imshow(images[0, 0, :, :])
        plt.subplot(1, 3, 2)
        plt.title("Real Image inv")
        plt.imshow(images_inv[0, 0, :, :])
        plt.subplot(1, 3, 3)
        plt.title("Latent Real Dist")
        plt.hist(z_flat, density=True, bins=50)
        plt.plot(x_space, y_z, label="mean={:1.2f}, std={:1.2f}".format(mean_z, std_z))
        plt.legend()
        plt.savefig(os.path.join(self.config.save_path, f"val_im_{self.current_epoch}.png"))
        plt.close()

    def test_step(self, batch, batch_idx, *args, **kwargs):

        path = os.path.join(self.config.save_path, "testing")
        ml_loss_data_path = os.path.join(path, "ml_values")
        os.makedirs(ml_loss_data_path, exist_ok=True)

        images = self.get_images(batch)
        z, jac = self.forward(images)

        ml_loss = self.maximum_likelihood_loss(z, jac)

        np.savez(os.path.join(ml_loss_data_path, f"test_batch_{batch_idx}"),
                 ml_loss=ml_loss.cpu().numpy()
                 )

    def build_model(self):
        model = Ff.SequenceINN(*self.dimensions)
        append_multi_scale_inn_blocks(
            model,
            num_scales=self.config.downsampling_levels,
            blocks_per_scale=[
                self.config.high_res_conv,
                self.config.middle_res_conv,
                self.config.low_res_conv
            ],
            dimensions=self.dimensions,
            subnets=[subnet_conv, subnet_conv, subnet_res_net],
            clamping=self.config.clamping,
            instant_downsampling=self.config.instant_downsampling
        )

        append_multi_scale_inn_blocks(
            model,
            num_scales=1,
            blocks_per_scale=self.config.low_res_conv,
            dimensions=self.dimensions,
            subnets=subnet_res_net,
            clamping=self.config.clamping
        )

        model.append(Fm.Flatten)

        return model

    def translate_image(self, image, input_domain="a"):
        pass
