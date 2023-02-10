import torch
from abc import ABC
import numpy as np
from src.trainers import DomainAdaptationTrainerBasePA
from src.visualization import col_bar
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


class DAInnBase(DomainAdaptationTrainerBasePA, ABC):

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

        p_a = torch.sum(z_a ** 2, dim=1)
        loss_a = 0.5 * p_a - jac_a
        loss_a = loss_a.mean()
        loss_b = 0.5 * torch.sum(z_b ** 2, dim=1) - jac_b
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

        ml_loss = self.maximum_likelihood_loss(z_a=z_a[0] if isinstance(z_a, tuple) else z_a,
                                               jac_a=jac_a,
                                               z_b=z_b[0] if isinstance(z_b, tuple) else z_b,
                                               jac_b=jac_b)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z_a, jac_a, z_b, jac_b

    def sample_inverted_image(self, image, mode="a", visualize=False):
        mn, mx = image.min(), image.max()
        torch.manual_seed(42)
        z, _ = self.forward(image, mode=mode)
        torch.manual_seed(42)
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

    def translate_image(self, image, input_domain="a"):
        domains = ["a", "b"]
        z, _ = self.forward(image, mode=input_domain)
        domains.remove(input_domain)
        translated_image, _ = self.forward(z, rev=True, mode=domains[0])
        return translated_image

    def get_inn_optimizer(self):
        inn_optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer,
                                                               factor=0.2,
                                                               patience=8,
                                                               threshold=0.005,
                                                               cooldown=2)
        return [inn_optimizer], [{"scheduler": scheduler, "monitor": "loss_step"}]

    def gan_inn_training_step(self, batch, optimizer_idx):

        images_a, images_b = self.get_images(batch)
        if images_b.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        if optimizer_idx == 0:
            z_a, jac_a = self.forward(images_a, mode="a")
            z_b, jac_b = self.forward(images_b, mode="b")

            ml_loss = self.maximum_likelihood_loss(z_a=z_a[0] if isinstance(z_a, tuple) else z_a,
                                                   jac_a=jac_a,
                                                   z_b=z_b[0] if isinstance(z_b, tuple) else z_b,
                                                   jac_b=jac_b)
            ml_loss *= self.config.ml_weight
            batch_dictionary = {"ml_loss": ml_loss}

            images_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            images_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            gen_a_loss = self.discriminator_a.calc_gen_loss(images_ba[0] if isinstance(images_ba, tuple) else images_ba)
            gen_b_loss = self.discriminator_b.calc_gen_loss(images_ab[0] if isinstance(images_ab, tuple) else images_ab)

            gen_loss = gen_a_loss + gen_b_loss

            gen_loss *= self.config.gan_weight
            batch_dictionary["gen_loss"] = gen_loss

        elif optimizer_idx == 1:
            z_a, jac_a = self.forward(images_a, mode="a")
            z_b, jac_b = self.forward(images_b, mode="b")

            images_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            images_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            dis_a_loss = self.discriminator_a.calc_dis_loss(
                images_ba[0].detach() if isinstance(images_ba, tuple) else images_ba.detach(),
                images_a[0] if isinstance(images_a, tuple) else images_a)

            dis_b_loss = self.discriminator_b.calc_dis_loss(
                images_ab[0].detach() if isinstance(images_ab, tuple) else images_ab.detach(),
                images_b[0] if isinstance(images_b, tuple) else images_b)

            dis_loss = dis_a_loss + dis_b_loss
            dis_loss *= self.config.gan_weight

            batch_dictionary = {"dis_loss": dis_loss}

        else:
            raise IndexError("There are more optimizers than specified!")

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(batch_dictionary)
        return batch_dictionary

    def inn_training_step(self, batch):
        images_a, images_b = self.get_images(batch)
        if images_b.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        batch_dictionary, z_a, jac_a, z_b, jac_b = self.maximum_likelihood_training(images_a, images_b)

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(losses_dict=batch_dictionary)

        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        images_a, images_b = self.get_images(batch)
        z_a, jac_a = self.forward(images_a, mode="a")
        images_ab, _ = self.forward(z_a, mode="b", rev=True)
        z_b, jac_b = self.forward(images_b, mode="b")
        images_aba, _ = self.forward(self.forward(images_ab, mode="b")[0], mode="a", rev=True)
        images_ba, _ = self.forward(z_b, mode="a", rev=True)
        images_bab, _ = self.forward(self.forward(images_ba, mode="a")[0], mode="b", rev=True)
        images_a = images_a[0].cpu().numpy() if isinstance(images_a, tuple) else images_a.cpu().numpy()
        images_b = images_b[0].cpu().numpy() if isinstance(images_b, tuple) else images_b.cpu().numpy()
        images_ab = images_ab[0].cpu().numpy() if isinstance(images_ab, tuple) else images_ab.cpu().numpy()
        images_ba = images_ba[0].cpu().numpy() if isinstance(images_ba, tuple) else images_ba.cpu().numpy()
        images_bab = images_bab[0].cpu().numpy() if isinstance(images_bab, tuple) else images_bab.cpu().numpy()
        images_aba = images_aba[0].cpu().numpy() if isinstance(images_aba, tuple) else images_aba.cpu().numpy()
        z_a = z_a[0].cpu().numpy() if isinstance(z_a, tuple) else z_a.cpu().numpy()
        z_b = z_b[0].cpu().numpy() if isinstance(z_b, tuple) else z_b.cpu().numpy()
        z_a_flat = z_a.flatten()
        z_b_flat = z_b.flatten()
        mean_z_a, std_z_a = norm.fit(z_a_flat)
        mean_z_b, std_z_b = norm.fit(z_b_flat)
        x_space = np.linspace(-3, 3, 500)
        y_z_a = norm.pdf(x_space, mean_z_a, std_z_a)
        y_z_b = norm.pdf(x_space, mean_z_b, std_z_b)
        plt.subplot(2, 4, 1)
        plt.title("Simulated Image")
        img_a = plt.imshow(images_a[0, 0, :, :])
        col_bar(img_a)
        plt.subplot(2, 4, 2)
        plt.title("Cycle reconstruction Sim")
        img_aba = plt.imshow(images_aba[0, 0, :, :])
        col_bar(img_aba)
        plt.subplot(2, 4, 3)
        plt.title("Latent Sim Dist")
        plt.hist(z_a_flat, density=True, bins=50)
        plt.plot(x_space, y_z_a, label="mean={:1.2f}, std={:1.2f}".format(mean_z_a, std_z_a))
        plt.legend()
        plt.subplot(2, 4, 4)
        plt.title("Simulation to Real Image")
        img_ab = plt.imshow(images_ab[0, 0, :, :])
        col_bar(img_ab)
        plt.subplot(2, 4, 5)
        plt.title("Real Image")
        img_b = plt.imshow(images_b[0, 0, :, :])
        col_bar(img_b)
        plt.subplot(2, 4, 6)
        plt.title("Cycle Reconstruction Real")
        img_bab = plt.imshow(images_bab[0, 0, :, :])
        col_bar(img_bab)
        plt.subplot(2, 4, 7)
        plt.title("Latent Real Dist")
        plt.hist(z_b_flat, density=True, bins=50)
        plt.plot(x_space, y_z_b, label="mean={:1.2f}, std={:1.2f}".format(mean_z_b, std_z_b))
        plt.legend()
        plt.subplot(2, 4, 8)
        plt.title("Real to Simulated Image")
        img_ba = plt.imshow(images_ba[0, 0, :, :])
        col_bar(img_ba)
        plt.savefig(os.path.join(self.config.save_path, f"val_im_{self.current_epoch}.png"))
        plt.close()
