import torch
from abc import ABC
import numpy as np
from src.trainers import DomainAdaptationTrainerBaseHSI
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


class DAInnBaseHSI(DomainAdaptationTrainerBaseHSI, ABC):

    def maximum_likelihood_loss(self,
                                z_a: torch.Tensor,
                                jac_a: torch.Tensor,
                                z_b: torch.Tensor,
                                jac_b: torch.Tensor) -> torch.Tensor:

        """
        Computes the maximum likelihood loss.

        :param z_a: Latent space representation of spectrum a.
        :param jac_a: Jacobian of spectrum a.
        :param z_b: Latent space representation of spectrum b.
        :param jac_b: Jacobian of spectrum b.
        :return: Maximum likelihood loss,
        """

        loss_a = 0.5 * torch.sum(z_a ** 2, dim=1) - jac_a
        loss_a = loss_a.mean()
        loss_b = 0.5 * torch.sum(z_b ** 2, dim=1) - jac_b
        loss_b = loss_b.mean()
        total_ml_loss = loss_a + loss_b
        total_ml_loss /= self.dimensions

        return total_ml_loss

    def maximum_likelihood_training(self, spectra_a, spectra_b, batch_dictionary: dict = None):
        """

        :param spectra_a:
        :param spectra_b:
        :param batch_dictionary:
        :return:
        """
        if batch_dictionary is None:
            batch_dictionary = dict()

        z_a, jac_a = self.forward(spectra_a, mode="a")
        z_b, jac_b = self.forward(spectra_b, mode="b")

        ml_loss = self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z_a, jac_a, z_b, jac_b

    def sample_inverted_image(self, spectrum, mode="a", visualize=False):
        torch.manual_seed(42)
        z, _ = self.forward(spectrum, mode=mode)
        torch.manual_seed(42)
        x_inv, _ = self.forward(z, rev=True, mode=mode)
        print(f"Maximum Deviation: {torch.max(torch.abs(x_inv - spectrum))}",
              f"Mean Deviation: {torch.mean(torch.abs(x_inv - spectrum))}")

        orig_spectrum = spectrum.cpu().detach().numpy().squeeze()
        inverted_spectrum = x_inv.cpu().detach().numpy().squeeze()

        if visualize:
            plt.plot(orig_spectrum[0], label="input spectrum")
            plt.plot(z[0].cpu().detach().numpy(), label="latent variable")
            plt.plot(inverted_spectrum[0], label="Inverse Image")
            plt.legend()
            plt.show()
        else:
            plt.close()

        return orig_spectrum, inverted_spectrum

    def translate_spectrum(self, spectrum, input_domain="a"):
        domains = ["a", "b"]
        z, _ = self.forward(spectrum, mode=input_domain)
        domains.remove(input_domain)
        translated_spectrum, _ = self.forward(z, rev=True, mode=domains[0])
        return translated_spectrum

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
        spectra_a, spectra_b = self.get_spectra(batch)
        if spectra_b.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        if optimizer_idx == 0:
            z_a, jac_a = self.forward(spectra_a, mode="a")
            z_b, jac_b = self.forward(spectra_b, mode="b")

            ml_loss = self.maximum_likelihood_loss(z_a=z_a[0] if isinstance(z_a, tuple) else z_a,
                                                   jac_a=jac_a,
                                                   z_b=z_b[0] if isinstance(z_b, tuple) else z_b,
                                                   jac_b=jac_b)
            ml_loss *= self.config.ml_weight
            batch_dictionary = {"ml_loss": ml_loss}

            spectra_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            spectra_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            gen_a_loss = self.discriminator_a.calc_gen_loss(spectra_ba[0] if isinstance(spectra_ba, tuple) else spectra_ba)
            gen_b_loss = self.discriminator_b.calc_gen_loss(spectra_ab[0] if isinstance(spectra_ab, tuple) else spectra_ab)

            gen_loss = gen_a_loss + gen_b_loss

            gen_loss *= self.config.gan_weight
            batch_dictionary["gen_loss"] = gen_loss

        elif optimizer_idx == 1:
            z_a, jac_a = self.forward(spectra_a, mode="a")
            z_b, jac_b = self.forward(spectra_b, mode="b")

            spectra_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            spectra_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            dis_a_loss = self.discriminator_a.calc_dis_loss(
                spectra_ba[0].detach() if isinstance(spectra_ba, tuple) else spectra_ba.detach(),
                spectra_a[0] if isinstance(spectra_a, tuple) else spectra_a)

            dis_b_loss = self.discriminator_b.calc_dis_loss(
                spectra_ab[0].detach() if isinstance(spectra_ab, tuple) else spectra_ab.detach(),
                spectra_b[0] if isinstance(spectra_b, tuple) else spectra_b)

            dis_loss = dis_a_loss + dis_b_loss
            dis_loss *= self.config.gan_weight

            batch_dictionary = {"dis_loss": dis_loss}

        else:
            raise IndexError("There are more optimizers than specified!")

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(batch_dictionary)
        return batch_dictionary

    def inn_training_step(self, batch):
        spectra_a, spectra_b = self.get_spectra(batch)
        if spectra_a.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        batch_dictionary, z_a, jac_a, z_b, jac_b = self.maximum_likelihood_training(spectra_a, spectra_b)

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(losses_dict=batch_dictionary)

        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(15, 7))
        spectra_a, spectra_b = self.get_spectra(batch)
        z_a, jac_a = self.forward(spectra_a, mode="a")
        spectra_ab, _ = self.forward(z_a, mode="b", rev=True)
        z_b, jac_b = self.forward(spectra_b, mode="b")
        spectra_aba, _ = self.forward(self.forward(spectra_ab, mode="b")[0], mode="a", rev=True)
        spectra_ba, _ = self.forward(z_b, mode="a", rev=True)
        spectra_bab, _ = self.forward(self.forward(spectra_ba, mode="a")[0], mode="b", rev=True)

        spectra_a = spectra_a[0].cpu().numpy()[0] if isinstance(spectra_a, tuple) else spectra_a.cpu().numpy()[0]
        spectra_b = spectra_b[0].cpu().numpy()[0] if isinstance(spectra_b, tuple) else spectra_b.cpu().numpy()[0]

        spectra_ab = spectra_ab[0].cpu().numpy()[0] if isinstance(spectra_ab, tuple) else spectra_ab.cpu().numpy()[0]
        spectra_ba = spectra_ba[0].cpu().numpy()[0] if isinstance(spectra_ba, tuple) else spectra_ba.cpu().numpy()[0]

        spectra_bab = spectra_bab[0].cpu().numpy()[0] if isinstance(spectra_bab, tuple) else spectra_bab.cpu().numpy()[0]
        spectra_aba = spectra_aba[0].cpu().numpy()[0] if isinstance(spectra_aba, tuple) else spectra_aba.cpu().numpy()[0]

        z_a = z_a[0].cpu().numpy()[0] if isinstance(z_a, tuple) else z_a.cpu().numpy()[0]
        z_b = z_b[0].cpu().numpy()[0] if isinstance(z_b, tuple) else z_b.cpu().numpy()[0]

        mean_z_a, std_z_a = norm.fit(z_a)
        mean_z_b, std_z_b = norm.fit(z_b)
        x_space = np.linspace(-3, 3, 500)
        y_z_a = norm.pdf(x_space, mean_z_a, std_z_a)
        y_z_b = norm.pdf(x_space, mean_z_b, std_z_b)

        minimum, maximum = np.min([spectra_a, spectra_b]), np.max([spectra_a, spectra_b])
        minimum -= 0.2 * np.abs(minimum)
        maximum += 0.2 * np.abs(maximum)

        plt.subplot(3, 1, 1)
        plt.title("HSI Spectra")
        organ_label_a = batch["mapping"][str(int(batch["seg_a"].cpu()))]
        organ_label_b = batch["mapping"][str(int(batch["seg_b"].cpu()))]
        plt.plot(spectra_a, color="green", linestyle="solid", label=f"{organ_label_a} spectrum domain A")
        plt.plot(spectra_aba, color="green", linestyle="", marker="o", label="cycle reconstructed spectrum A")
        plt.plot(spectra_b, color="blue", linestyle="solid", label=f"{organ_label_b} spectrum domain B")
        plt.plot(spectra_bab, color="blue", linestyle="", marker="o", label="cycle reconstructed spectrum B")
        plt.ylim(minimum, maximum)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.title("Domain adapted spectra")
        plt.plot(spectra_ab, color="green", linestyle="dashed", label=f"{organ_label_a} spectrum domain AB")
        plt.plot(spectra_ba, color="blue", linestyle="dashed", label=f"{organ_label_b} spectrum domain BA")
        plt.ylim(minimum, maximum)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.hist(z_a, color="green", density=True, bins=50, alpha=0.5, label="spectrum domain A")
        plt.plot(x_space, y_z_a, color="green", label="mean={:1.2f}, std={:1.2f}".format(mean_z_a, std_z_a))

        plt.hist(z_b, color="blue", density=True, bins=50, alpha=0.5, label="spectrum domain B")
        plt.plot(x_space, y_z_b, color="blue", label="mean={:1.2f}, std={:1.2f}".format(mean_z_b, std_z_b))
        plt.legend()

        plt.savefig(os.path.join(self.config.save_path, f"val_spectrum_{self.current_epoch}.png"))
        plt.close()
