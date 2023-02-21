import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from src.trainers import DomainAdaptationTrainerBaseHSI
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os
from src.models.inn_subnets import weight_init


class WAICTrainerHSI(DomainAdaptationTrainerBaseHSI):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.n_blocks = self.config.n_blocks
        self.model = self.build_model()

    @staticmethod
    def get_spectra(batch) -> torch.Tensor:
        spectra = batch["spectra_b"]
        spectra = spectra.cuda(non_blocking=True)

        return spectra

    def forward(self, inp, *args, **kwargs):

        out, jac = self.model(inp, *args, **kwargs)

        return out, jac

    def maximum_likelihood_loss(self,
                                z: torch.Tensor,
                                jac: torch.Tensor) -> torch.Tensor:

        """
        Computes the maximum likelihood loss.

        :param z: Latent space representation of the input spectrum.
        :param jac: Jacobian of the input spectrum.
        :return: Maximum likelihood loss,
        """

        p = torch.sum(z ** 2, dim=1)
        loss = 0.5 * p - jac
        loss = loss.mean()
        ml_loss = loss / self.dimensions

        return ml_loss

    def maximum_likelihood_training(self, spectra, batch_dictionary: dict = None):
        """

        :param spectra:
        :param batch_dictionary:
        :return:
        """

        if batch_dictionary is None:
            batch_dictionary = dict()

        z, jac = self.forward(spectra)

        ml_loss = self.maximum_likelihood_loss(z=z, jac=jac)
        batch_dictionary["ml_loss"] = ml_loss

        return batch_dictionary, z, jac

    def inn_training_step(self, batch):
        spectra = self.get_spectra(batch)
        if spectra.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        batch_dictionary, z, jac = self.maximum_likelihood_training(spectra)

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

    def subnet(self, ch_in, ch_out):
        net = nn.Sequential(
            nn.Linear(ch_in, self.config.n_hidden),
            nn.ReLU(),
            nn.Linear(self.config.n_hidden, ch_out),
        )

        net.apply(lambda m: weight_init(m, gain=1.))
        return net

    def validation_step(self, batch, batch_idx):

        plt.figure(figsize=(20, 5))
        spectra = self.get_spectra(batch)
        z, jac = self.forward(spectra)
        spectra_inv, _ = self.forward(z, rev=True)

        spectra = spectra.cpu().numpy()
        spectra_inv = spectra_inv.cpu().numpy()
        z = z.cpu().numpy()
        z_flat = z.flatten()

        mean_z, std_z = norm.fit(z_flat)

        x_space = np.linspace(-3, 3, 500)
        y_z = norm.pdf(x_space, mean_z, std_z)

        organ_label = batch["mapping"][str(int(batch["seg_b"].cpu()))]

        plt.subplot(1, 3, 1)
        plt.title("Real spectrum")
        plt.plot(spectra[0], color="green", linestyle="solid", label=f"{organ_label} spectrum")
        plt.subplot(1, 3, 2)
        plt.title("Real spectrum inv")
        plt.plot(spectra_inv[0], color="green", linestyle="solid", label=f"{organ_label} spectrum")
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

        spectra = self.get_spectra(batch)
        z, jac = self.forward(spectra)

        ml_loss = self.maximum_likelihood_loss(z, jac)

        np.savez(os.path.join(ml_loss_data_path, f"test_batch_{batch_idx}"),
                 ml_loss=ml_loss.cpu().numpy()
                 )

    def build_model(self):

        model = Ff.SequenceINN(self.dimensions)

        for c_block in range(self.n_blocks):
            model.append(
                Fm.AllInOneBlock,
                subnet_constructor=self.subnet,
                affine_clamping=self.config.clamping,
                global_affine_init=self.config.actnorm,
                permute_soft=False,
                learned_householder_permutation=self.config.n_reflections,
                reverse_permutation=True,
            )

        return model

    def translate_spectrum(self, spectrum, input_domain="a"):
        pass
