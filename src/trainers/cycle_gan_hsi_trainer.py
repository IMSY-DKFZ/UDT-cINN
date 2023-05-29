import numpy as np
from omegaconf import DictConfig
from src.trainers import DomainAdaptationTrainerBaseHSI
import torch
from src.models.discriminator import DiscriminatorHSI
from src.models.vae_hsi import VariationalAutoencoderHSI

from src.models.inn_subnets import weight_init
import matplotlib.pyplot as plt
import os


class CycleGANHSITrainer(DomainAdaptationTrainerBaseHSI):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        if self.config.condition == "segmentation":
            self.config.gen.conditional_input_dim = self.dimensions + self.config.data.n_classes

        self.gen_a = VariationalAutoencoderHSI(config=experiment_config.gen, input_dim=self.dimensions)
        self.gen_b = VariationalAutoencoderHSI(config=experiment_config.gen, input_dim=self.dimensions)

        self.dis_a = DiscriminatorHSI(self.config.dis, self.dimensions)
        self.dis_b = DiscriminatorHSI(self.config.dis, self.dimensions)

        # Network weight initialization
        self.apply(lambda m: weight_init(m, gain=1.))
        self.dis_a.apply(lambda m: weight_init(m, gain=1., method="gaussian"))
        self.dis_b.apply(lambda m: weight_init(m, gain=1., method="gaussian"))

    def forward(self, inp, mode="a", *args, **kwargs):
        if mode == "a":
            out = self.gen_a(inp)
        elif mode == "b":
            out = self.gen_b(inp)
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")
        return out

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        spectra_a, spectra_b = self.get_spectra(batch)

        conditioning = False
        if isinstance(spectra_a, tuple):
            conditioning = True

            seg_a = self.get_label_conditions(spectra_a[1], n_labels=self.config.data.n_classes)
            seg_a = seg_a.cuda()

            seg_b = self.get_label_conditions(labels=0, n_labels=self.config.data.n_classes, labels_size=spectra_a[1].size())
            seg_b = seg_b.cuda()

        if optimizer_idx == 0:

            spectra_ab = self.gen_ab(spectra_a if not conditioning else torch.cat([spectra_a[0], seg_a], dim=1))
            spectra_ba = self.gen_ba(spectra_b if not conditioning else torch.cat([spectra_b, seg_b], dim=1))

            spectra_aba = self.gen_ba(spectra_ab if not conditioning else torch.cat([spectra_ab, seg_a], dim=1))
            spectra_bab = self.gen_ab(spectra_ba if not conditioning else torch.cat([spectra_ba, seg_a], dim=1))

            # cycle consistency loss
            loss_gen_cyc_x_a = self.recon_criterion(spectra_aba, spectra_a if not conditioning else spectra_a[0])
            loss_gen_cyc_x_b = self.recon_criterion(spectra_bab, spectra_b)

            # GAN loss
            gen_ba_loss = self.dis_a.calc_gen_loss(spectra_ba)
            gen_ab_loss = self.dis_b.calc_gen_loss(spectra_ab)

            gen_loss = self.config["gan_w"] * (gen_ab_loss + gen_ba_loss)

            cc_recon_loss = self.config['recon_x_cyc_w'] * (loss_gen_cyc_x_a + loss_gen_cyc_x_b)

            batch_dictionary = {"gen_loss": gen_loss,
                                "cc_recon_loss": cc_recon_loss,
                                }

        elif optimizer_idx == 1:
            spectra_ab = self.gen_ab(spectra_a if not conditioning else torch.cat([spectra_a[0], seg_a], dim=1))
            spectra_ba = self.gen_ba(spectra_b if not conditioning else torch.cat([spectra_b, seg_b], dim=1))

            loss_dis_a = self.dis_a.calc_dis_loss(spectra_ba.detach(), spectra_a if not conditioning else spectra_a[0])
            loss_dis_b = self.dis_b.calc_dis_loss(spectra_ab.detach(), spectra_b)

            loss_dis_total = self.config['gan_w'] * (loss_dis_a + loss_dis_b)
            batch_dictionary = {"dis_loss": loss_dis_total}

        else:
            raise IndexError("There are more optimizers than specified!")

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(batch_dictionary)
        return batch_dictionary

    def configure_optimizers(self):
        beta1 = self.config['beta1']
        beta2 = self.config['beta2']
        gen_params = list(self.gen_ab.parameters()) + list(self.gen_ba.parameters())
        gen_optimizer = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                         lr=self.config.lr, betas=(beta1, beta2),
                                         weight_decay=self.config['weight_decay'])

        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                         lr=self.config.lr, betas=(beta1, beta2),
                                         weight_decay=self.config["weight_decay"])

        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=self.config['step_size'],
                                                        gamma=self.config['gamma'], last_epoch=-1)
        dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size=self.config['step_size'],
                                                        gamma=self.config['gamma'], last_epoch=-1)

        return [gen_optimizer, dis_optimizer], \
               [{"scheduler": gen_scheduler, "monitor": "loss_step"},
                {"scheduler": dis_scheduler, "monitor": "loss_step"}]

    def build_generator(self):
        pass

    def sample_inverted_image(self, image, mode="a", visualize=False):
        pass

    def translate_spectrum(self, spectrum, input_domain="a"):
        if self.config.condition == "segmentation":
            if isinstance(spectrum, tuple):
                seg = self.get_label_conditions(spectrum[1], n_labels=self.config.data.n_classes)
                spectrum = spectrum[0]

            else:
                seg = self.get_label_conditions(labels=0, n_labels=self.config.data.n_classes,
                                                labels_size=spectrum.size())

            seg = seg.cuda()
            input_spectrum = torch.cat([spectrum, seg], dim=1)
        else:
            input_spectrum = spectrum

        translated_spectrum = self.forward(input_spectrum, mode=input_domain)
        return translated_spectrum

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        spectra_a, spectra_b = self.get_spectra(batch)

        conditioning = False
        if isinstance(spectra_a, tuple):
            conditioning = True

            seg_a = self.get_label_conditions(spectra_a[1], n_labels=self.config.data.n_classes)
            seg_a = seg_a.cuda()

            seg_b = self.get_label_conditions(labels=0, n_labels=self.config.data.n_classes,
                                              labels_size=spectra_a[1].size())
            seg_b = seg_b.cuda()

        spectra_ab = self.gen_a(spectra_a if not conditioning else torch.cat([spectra_a[0], seg_a], dim=1))
        spectra_ba = self.gen_b(spectra_b if not conditioning else torch.cat([spectra_b, seg_b], dim=1))

        spectra_aba = self.gen_b(spectra_ab if not conditioning else torch.cat([spectra_ab, seg_a], dim=1))
        spectra_bab = self.gen_a(spectra_ba if not conditioning else torch.cat([spectra_ba, seg_a], dim=1))

        spectra_a = spectra_a.cpu().numpy()[0] if not conditioning else spectra_a[0].cpu().numpy()[0]
        spectra_b = spectra_b.cpu().numpy()[0]
        spectra_ab = spectra_ab.cpu().numpy()[0]
        spectra_ba = spectra_ba.cpu().numpy()[0]
        spectra_bab = spectra_bab.cpu().numpy()[0]
        spectra_aba = spectra_aba.cpu().numpy()[0]

        minimum, maximum = np.min([spectra_a, spectra_b]), np.max([spectra_a, spectra_b])
        minimum -= 0.2 * np.abs(minimum)
        maximum += 0.2 * np.abs(maximum)

        plt.subplot(2, 1, 1)
        plt.title("HSI Spectra")
        organ_label_a = batch["mapping"][str(int(batch["seg_a"][0].cpu()))]
        organ_label_b = batch["mapping"][str(int(batch["seg_b"][0].cpu()))]
        plt.plot(spectra_a, color="green", linestyle="solid", label=f"{organ_label_a} spectrum domain A")
        plt.plot(spectra_aba, color="green", linestyle="", marker="o", label="cycle reconstructed spectrum A")
        plt.plot(spectra_b, color="blue", linestyle="solid", label=f"{organ_label_b} spectrum domain B")
        plt.plot(spectra_bab, color="blue", linestyle="", marker="o", label="cycle reconstructed spectrum B")
        plt.ylim(minimum, maximum)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Domain adapted spectra")
        plt.plot(spectra_ab, color="green", linestyle="dashed", label=f"{organ_label_a} spectrum domain AB")
        plt.plot(spectra_ba, color="blue", linestyle="dashed", label=f"{organ_label_b} spectrum domain BA")
        plt.ylim(minimum, maximum)
        plt.legend()

        plt.savefig(os.path.join(self.config.save_path, f"val_spectrum_{self.current_epoch}.png"))
        plt.close()
