from omegaconf import DictConfig
from src.trainers import DomainAdaptationTrainerBasePA
import torch
from src.models.discriminator import MultiScaleDiscriminator
from src.models.vae import VariationalAutoEncoder
from src.models.inn_subnets import weight_init
import matplotlib.pyplot as plt
import os


class CycleGANTrainer(DomainAdaptationTrainerBasePA):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.gen_ab = VariationalAutoEncoder(self.config.gen, self.dimensions[0])
        self.gen_ba = VariationalAutoEncoder(self.config.gen, self.dimensions[0])

        self.dis_a = MultiScaleDiscriminator(self.config.dis, self.dimensions[0])
        self.dis_b = MultiScaleDiscriminator(self.config.dis, self.dimensions[0])

        # Network weight initialization
        self.apply(lambda m: weight_init(m, gain=1.))
        self.dis_a.apply(lambda m: weight_init(m, gain=1., method="gaussian"))
        self.dis_b.apply(lambda m: weight_init(m, gain=1., method="gaussian"))

    def forward(self, inp, mode="a", *args, **kwargs):
        if mode == "a":
            out = self.gen_ab(inp)
        elif mode == "b":
            out = self.gen_ba(inp)
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")
        return out

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        images_a, images_b = self.get_images(batch)

        if optimizer_idx == 0:

            images_ab = self.gen_ab(images_a)
            images_ba = self.gen_ba(images_b)

            images_aba = self.gen_ba(images_ab)
            images_bab = self.gen_ab(images_ba)

            # cycle consistency loss
            loss_gen_cyc_x_a = self.recon_criterion(images_aba, images_a)
            loss_gen_cyc_x_b = self.recon_criterion(images_bab, images_b)

            # GAN loss
            gen_ba_loss = self.dis_a.calc_gen_loss(images_ba)
            gen_ab_loss = self.dis_b.calc_gen_loss(images_ab)

            gen_loss = self.config["gan_w"] * (gen_ab_loss + gen_ba_loss)

            cc_recon_loss = self.config['recon_x_cyc_w'] * (loss_gen_cyc_x_a + loss_gen_cyc_x_b)

            batch_dictionary = {"gen_loss": gen_loss,
                                "cc_recon_loss": cc_recon_loss,
                                }

        elif optimizer_idx == 1:
            images_ab = self.gen_ab(images_a)
            images_ba = self.gen_ba(images_b)

            loss_dis_a = self.dis_a.calc_dis_loss(images_ba.detach(), images_a)
            loss_dis_b = self.dis_b.calc_dis_loss(images_ab.detach(), images_b)

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

    def translate_image(self, image, input_domain="a"):
        translated_image = self.forward(image, mode=input_domain)
        return translated_image

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        images_a, images_b = self.get_images(batch)

        images_ab = self.gen_ab(images_a)
        images_ba = self.gen_ba(images_b)

        images_aba = self.gen_ba(images_ab)
        images_bab = self.gen_ab(images_ba)

        images_a = images_a.cpu().numpy()
        images_b = images_b.cpu().numpy()
        images_ab = images_ab.cpu().numpy()
        images_ba = images_ba.cpu().numpy()
        images_bab = images_bab.cpu().numpy()
        images_aba = images_aba.cpu().numpy()

        plt.subplot(2, 3, 1)
        plt.title("Simulated Image")
        plt.imshow(images_a[0, 0, :, :])
        plt.subplot(2, 3, 2)
        plt.title("Simulation to Real Image")
        plt.imshow(images_ab[0, 0, :, :])
        plt.subplot(2, 3, 3)
        plt.title("Cycle reconstruction Sim")
        plt.imshow(images_aba[0, 0, :, :])
        plt.subplot(2, 3, 4)
        plt.title("Real Image")
        plt.imshow(images_b[0, 0, :, :])
        plt.subplot(2, 3, 5)
        plt.title("Real to Simulated Image")
        plt.imshow(images_ba[0, 0, :, :])
        plt.subplot(2, 3, 6)
        plt.title("Cycle Reconstruction Real")
        plt.imshow(images_bab[0, 0, :, :])
        plt.savefig(os.path.join(self.config.save_path, f"val_im_{self.current_epoch}.png"))
        plt.close()
