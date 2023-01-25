import numpy as np
from omegaconf import DictConfig
from src.trainers import DomainAdaptationTrainerBasePA
import torch
from src.models.discriminator import MultiScaleDiscriminator
from src.models.vae import VariationalAutoEncoder
from src.models.inn_subnets import subnet_conv, weight_init
import matplotlib.pyplot as plt
import os


class UNIT(DomainAdaptationTrainerBasePA):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        if self.config.condition == "segmentation":
            self.config.gen.conditional_input_dim = self.dimensions[0] + self.config.data.n_classes

        self.gen_a = VariationalAutoEncoder(self.config.gen, self.dimensions[0])
        self.gen_b = VariationalAutoEncoder(self.config.gen, self.dimensions[0])

        self.dis_a = MultiScaleDiscriminator(self.config.dis, self.dimensions[0])
        self.dis_b = MultiScaleDiscriminator(self.config.dis, self.dimensions[0])

        # Network weight initialization
        self.apply(lambda m: weight_init(m, gain=1.))
        self.dis_a.apply(lambda m: weight_init(m, gain=1., method="gaussian"))
        self.dis_b.apply(lambda m: weight_init(m, gain=1., method="gaussian"))

    def compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def forward(self, inp, mode="a", *args, **kwargs):
        if mode == "a":
            latent, std = self.gen_a.encode(inp)
            out = self.gen_b.decode(latent + std)
        elif mode == "b":
            latent, std = self.gen_b.encode(inp)
            out = self.gen_a.decode(latent + std)
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")
        return out

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        images_a, images_b = self.get_images(batch)

        conditioning = False
        if isinstance(images_a, tuple):
            conditioning = True
            n_classes = self.config.data.n_classes

            seg_a = images_a[1]
            seg_a = torch.stack(
                [(seg_a == label) + torch.rand_like(seg_a) * 0.1 for label in range(n_classes)],
                dim=1
            )
            seg_a /= torch.linalg.norm(seg_a, dim=1, keepdim=True, ord=1)
            seg_a = seg_a.cuda()

            seg_b = np.random.choice(range(n_classes), size=images_a[1].size(),
                                     p=list(self.config.data.class_prevalences.values()))
            seg_b = torch.from_numpy(seg_b).type(torch.float32)
            seg_b = torch.stack(
                [(seg_b == label) + torch.rand_like(seg_b) * 0.1 for label in range(n_classes)],
                dim=1
            )
            seg_b /= torch.linalg.norm(seg_b, dim=1, keepdim=True, ord=1)
            seg_b = seg_b.cuda()

        if optimizer_idx == 0:
            latent_mean_a, latent_std_a = self.gen_a.encode(images_a if not conditioning else torch.cat([images_a[0], seg_a], dim=1))
            latent_mean_b, latent_std_b = self.gen_b.encode(images_b if not conditioning else torch.cat([images_b, seg_b], dim=1))
            # decode (within domain)
            images_a_recon = self.gen_a.decode(latent_mean_a + latent_std_a)
            images_b_recon = self.gen_b.decode(latent_mean_b + latent_std_b)
            # decode (cross domain)
            images_ba = self.gen_a.decode(latent_mean_b + latent_std_b)
            images_ab = self.gen_b.decode(latent_mean_a + latent_std_a)
            # encode again
            latent_mean_ba_recon, latent_std_ba_recon = self.gen_a.encode(images_ba if not conditioning else torch.cat([images_ba, seg_a], dim=1))
            latent_mean_ab_recon, latent_std_ab_recon = self.gen_b.encode(images_ab if not conditioning else torch.cat([images_ab, seg_a], dim=1))
            # decode again (if needed)
            if self.config["recon_x_cyc_w"]:
                images_aba = self.gen_a.decode(latent_mean_ab_recon + latent_std_ab_recon)
                images_bab = self.gen_b.decode(latent_mean_ba_recon + latent_std_ba_recon)
            else:
                images_aba = None
                images_bab = None

            # reconstruction loss
            recon_images_a_loss = self.recon_criterion(images_a_recon, images_a if not conditioning else images_a[0])
            recon_images_b_loss = self.recon_criterion(images_b_recon, images_b)
            loss_gen_recon_kl_a = self.compute_kl(latent_mean_a)
            loss_gen_recon_kl_b = self.compute_kl(latent_mean_b)
            loss_gen_cyc_x_a = self.recon_criterion(images_aba, images_a if not conditioning else images_a[0])
            loss_gen_cyc_x_b = self.recon_criterion(images_bab, images_b)
            loss_gen_recon_kl_cyc_aba = self.compute_kl(latent_mean_ab_recon)
            loss_gen_recon_kl_cyc_bab = self.compute_kl(latent_mean_ba_recon)

            # GAN loss
            gen_a_loss = self.dis_a.calc_gen_loss(images_ba)
            gen_b_loss = self.dis_b.calc_gen_loss(images_ab)

            gen_loss = self.config["gan_w"] * (gen_a_loss + gen_b_loss)
            recon_loss = self.config["recon_x_w"] * (recon_images_a_loss + recon_images_b_loss)
            cc_recon_loss = self.config['recon_x_cyc_w'] * (loss_gen_cyc_x_a + loss_gen_cyc_x_b)
            kl_loss = self.config["recon_kl_w"] * (loss_gen_recon_kl_a + loss_gen_recon_kl_b)
            cc_kl_loss = self.config['recon_kl_cyc_w'] * (loss_gen_recon_kl_cyc_aba + loss_gen_recon_kl_cyc_bab)

            batch_dictionary = {"gen_loss": gen_loss,
                                "recon_loss": recon_loss,
                                "cc_recon_loss": cc_recon_loss,
                                "kl_loss": kl_loss,
                                "cc_kl_loss": cc_kl_loss
                                }

        elif optimizer_idx == 1:
            latent_mean_a, latent_std_a = self.gen_a.encode(images_a if not conditioning else torch.cat([images_a[0], seg_a], dim=1))
            latent_mean_b, latent_std_b = self.gen_b.encode(images_b if not conditioning else torch.cat([images_b, seg_b], dim=1))

            # decode (cross domain)
            images_ba = self.gen_a.decode(latent_mean_b + latent_std_b)
            images_ab = self.gen_b.decode(latent_mean_a + latent_std_a)

            loss_dis_a = self.dis_a.calc_dis_loss(images_ba.detach(), images_a if not conditioning else images_a[0])
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
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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

    def sample_inverted_image(self, image, mode="a", visualize=False):
        pass

    def translate_image(self, image, input_domain="a"):
        n_classes = self.config.data.n_classes
        if isinstance(image, tuple):
            seg = image[1]
            seg = torch.stack(
                [(seg == label) + torch.rand_like(seg) * 0.1 for label in range(n_classes)],
                dim=1
            )
            seg /= torch.linalg.norm(seg, dim=1, keepdim=True, ord=1)
            seg = seg.cuda()
            image = image[0]

        else:
            seg_size = list(image.size())
            seg_size.pop(1)
            seg = np.random.choice(range(n_classes), size=seg_size,
                                   p=list(self.config.data.class_prevalences.values()))
            seg = torch.from_numpy(seg).type(torch.float32)
            seg = torch.stack(
                [(seg == label) + torch.rand_like(seg) * 0.1 for label in range(n_classes)],
                dim=1
            )

            seg /= torch.linalg.norm(seg, dim=1, keepdim=True, ord=1)
            seg = seg.cuda()

        input_image = torch.cat([image, seg], dim=1)

        translated_image = self.forward(input_image, mode=input_domain)
        return translated_image

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        images_a, images_b = self.get_images(batch)

        conditioning = False
        if isinstance(images_a, tuple):
            conditioning = True
            n_classes = self.config.data.n_classes

            seg_a = images_a[1]
            seg_a = torch.stack(
                [(seg_a == label) + torch.rand_like(seg_a) * 0.1 for label in range(n_classes)],
                dim=1
            )
            seg_a /= torch.linalg.norm(seg_a, dim=1, keepdim=True, ord=1)
            seg_a = seg_a.cuda()

            seg_b = np.random.choice(range(n_classes), size=images_a[1].size(),
                                     p=list(self.config.data.class_prevalences.values()))
            seg_b = torch.from_numpy(seg_b).type(torch.float32)
            seg_b = torch.stack(
                [(seg_b == label) + torch.rand_like(seg_b) * 0.1 for label in range(n_classes)],
                dim=1
            )
            seg_b /= torch.linalg.norm(seg_b, dim=1, keepdim=True, ord=1)
            seg_b = seg_b.cuda()

        latent_mean_a, latent_std_a = self.gen_a.encode(
            images_a if not conditioning else torch.cat([images_a[0], seg_a], dim=1))
        latent_mean_b, latent_std_b = self.gen_b.encode(
            images_b if not conditioning else torch.cat([images_b, seg_b], dim=1))
        # decode (within domain)
        images_a_recon = self.gen_a.decode(latent_mean_a + latent_std_a)
        images_b_recon = self.gen_b.decode(latent_mean_b + latent_std_b)
        # decode (cross domain)
        images_ba = self.gen_a.decode(latent_mean_b + latent_std_b)
        images_ab = self.gen_b.decode(latent_mean_a + latent_std_a)
        # encode again
        latent_mean_ba_recon, latent_std_ba_recon = self.gen_a.encode(
            images_ba if not conditioning else torch.cat([images_ba, seg_a], dim=1))
        latent_mean_ab_recon, latent_std_ab_recon = self.gen_b.encode(
            images_ab if not conditioning else torch.cat([images_ab, seg_a], dim=1))

        # decode again (if needed)
        images_aba = self.gen_a.decode(latent_mean_ab_recon + latent_std_ab_recon)
        images_bab = self.gen_b.decode(latent_mean_ba_recon + latent_std_ba_recon)

        images_a = images_a.cpu().numpy() if not conditioning else images_a[0].cpu().numpy()
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
