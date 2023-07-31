from omegaconf import DictConfig
from src.trainers import DAInnBase
import torch
import numpy as np
from src.models.align_flow_model import Flow2Flow, chain, get_param_groups, get_lr_scheduler
from src.visualization import col_bar
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


class AlignFlowTrainerHSI(DAInnBase):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.model = Flow2Flow(self.config)

    def forward(self, inp, mode="a", *args, **kwargs):

        if mode == "a":
            out, jac = self.model.g_src(inp, *args, **kwargs)
        else:
            out, jac = self.model.g_tgt(inp, *args, **kwargs)

        return out, jac

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        images_a, images_b = self.get_images(batch)
        if images_b.size()[0] != self.config.batch_size:
            print("Skipped batch because of uneven data_sizes")
            return None

        if optimizer_idx == 0:
            batch_dictionary = self.model.backward_g(images_a, images_b)

        elif optimizer_idx == 1:
            batch_dictionary = self.model.backward_d(images_a, images_b)

        else:
            raise IndexError("There are more optimizers than specified!")

        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary)
        self.log_losses(batch_dictionary)
        return batch_dictionary

    def configure_optimizers(self):
        g_src_params = get_param_groups(self.model.g_src, self.config.weight_norm_l2, norm_suffix='weight_g')
        g_tgt_params = get_param_groups(self.model.g_tgt, self.config.weight_norm_l2, norm_suffix='weight_g')
        opt_g = torch.optim.Adam(chain(g_src_params, g_tgt_params),
                                 lr=self.config.rnvp_lr,
                                 betas=(self.config.rnvp_beta_1, self.config.rnvp_beta_2))
        opt_d = torch.optim.Adam(chain(self.model.d_tgt.parameters(), self.model.d_src.parameters()),
                                 lr=self.config.lr,
                                 betas=(self.config.beta_1, self.config.beta_2))

        return [opt_g, opt_d], [get_lr_scheduler(opt, self.config) for opt in [opt_g, opt_d]]

    def validation_step(self, batch, batch_idx):
        plt.figure(figsize=(20, 5))
        self.model.src, self.model.tgt = self.get_images(batch)

        if self.model.clamp_jacobian:
            # Double batch size with perturbed inputs for Jacobian Clamping
            self.model._jc_preprocess()

        # Forward src -> lat: Get MLE loss
        self.model.src2lat, sldj_src2lat = self.model.g_src(self.model.src, rev=False)
        self.model.loss_mle_src = self.model.lambda_mle * self.model.mle_loss_fn(self.model.src2lat, sldj_src2lat)

        # Finish src -> lat -> tgt: Say target is real to invert loss
        self.model.src2tgt, _ = self.model.g_tgt(self.model.src2lat, rev=True)
        # self.src2tgt = torch.tanh(src2tgt)

        # Forward tgt -> lat: Get MLE loss
        self.model.tgt2lat, sldj_tgt2lat = self.model.g_tgt(self.model.tgt, rev=False)
        self.model.loss_mle_tgt = self.model.lambda_mle * self.model.mle_loss_fn(self.model.tgt2lat, sldj_tgt2lat)

        # Finish tgt -> lat -> src: Say source is real to invert loss
        self.model.tgt2src, _ = self.model.g_src(self.model.tgt2lat, rev=True)
        # self.tgt2src = torch.tanh(tgt2src)

        # Jacobian Clamping loss
        if self.model.clamp_jacobian:
            # Split inputs and outputs from Jacobian Clamping
            self.model._jc_postprocess()
            self.model.loss_jc_src = self.model.jc_loss_fn(self.model.src2tgt, self.model.src2tgt_jc, self.model.src, self.model.src_jc)
            self.model.loss_jc_tgt = self.model.jc_loss_fn(self.model.tgt2src, self.model.tgt2src_jc, self.model.tgt, self.model.tgt_jc)
            self.model.loss_jc = self.model.loss_jc_src + self.model.loss_jc_tgt
        else:
            self.model.loss_jc_src = self.model.loss_jc_tgt = self.model.loss_jc = 0.

        # GAN loss
        self.model.loss_gan_src = self.model.gan_loss_fn(self.model.d_tgt(self.model.src2tgt), is_tgt_real=True)
        self.model.loss_gan_tgt = self.model.gan_loss_fn(self.model.d_src(self.model.tgt2src), is_tgt_real=True)

        # Total losses
        self.model.loss_gan = self.model.loss_gan_src + self.model.loss_gan_tgt
        self.model.loss_mle = self.model.loss_mle_src + self.model.loss_mle_tgt

        batch_dictionary = {"val_gan_loss": self.model.loss_gan, "val_mle_loss": self.model.loss_mle, "val_jac_loss": self.model.loss_jc}

        src2tgt = self.model.src2tgt_buffer.sample(self.model.src2tgt)
        self.model.loss_d_tgt = self.model._forward_d(self.model.d_tgt, self.model.tgt, src2tgt)

        # Forward src discriminator
        tgt2src = self.model.tgt2src_buffer.sample(self.model.tgt2src)
        self.model.loss_d_src = self.model._forward_d(self.model.d_src, self.model.src, tgt2src)

        # Backprop
        batch_dictionary["dis_loss"] = self.model.loss_d_tgt + self.model.loss_d_src
        batch_dictionary = self.aggregate_total_loss(losses_dict=batch_dictionary, val_run=True)
        self.log_losses(batch_dictionary)

        if batch_idx == 9:
            images_a = self.model.src[0].cpu().numpy()
            images_b = self.model.tgt[0].cpu().numpy()
            images_ab = self.model.src2tgt[0].cpu().numpy()
            images_ba = self.model.tgt2src[0].cpu().numpy()
            z_a = self.model.src2lat[0].cpu().numpy()
            z_b = self.model.tgt2lat[0].cpu().numpy()
            z_a_flat = z_a.flatten()
            z_b_flat = z_b.flatten()
            mean_z_a, std_z_a = norm.fit(z_a_flat)
            mean_z_b, std_z_b = norm.fit(z_b_flat)
            x_space = np.linspace(-3, 3, 500)
            y_z_a = norm.pdf(x_space, mean_z_a, std_z_a)
            y_z_b = norm.pdf(x_space, mean_z_b, std_z_b)
            plt.subplot(2, 4, 1)
            plt.title("Simulated Image")
            img_a = plt.imshow(images_a[0, :, :])
            col_bar(img_a)
            plt.subplot(2, 3, 2)
            plt.title("Latent Sim Dist")
            plt.hist(z_a_flat, density=True, bins=50)
            plt.plot(x_space, y_z_a, label="mean={:1.2f}, std={:1.2f}".format(mean_z_a, std_z_a))
            plt.legend()
            plt.subplot(2, 3, 3)
            plt.title("Simulation to Real Image")
            img_ab = plt.imshow(images_ab[0, :, :])
            col_bar(img_ab)
            plt.subplot(2, 3, 4)
            plt.title("Real Image")
            img_b = plt.imshow(images_b[0, :, :])
            col_bar(img_b)
            plt.subplot(2, 3, 5)
            plt.title("Latent Real Dist")
            plt.hist(z_b_flat, density=True, bins=50)
            plt.plot(x_space, y_z_b, label="mean={:1.2f}, std={:1.2f}".format(mean_z_b, std_z_b))
            plt.legend()
            plt.subplot(2, 3, 6)
            plt.title("Real to Simulated Image")
            img_ba = plt.imshow(images_ba[0, :, :])
            col_bar(img_ba)
            plt.savefig(os.path.join(self.config.save_path, f"val_im_{self.current_epoch}.png"))
            plt.close()

    # def test_step(self, batch, batch_idx, *args, **kwargs):
    #     path = os.path.join(self.config.save_path, "testing")
    #     generated_image_data_path = os.path.join(path, "generated_image_data")
    #     os.makedirs(generated_image_data_path, exist_ok=True)
    #
    #     images_a, images_b = self.get_images(batch)
    #
    #     if len(images_a) == 5 or len(images_a) == 5:
    #         images_a, images_b = torch.squeeze(images_a, dim=0), torch.squeeze(images_b, dim=0)
    #
    #
    #
    #     images_ab = self.translate_image(images_a, input_domain="a")
    #     images_ba = self.translate_image(images_b, input_domain="b")
    #
    #     images_a = images_a[0].cpu().numpy() if isinstance(images_a, tuple) else images_a.cpu().numpy()
    #     images_b = images_b[0].cpu().numpy() if isinstance(images_b, tuple) else images_b.cpu().numpy()
    #     images_ab = images_ab[0].cpu().numpy() if isinstance(images_ab, tuple) else images_ab.cpu().numpy()
    #     images_ba = images_ba[0].cpu().numpy() if isinstance(images_ba, tuple) else images_ba.cpu().numpy()
    #
    #     if self.config.normalization not in ["None", "none"]:
    #         if self.config.normalization == "standardize":
    #             images_a = images_a * self.config.data.std_a + self.config.data.mean_a
    #             images_ba = images_ba * self.config.data.std_a + self.config.data.mean_a
    #
    #             images_b = images_b * self.config.data.std_b + self.config.data.mean_b
    #             images_ab = images_ab * self.config.data.std_b + self.config.data.mean_b
    #
    #     np.savez(os.path.join(generated_image_data_path, f"test_batch_{batch_idx}"),
    #              images_a=images_a,
    #              images_b=images_b,
    #              images_ab=images_ab,
    #              images_ba=images_ba,
    #              seg_a=batch["seg_a"],
    #              seg_b=batch["seg_b"],
    #              oxy_a=batch["oxy_a"],
    #              oxy_b=batch["oxy_b"],
    #              )
    #
    #     if True:
    #         generated_images_path = os.path.join(path, "generated_images")
    #         os.makedirs(generated_images_path, exist_ok=True)
    #         plt.figure(figsize=(6, 6))
    #         plt.subplot(2, 2, 1)
    #         plt.title("Domain A")
    #         img_a = plt.imshow(images_a[0, 0, :, :])
    #         col_bar(img_a)
    #         plt.subplot(2, 2, 2)
    #         plt.title("Domain A to Domain B")
    #         img_ab = plt.imshow(images_ab[0, 0, :, :])
    #         col_bar(img_ab)
    #         plt.subplot(2, 2, 3)
    #         plt.title("Domain B")
    #         img_b = plt.imshow(images_b[0, 0, :, :])
    #         col_bar(img_b)
    #         plt.subplot(2, 2, 4)
    #         plt.title("Domain B to Domain A")
    #         img_ba = plt.imshow(images_ba[0, 0, :, :])
    #         col_bar(img_ba)
    #         plt.savefig(os.path.join(generated_images_path, f"test_batch_{batch_idx}.png"))
    #         plt.close()