import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from domain_adaptation.models.domain_adaptation_inn_base import DomainAdaptationInn
from abc import ABC, abstractmethod
from typing import overload
import torch
import torch.nn.functional as F
from domain_adaptation.models.discriminator import MultiScaleDiscriminator
from domain_adaptation.models.inn_subnets import *


class GANDomainAdaptationInn(DomainAdaptationInn, ABC):
    def __init__(self, config: DictConfig):
        super().__init__(config=config)

        self.discriminator_a, self.discriminator_b = self.build_discriminators()
        self.enc_dec_a, self.enc_dec_b, self.shared_blocks = self.inn_model

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
        if mode == "a":
            encoder_decoder = self.enc_dec_a
        elif mode == "b":
            encoder_decoder = self.enc_dec_b
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")

        if kwargs.get("rev"):
            tmp_encoded, tmp_jac = self.shared_blocks(inp, *args, **kwargs)
            out, jac = encoder_decoder(tmp_encoded, *args, **kwargs)
        else:
            tmp_encoded, tmp_jac = encoder_decoder(inp, *args, **kwargs)
            out, jac = self.shared_blocks(tmp_encoded, *args, **kwargs)
        return out, jac + tmp_jac

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        images_a, images_b = self.get_images(batch)

        if optimizer_idx == 0:
            z_a, jac_a = self.forward(images_a, mode="a")
            z_b, jac_b = self.forward(images_b, mode="b")

            ml_loss = self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)
            batch_dictionary = {"ml_loss": ml_loss.detach()}

            images_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            images_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            # valid_a = torch.ones(images_a.size(0), 1)
            # valid_a = valid_a.type_as(images_a)
            #
            # valid_b = torch.ones(images_b.size(0), 1)
            # valid_b = valid_b.type_as(images_b)

            gen_a_loss = self.discriminator_a.calc_gen_loss(images_ba)
            gen_b_loss = self.discriminator_b.calc_gen_loss(images_ab)
            # gen_a_loss = self.adversarial_loss(self.discriminator_a(images_ba), valid_a)
            # gen_b_loss = self.adversarial_loss(self.discriminator_b(images_ab), valid_b)
            gen_loss = gen_a_loss + gen_b_loss
            gen_loss = torch.log(gen_loss)
            gen_loss *= self.config.gan_weight
            batch_dictionary["gen_loss"] = gen_loss.detach()

            total_loss = ml_loss + gen_loss

            if self.config.spectral_consistency:
                spectral_consistency_loss = self.spectral_consistency_loss(images_a,
                                                                           images_b,
                                                                           images_ab,
                                                                           images_ba)

                batch_dictionary["sc_loss"] = spectral_consistency_loss
                total_loss = ml_loss + gen_loss + spectral_consistency_loss

        elif optimizer_idx == 1:
            z_a, jac_a = self.forward(images_a, mode="a")
            z_b, jac_b = self.forward(images_b, mode="b")

            ml_loss = self.maximum_likelihood_loss(z_a=z_a, jac_a=jac_a, z_b=z_b, jac_b=jac_b)
            batch_dictionary = {"ml_loss": ml_loss.detach()}

            images_ab, _ = self.forward(z_a, mode="b", rev=True, jac=False)
            images_ba, _ = self.forward(z_b, mode="a", rev=True, jac=False)

            dis_a_loss = self.discriminator_a.calc_dis_loss(images_ba, images_a)
            dis_b_loss = self.discriminator_b.calc_dis_loss(images_ab, images_b)
            # valid_a = torch.ones(images_a.size(0), 1)
            # valid_a = valid_a.type_as(images_a)
            #
            # valid_b = torch.ones(images_b.size(0), 1)
            # valid_b = valid_b.type_as(images_b)
            #
            # real_loss_a = self.adversarial_loss(self.discriminator_a(images_a), valid_a)
            # real_loss_b = self.adversarial_loss(self.discriminator_b(images_b), valid_b)
            #
            # # how well can it label as fake?
            # fake_a = torch.zeros(images_a.size(0), 1)
            # fake_b = fake_a.type_as(images_a)
            #
            # fake_loss_a = self.adversarial_loss(self.discriminator_a(images_ba.detach()), fake_a)
            # fake_loss_b = self.adversarial_loss(self.discriminator_b(images_ab.detach()), fake_b)
            #
            # # discriminator loss is the average of these
            # dis_a_loss = (real_loss_a + fake_loss_a) / 2
            # dis_b_loss = (real_loss_b + fake_loss_b) / 2
            dis_loss = dis_a_loss + dis_b_loss
            dis_loss = torch.log(dis_loss)
            dis_loss *= self.config.gan_weight
            batch_dictionary["dis_loss"] = dis_loss.detach()

            total_loss = ml_loss + dis_loss

        else:
            raise IndexError("There are more optimizers than specified!")

        batch_dictionary["loss"] = total_loss
        self.log_losses(batch_dictionary)
        return batch_dictionary

    @overload
    @abstractmethod
    def build_discriminators(self):
        pass

    def build_discriminators(self):
        return MultiScaleDiscriminator(self.config.dis, self.channels), \
               MultiScaleDiscriminator(self.config.dis, self.channels)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        inn_optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(inn_optimizer,
                                                               factor=0.2,
                                                               patience=8,
                                                               threshold=0.005,
                                                               cooldown=2)

        dis_params = list(self.discriminator_a.parameters()) + list(self.discriminator_a.parameters())
        dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=self.config.learning_rate)

        return [inn_optimizer, dis_optimizer], [{"scheduler": scheduler, "monitor": "loss_epoch"}]

    def build_model(self):
        return self.build_enc_dec(), self.build_enc_dec(), self.build_shared_blocks()

    def build_enc_dec(self):
        downsampling_block = Fm.IRevNetDownsampling
        if self.config.downsampling_type == "irevnet":
            downsampling_block = Fm.IRevNetDownsampling
        elif self.config.downsampling_type == "haar":
            downsampling_block = Fm.HaarDownsampling

        nodes = Ff.SequenceINN(*self.config.data.dimensions)

        # Higher resolution convolutional part
        for k in range(self.config.high_res_conv):
            nodes = self.append_all_in_one_block(nodes, sub_network=subnet_conv)

        nodes.append(downsampling_block)

        # middle resolution conv part
        for k in range(self.config.middle_res_conv):
            if k % 2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv

            nodes = self.append_all_in_one_block(nodes, sub_network=subnet)

        nodes.append(downsampling_block)

        return nodes

    def build_shared_blocks(self, input_shape=None):
        # # Lower resolution convolutional part
        nodes = Ff.SequenceINN(256, 64, 32)

        for k in range(self.config.low_res_conv):
            nodes = self.append_all_in_one_block(nodes, sub_network=subnet_res_net)

        # nodes.append(Fm.Flatten)
        return nodes
