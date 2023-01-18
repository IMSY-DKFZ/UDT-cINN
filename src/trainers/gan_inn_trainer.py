import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from src.trainers import DAInnBase
import torch
import torch.nn.functional as F
from src.models.discriminator import MultiScaleDiscriminator
from src.models.multiscale_invertible_blocks import append_multi_scale_inn_blocks
from src.models.inn_subnets import subnet_conv
from src.utils.dimensionality_calculations import calculate_downscale_dimensionality


class GANDomainAdaptationInn(DAInnBase):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.discriminator_a = MultiScaleDiscriminator(self.config.dis, self.channels)
        self.discriminator_b = MultiScaleDiscriminator(self.config.dis, self.channels)
        self.enc_dec_a = self.build_enc_dec()
        self.enc_dec_b = self.build_enc_dec()
        self.shared_blocks = self.build_shared_blocks()

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
        return self.gan_inn_training_step(batch, optimizer_idx)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        inn_optimizer, scheduler_list = self.get_inn_optimizer()

        dis_params = list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters())
        dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=self.config.learning_rate)

        return [*inn_optimizer, dis_optimizer], scheduler_list

    def build_enc_dec(self):
        model = Ff.SequenceINN(*self.dimensions)
        append_multi_scale_inn_blocks(
            model,
            num_scales=self.config.downsampling_levels - 1,
            blocks_per_scale=[
                self.config.high_res_conv,
                self.config.middle_res_conv,
            ],
            dimensions=self.dimensions,
            subnets=subnet_conv,
            clamping=self.config.clamping,
            instant_downsampling=self.config.instant_downsampling
        )

        model.append(Fm.HaarDownsampling)

        return model

    def build_shared_blocks(self):
        # # Lower resolution convolutional part
        if self.config.instant_downsampling:
            dimension_calc_correction = 0
        else:
            dimension_calc_correction = 1
        dims = calculate_downscale_dimensionality(self.dimensions,
                                                  2**(self.config.downsampling_levels - dimension_calc_correction))
        model = Ff.SequenceINN(*dims)

        append_multi_scale_inn_blocks(
            model,
            num_scales=1,
            blocks_per_scale=self.config.low_res_conv,
            dimensions=self.dimensions,
            subnets=subnet_conv,
            clamping=self.config.clamping
        )

        model.append(Fm.Flatten)
        return model
