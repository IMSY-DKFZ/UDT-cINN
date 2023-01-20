import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig,ListConfig
from src.trainers import DAInnBase
import torch
from src.models.discriminator import MultiScaleDiscriminator
from src.models.inn_subnets import subnet_conv, subnet_res_net, subnet_conv_adaptive, subnet_res_net_adaptive
from src.utils.dimensionality_calculations import calculate_downscale_dimensionality
from src.models.multiscale_invertible_blocks import append_multi_scale_inn_blocks, append_all_in_one_block


class GanCondinitionalDomainAdaptationINN(DAInnBase):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.model = self.build_model()

        self.condition_shapes = self.get_condition_shapes()

        self.discriminator_a = MultiScaleDiscriminator(self.config.dis, self.channels)
        self.discriminator_b = MultiScaleDiscriminator(self.config.dis, self.channels)

    def get_condition_shapes(self):
        condition_shapes = list()

        model_conditions = self.model.conditions
        unique_conditions = set(model_conditions)
        unique_conditions.remove(None)
        unique_conditions = sorted(unique_conditions)

        block_shapes = self.model.shapes

        for unique_condition in unique_conditions:
            cond_idx = model_conditions.index(unique_condition)
            condition_shapes.append(list(block_shapes[cond_idx]))

        return condition_shapes

    def get_conditions(self, batch_size):
        conditions = dict()
        conditions["a"] = list()
        conditions["b"] = list()

        for condition_shape in self.condition_shapes:
            dims = condition_shape.copy()

            dims.insert(0, batch_size)
            dims[1] = 2
            condition_a = torch.zeros(*dims)
            cond_noise = torch.rand(*dims) * 0.1
            condition_a[:, 0, :, :] = 1 - cond_noise[:, 0, :, :]
            condition_a[:, 1, :, :] = cond_noise[:, 0, :, :]
            condition_b = torch.zeros(*dims)
            condition_b[:, 1, :, :] = 1 - cond_noise[:, 1, :, :]
            condition_b[:, 0, :, :] = cond_noise[:, 1, :, :]
            conditions["a"].append(condition_a.cuda())
            conditions["b"].append(condition_b.cuda())
        return conditions

    def forward(self, inp, mode="a", *args, **kwargs):

        if mode == "a":
            conditions = self.get_conditions(inp.size()[0])["a"]
        elif mode == "b":
            conditions = self.get_conditions(inp.size()[0])["b"]
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")

        out, jac = self.model(inp, c=conditions, *args, **kwargs)

        return out, jac

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        return self.gan_inn_training_step(batch, optimizer_idx)

    def configure_optimizers(self):
        inn_optimizer, scheduler_list = self.get_inn_optimizer()

        dis_params = list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters())
        dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=self.config.learning_rate)

        return [*inn_optimizer, dis_optimizer], scheduler_list

    def build_model(self):
        model = Ff.SequenceINN(*self.dimensions)
        k = self.config.instant_downsampling

        # if self.config.instant_downsampling and ((isinstance(self.config.data.used_channels, (list, ListConfig))
        #                                           and len(self.config.data.used_channels) >= 1)
        #                                          or self.config.data.used_channels >= 16):
        #     append_multi_scale_inn_blocks(
        #         model,
        #         num_scales=1,
        #         blocks_per_scale=1,
        #         dimensions=self.dimensions,
        #         subnets=subnet_conv,
        #         conditional_blocks=False,
        #         clamping=self.config.clamping,
        #         instant_downsampling=False
        #     )

        append_multi_scale_inn_blocks(
            model,
            num_scales=self.config.downsampling_levels,
            blocks_per_scale=[
                4,
                self.config.high_res_conv,
                self.config.middle_res_conv,
                self.config.low_res_conv,
                2#self.config.low_res_conv
            ],
            dimensions=self.dimensions,
            subnets=[subnet_conv, subnet_conv, subnet_conv_adaptive, subnet_conv_adaptive, subnet_conv],
            conditional_blocks=True,
            clamping=self.config.clamping,
            instant_downsampling=self.config.instant_downsampling,
            varying_kernel_size=False
        )

        # append_multi_scale_inn_blocks(
        #     model,
        #     num_scales=1,
        #     blocks_per_scale=self.config.low_res_conv,
        #     dimensions=self.dimensions,
        #     subnets=subnet_res_net_adaptive,
        #     clamping=self.config.clamping
        # )

        model.append(Fm.Flatten)

        return model
