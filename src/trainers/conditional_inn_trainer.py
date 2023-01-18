import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from src.trainers import DAInnBase
import torch
from src.models.inn_subnets import subnet_conv, subnet_res_net
from src.utils.dimensionality_calculations import calculate_downscale_dimensionality
from src.models.multiscale_invertible_blocks import append_multi_scale_inn_blocks


class CondinitionalDomainAdaptationINN(DAInnBase):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.conditions = self.get_conditions()
        self.model = self.build_model()

    def get_conditions(self):
        conditions = dict()
        conditions["a"] = list()
        conditions["b"] = list()
        for level in range(self.config.downsampling_levels):
            dims = calculate_downscale_dimensionality(self.config.data.dimensions,
                                                      2 ** (level + self.config.instant_downsampling))
            dims.insert(0, self.config.batch_size)
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
            conditions = self.conditions["a"]
        elif mode == "b":
            conditions = self.conditions["b"]
        else:
            raise AttributeError("Specify either mode 'a' or 'b'!")

        out, jac = self.model(inp, c=conditions, *args, **kwargs)

        return out, jac

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.inn_training_step(batch)

    def configure_optimizers(self):
        return self.get_inn_optimizer()

    def build_model(self):
        model = Ff.SequenceINN(*self.dimensions)
        append_multi_scale_inn_blocks(
            model,
            num_scales=self.config.downsampling_levels,
            blocks_per_scale=[
                self.config.high_res_conv,
                self.config.middle_res_conv,
                self.config.low_res_conv
            ],
            dimensions=self.dimensions,
            subnets=[subnet_conv, subnet_conv, subnet_res_net],
            conditional_blocks=True,
            clamping=self.config.clamping,
            instant_downsampling=self.config.instant_downsampling
        )

        append_multi_scale_inn_blocks(
            model,
            num_scales=1,
            blocks_per_scale=self.config.low_res_conv,
            dimensions=self.dimensions,
            subnets=subnet_res_net,
            clamping=self.config.clamping
        )

        model.append(Fm.Flatten)

        return model
