import FrEIA.framework as Ff
import FrEIA.modules as Fm
from omegaconf import DictConfig
from src.trainers import DAInnBaseHSI
import torch
import torch.nn as nn
from src.models.discriminator import DiscriminatorHSI
from src.models.inn_subnets import weight_init


class GanCondinitionalDomainAdaptationINNHSI(DAInnBaseHSI):
    def __init__(self, experiment_config: DictConfig):
        super().__init__(experiment_config=experiment_config)

        self.n_blocks = self.config.n_blocks
        self.conditional_blocks = self.config.n_conditional_blocks

        self.model = self.build_model()

        self.discriminator_a = DiscriminatorHSI(self.config.dis, self.dimensions)
        self.discriminator_b = DiscriminatorHSI(self.config.dis, self.dimensions)

    def get_conditions(self, batch_size):
        conditions = dict()
        conditions["a"] = list()
        conditions["b"] = list()

        dims = self.dimensions
        dims = [batch_size, dims]

        for conditional_block in range(self.conditional_blocks):
            cond_noise_a = torch.rand(*dims) * 0.1
            condition_a = 1 - cond_noise_a

            cond_noise_b = torch.rand(*dims) * 0.1
            condition_b = 0 + cond_noise_b

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

    def subnet(self, ch_in, ch_out):
        net = nn.Sequential(
            nn.Linear(ch_in, self.config.n_hidden),
            nn.ReLU(),
            nn.Linear(self.config.n_hidden, ch_out),
        )

        net.apply(lambda m: weight_init(m, gain=1.))
        return net

    def build_model(self):
        model = Ff.SequenceINN(self.dimensions)

        n_shared_blocks = int(self.n_blocks - self.conditional_blocks)

        for c_block in range(self.conditional_blocks):
            model.append(
                Fm.AllInOneBlock,
                subnet_constructor=self.subnet,
                cond=c_block,
                cond_shape=(self.dimensions,),
                affine_clamping=self.config.clamping,
                global_affine_init=self.config.actnorm,
                permute_soft=False,
                learned_householder_permutation=self.config.n_reflections,
                reverse_permutation=True,
            )

        for _ in range(n_shared_blocks):
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
