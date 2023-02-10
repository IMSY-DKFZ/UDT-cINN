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

    def get_conditions(self, batch_size, mode: str = "a", segmentation: torch.Tensor = None):
        conditions = list()

        dims = [batch_size, 2]
        condition = torch.zeros(*dims)
        cond_noise = torch.rand(*dims) * 0.1

        if mode == "a":
            condition[:, 0] = 1 - cond_noise[:, 0]
            condition[:, 1] = cond_noise[:, 0]
        elif mode == "b":
            condition[:, 1] = 1 - cond_noise[:, 1]
            condition[:, 0] = cond_noise[:, 1]

        if segmentation is not None:
            one_hot_seg = self.get_label_conditions(segmentation,
                                                    n_labels=self.config.data.n_classes,
                                                    labels_size=condition.size()
                                                    )

            condition = torch.cat((condition, one_hot_seg), dim=1)

        conditions.append(condition.cuda())

        return conditions

    def forward(self, inp, mode="a", *args, **kwargs):
        if self.config.condition == "segmentation":
            segmentation = 0
            if isinstance(inp, tuple):
                segmentation = inp[1].clone()
                return_segmentation = inp[1].clone()
                inp = inp[0]
        else:
            segmentation = None

        conditions = self.get_conditions(inp.size()[0], mode=mode, segmentation=segmentation)
        out, jac = self.model(inp, c=conditions, *args, **kwargs)

        if (isinstance(segmentation, int) and segmentation == 0) or segmentation is None:
            return out, jac
        else:
            return (out, return_segmentation), jac

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

        condition_shape = 2 if self.config.condition != "segmentation" else 2 + self.config.data.n_classes

        for c_block in range(self.conditional_blocks):
            model.append(
                Fm.AllInOneBlock,
                subnet_constructor=self.subnet,
                cond=0,
                cond_shape=(condition_shape,),
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
