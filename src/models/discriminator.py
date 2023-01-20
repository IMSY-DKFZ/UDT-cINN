import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.basic_blocks import ConvBlock2D


class MultiScaleDiscriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, config, input_dim):
        super(MultiScaleDiscriminator, self).__init__()
        self.n_layer = config.n_layer
        self.gan_type = config.gan_type
        self.dim = config.dim
        self.norm = config.normalization
        self.activ = config.activation
        self.num_scales = config.num_scales
        self.pad_type = config.padding_type
        self.dropout = config.dropout
        self.dropout_p = config.dropout_p
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [ConvBlock2D(self.input_dim, dim, 4, 2, 1, normalization_type="none", activation_type=self.activ, padding_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [ConvBlock2D(dim, dim * 2, 4, 2, 1, normalization_type=self.norm, activation_type=self.activ, padding_type=self.pad_type,
                                  dropout=self.dropout, dropout_p=self.dropout_p)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(torch.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == "nsgan":
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class DiscriminatorHSI(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, config, input_dim):
        super(DiscriminatorHSI, self).__init__()
        self.n_layer = config.n_layer
        self.gan_type = config.gan_type

        self.hidden_dim = config.dim
        self.input_dim = input_dim

        # activation type
        if config.activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif config.activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif config.activation == "none":
            self.activation = None
        else:
            raise KeyError(f"Please use a supported activation type: {config.activ}")

        # dropout layer
        if config.dropout:
            self.dropout = nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

        self.model = self.build_model()

    def build_model(self):
        model_list = list()

        model_list.append(nn.Linear(self.input_dim, self.hidden_dim))

        for _ in range(self.n_layer - 2):
            if self.dropout is not None:
                model_list.append(self.dropout)
            if self.activation is not None:
                model_list.append(self.activation)

            model_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        model_list.append(nn.Linear(self.hidden_dim, self.input_dim))

        return nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        loss = 0

        if self.gan_type == "lsgan":
            loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        elif self.gan_type == "nsgan":
            all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all0) +
                               F.binary_cross_entropy(torch.sigmoid(out1), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        out0 = self.forward(input_fake)
        loss = 0
        if self.gan_type == "lsgan":
            loss += torch.mean((out0 - 1)**2) # LSGAN
        elif self.gan_type == "nsgan":
            all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
