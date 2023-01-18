import torch.nn as nn
from src.models.basic_blocks import ConvBlock2D, ResBlocks


class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, normalization_type, activation_type, padding_type,
                 dimensionality_reduction: (bool, int) = False):
        super(Encoder, self).__init__()

        if dimensionality_reduction:
            scale_dim = self.reduce_dim
            if isinstance(dimensionality_reduction, int):
                self.scaling_factor = dimensionality_reduction
            else:
                self.scaling_factor = 2
        else:
            scale_dim = self.increase_dim
            self.scaling_factor = 2

        self.model = []
        self.model += [ConvBlock2D(input_dim, dim, kernel_size=7, stride=1, padding=3,
                                   normalization_type=normalization_type, activation_type=activation_type,
                                   padding_type=padding_type)]
        # Downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvBlock2D(dim, scale_dim(dim), kernel_size=4, stride=2, padding=1,
                                       normalization_type=normalization_type, activation_type=activation_type,
                                       padding_type=padding_type)]
            dim = scale_dim(dim)
        # residual blocks
        self.model += [ResBlocks(n_res, dim,
                                 normalization_type=normalization_type, activation_type=activation_type,
                                 padding_type=padding_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

    def reduce_dim(self, dimension):
        return dimension // self.scaling_factor

    def increase_dim(self, dimension):
        return dimension * self.scaling_factor


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, normalization_type='none', activation_type='relu',
                 padding_type='replicate', dimensionality_reduction: (bool, int) = False):
        super(Decoder, self).__init__()

        if dimensionality_reduction:
            scale_dim = self.reduce_dim
            if isinstance(dimensionality_reduction, int):
                self.scaling_factor = dimensionality_reduction
            else:
                self.scaling_factor = 2
        else:
            scale_dim = self.increase_dim
            self.scaling_factor = 2

        self.model = []

        self.model += [ResBlocks(n_res, dim, normalization_type, activation_type, padding_type=padding_type)]
        # Upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           ConvBlock2D(dim, scale_dim(dim), kernel_size=5, stride=1, padding=2,
                                       normalization_type='instance_norm', activation_type=activation_type,
                                       padding_type=padding_type)]
            dim = scale_dim(dim)
        # use reflection padding in the last conv layer
        self.model += [ConvBlock2D(dim, output_dim, kernel_size=7, stride=1, padding=3,
                                   normalization_type='none', activation_type='none', padding_type=padding_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

    def reduce_dim(self, dimension):
        return dimension * self.scaling_factor

    def increase_dim(self, dimension):
        return dimension // self.scaling_factor
