import torch
import torch.nn as nn
from src.models.encoder_decoder_hsi import EncoderHSI, DecoderHSI


# class VariationalAutoEncoder(nn.Module):
#     # VAE architecture
#     def __init__(self, config, input_dim):
#         super(VariationalAutoEncoder, self).__init__()
#         dim = config['dim']
#         n_downsample = config['n_downsample']
#         n_res = config['n_res']
#         activ = config['activ']
#         pad_type = config['pad_type']
#         dimension_reduction = config['reduce_dim']
#
#         if "conditional_input_dim" in config:
#             output_dim = input_dim
#             input_dim = config.conditional_input_dim
#         else:
#             output_dim = input_dim
#
#         # content encoder
#         self.enc = Encoder(n_downsample, n_res, input_dim, dim, normalization_type='instance_norm',
#                            activation_type=activ, padding_type=pad_type, dimensionality_reduction=dimension_reduction)
#         self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, output_dim, normalization_type='instance_norm',
#                            activation_type=activ, padding_type=pad_type, dimensionality_reduction=dimension_reduction)
#
#     def forward(self, images):
#         # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
#         hiddens, _ = self.encode(images)
#         if self.training == True:
#             noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
#             images_recon = self.decode(hiddens + noise)
#         else:
#             images_recon = self.decode(hiddens)
#         return images_recon
#
#     def encode(self, images):
#         hiddens = self.enc(images)
#         noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
#         return hiddens, noise
#
#     def decode(self, hiddens):
#         images = self.dec(hiddens)
#         return images


class VariationalAutoencoderHSI(nn.Module):

    def __init__(self, config, input_dim):
        super(VariationalAutoencoderHSI, self).__init__()

        max_dim = config['max_dim']
        min_dim = config['min_dim']
        n_layers = config['n_layers']
        activation_type = config['activ']

        if "conditional_input_dim" in config:
            output_dim = input_dim
            input_dim = config.conditional_input_dim
        else:
            output_dim = input_dim

        self.Encoder = EncoderHSI(n_layers=n_layers, input_dim=input_dim, max_dim=max_dim, min_dim=min_dim,
                                  activation_type=activation_type)
        self.Decoder = DecoderHSI(n_layers=n_layers, output_dim=output_dim, max_dim=max_dim, min_dim=min_dim,
                                  activation_type=activation_type)

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, z):

        return self.Decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 100))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
