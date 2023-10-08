import torch
import torch.nn as nn
from torch.autograd import Variable
from src.models.encoder_decoder import Encoder, Decoder


class VariationalAutoEncoder(nn.Module):
    # VAE architecture
    def __init__(self, config, input_dim):
        super(VariationalAutoEncoder, self).__init__()
        dim = config['dim']
        n_downsample = config['n_downsample']
        n_res = config['n_res']
        activ = config['activ']
        pad_type = config['pad_type']
        dimension_reduction = config['reduce_dim']

        if "conditional_input_dim" in config:
            output_dim = input_dim
            input_dim = config.conditional_input_dim
        else:
            output_dim = input_dim

        # content encoder
        self.enc = Encoder(n_downsample, n_res, input_dim, dim, normalization_type='instance_norm',
                           activation_type=activ, padding_type=pad_type, dimensionality_reduction=dimension_reduction)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, output_dim, normalization_type='instance_norm',
                           activation_type=activ, padding_type=pad_type, dimensionality_reduction=dimension_reduction)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens, std = self.encode(images)
        # if self.training == True:
        #     noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        #     images_recon = self.decode(hiddens + noise)
        # else:
        images_recon = self.decode(hiddens + std)
        return images_recon

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


if __name__ == "__main__":
    from collections import OrderedDict
    import numpy as np
    import matplotlib.pyplot as plt

    config = {
        "dim": 64,
        "activ": "relu",
        "n_downsample": 2,
        "n_res": 4,
        "pad_type": "reflect",
        "reduce_dim": 4
    }

    vae = VariationalAutoEncoder(config, 16).cuda()
    vae_checkpoint = torch.load("")["state_dict"]
    # IMPORTANT: assumes that vae weights are stored in model.weights...
    new_state_dict = OrderedDict()
    for key, value in vae_checkpoint.items():
        new_state_dict[key[6:]] = value

    vae.load_state_dict(new_state_dict)

    batch_size = 16

    inp = torch.randn([batch_size, 4, 32, 64]).cuda()

    out = vae.decode(inp)
    image = out.cpu().detach().numpy()

    print(np.mean(image),
          np.std(image))

    for i in range(batch_size):
        plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i + 1)
        plt.imshow(image[i, 0, :, :])
    plt.show()

