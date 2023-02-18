import torch.nn as nn
from src.models.basic_blocks import ConvBlock2D, ResBlocks


class EncoderHSI(nn.Module):
    def __init__(self, n_layers, input_dim, max_dim, min_dim, activation_type):
        super(EncoderHSI, self).__init__()

        # activation type
        if activation_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == "none":
            self.activation = None
        else:
            raise KeyError(f"Please use a supported activation type: {activation_type}")

        self.model = list()
        self.model += [nn.Linear(input_dim, max_dim)]

        # Exctration blocks
        for layer in range(n_layers - 1):
            self.model += [nn.Linear(max_dim, max_dim)]

        self.model = nn.Sequential(*self.model)

        # Mu and Std layers
        self.fc_mu = nn.Linear(max_dim, min_dim)
        self.fc_sig = nn.Linear(max_dim, min_dim)

    def forward(self, x):
        tmp_output = self.activation(self.model(x))
        mu_output = self.fc_mu(tmp_output)
        output_logvar = self.fc_sig(tmp_output)
        return mu_output, output_logvar


class DecoderHSI(nn.Module):
    def __init__(self, n_layers, output_dim, max_dim, min_dim, activation_type):
        super(DecoderHSI, self).__init__()

        # activation type
        if activation_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == "none":
            self.activation = None
        else:
            raise KeyError(f"Please use a supported activation type: {activation_type}")

        self.model = list()
        self.model += [nn.Linear(min_dim, max_dim)]

        # Exctration blocks
        for layer in range(n_layers - 2):
            self.model += [nn.Linear(max_dim, max_dim)]

        self.model += [nn.Linear(max_dim, output_dim)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
