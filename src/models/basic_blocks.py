import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, normalization_type="none", activation_type="relu", padding_type="replicate",
                 dropout=False, dropout_p=0.2):
        super(ConvBlock2D, self).__init__()

        if padding > 0:
            self.padding_bool = True
            # Padding type
            if padding_type == "reflect":
                self.padding = nn.ReflectionPad2d(padding)
            elif padding_type == "replicate":
                self.padding = nn.ReplicationPad2d(padding)
            else:
                raise KeyError(f"Please use a supported padding type: {padding_type}")

        # Normalization type
        if normalization_type == "batch_norm":
            self.normalization = nn.BatchNorm2d(output_dim)
        elif normalization_type == "instance_norm":
            self.normalization = nn.InstanceNorm2d(output_dim)
        elif normalization_type == 'layer_norm':
            self.normalization = nn.LayerNorm(output_dim)
        elif normalization_type == "none":
            self.normalization = None
        else:
            raise KeyError(f"Please use a supported normalization type: {normalization_type}")

        # activation type
        if activation_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == "none":
            self.activation = None
        else:
            raise KeyError(f"Please use a supported activation type: {activation_type}")

        if dropout:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None

        # Convolution
        self.convolution = nn.Conv2d(input_dim, output_dim, kernel_size, stride)

    def forward(self, x):
        if self.padding_bool:
            x = self.padding(x)
        x = self.convolution(x)
        if self.normalization:
            x = self.normalization(x)
        if self.dropout:
            x = self.dropout(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim,
                 normalization_type="none", activation_type="relu", padding_type="replicate"):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBlock2D(dim, dim, kernel_size=3, stride=1, padding=1,
                              normalization_type=normalization_type, activation_type=activation_type,
                              padding_type=padding_type)]
        model += [ConvBlock2D(dim, dim, kernel_size=3, stride=1, padding=1,
                              normalization_type=normalization_type, activation_type="none",
                              padding_type=padding_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim,
                 normalization_type="none", activation_type="relu", padding_type="replicate"):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    normalization_type=normalization_type, activation_type=activation_type,
                                    padding_type=padding_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
