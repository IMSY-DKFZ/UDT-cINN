import torch.nn as nn
from src.models.basic_blocks import ResBlocks


def weight_init(m, gain=1., method="kaiming"):
    if isinstance(m, nn.Conv2d):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight.data)
        elif method == "gaussian":
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.zero_().add_(gain)
        m.bias.data.zero_()
    if isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight.data)
        elif method == "gaussian":
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.zero_()


def subnet_res_net(c_in, c_out, n_res=1, kernel_size=(3, 3)) -> nn.Module:
    net = nn.Sequential(nn.Conv2d(c_in, 256,   kernel_size, padding=1), nn.ReLU(),
                        ResBlocks(num_blocks=n_res, dim=256),
                        nn.Conv2d(256, c_out, kernel_size, padding=1))
    net.apply(lambda m: weight_init(m, gain=1.))
    return net


def subnet_res_net_adaptive(c_in, c_out, n_res=1, kernel_size=(3, 3)) -> nn.Module:
    net = nn.Sequential(nn.Conv2d(c_in, c_in,   kernel_size, padding=1), nn.ReLU(),
                        ResBlocks(num_blocks=n_res, dim=c_in),
                        nn.Conv2d(c_in, c_out, kernel_size, padding=1))
    net.apply(lambda m: weight_init(m, gain=1.))
    return net


def subnet_conv(c_in, c_out, kernel_size=(3, 3)) -> nn.Module:
    net = nn.Sequential(nn.Conv2d(c_in, 256,   kernel_size, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  256, kernel_size, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, kernel_size, padding=1))
    net.apply(lambda m: weight_init(m, gain=1.))
    return net


def subnet_conv_adaptive(c_in, c_out, kernel_size=(3, 3)) -> nn.Module:
    multiplicator = 4 if c_in >= 256 else 8
    net = nn.Sequential(nn.Conv2d(c_in, multiplicator*c_in,   kernel_size, padding=1), nn.ReLU(),
                        nn.Conv2d(multiplicator*c_in,  multiplicator*c_in, kernel_size, padding=1), nn.ReLU(),
                        nn.Conv2d(multiplicator*c_in,  c_out, kernel_size, padding=1))
    net.apply(lambda m: weight_init(m, gain=1.))
    return net


def subnet_conv_1x1(c_in, c_out, kernel_size=(1, 1)) -> nn.Module:
    net = nn.Sequential(nn.Conv2d(c_in, 256,   kernel_size), nn.ReLU(),
                        nn.Conv2d(256,  c_out, kernel_size))
    net.apply(lambda m: weight_init(m, gain=1.))
    return net


def cond_net_fc(c_in, c_out) -> nn.Module:
    return nn.Sequential(nn.Linear(c_in, 16), nn.ReLU(),
                         nn.Linear(16, c_out))
