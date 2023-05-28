"""
This code has been taken from https://github.com/ermongroup/alignflow in order to reproduce their model.
Their code has been adapted in order to fit into the pytorch lightning framework used in this main repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler as torch_scheduler
import numpy as np
import functools
from enum import IntEnum
import random
from itertools import chain


def un_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Reverse normalization of an image. Move to CPU.

    Args:
        tensor (torch.Tensor): Tensor with pixel values in range (-1, 1).
        mean (tuple): Mean per channel.
        std (tuple): Standard deviation per channel.

    Returns:
        tensor: Un-normalized image as PyTorch Tensor in range [0, 255], on CPU.
    """
    tensor = tensor.cpu().float()
    for i in range(len(mean)):
        tensor[:, i, :, :] *= std[i]
        tensor[:, i, :, :] += mean[i]
    tensor *= 255.
    tensor = tensor.type(torch.uint8)

    return tensor

def squeeze_2x2(x, rev=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py

    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        rev (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    if alt_order:
        n, c, h, w = x.size()

        if rev:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if rev:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if rev:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        return functools.partial(nn.GroupNorm, num_groups=16)
    else:
        raise NotImplementedError('Invalid normalization type: {}'.format(norm_type))


def init_model(model, init_method='normal'):
    """Initialize model parameters.

    Args:
        model: Model to initialize.
        init_method: Name of initialization method: 'normal' or 'xavier'.
    """
    # Initialize model parameters
    if init_method == 'normal':
        model.apply(_normal_init)
    elif init_method == 'xavier':
        model.apply(_xavier_init)
    else:
        raise NotImplementedError('Invalid weights initializer: {}'.format(init_method))


def _normal_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('Linear') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.

    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    if max_norm > 0:
        for group in optimizer.param_groups:
            clip_grad_norm_(group['params'], max_norm, norm_type)


def get_lr_scheduler(optimizer, args):
    """Get learning rate scheduler."""
    if args.lr_policy == 'step':
        scheduler = torch_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = torch_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'linear':
        # After `lr_warmup_epochs` epochs, decay linearly to 0 for `lr_decay_epochs` epochs.
        def get_lr_multiplier(epoch):
            init_epoch = 1
            return 1.0 - max(0, epoch + init_epoch - args.lr_warmup_epochs) / float(args.lr_decay_epochs + 1)
        scheduler = torch_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    else:
        return NotImplementedError('Invalid learning rate policy: {}'.format(args.lr_policy))
    return scheduler


def _xavier_init(model):
    """Apply Xavier initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('Linear') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def checkerboard_like(x, rev=False):
    """Get a checkerboard mask for `x`, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        x (torch.Tensor): Tensor that will be masked with `x`.
        rev (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.

    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    height, width = x.size(2), x.size(3)
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=x.dtype, device=x.device, requires_grad=False)

    if rev:
        mask = 1. - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x


class STResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
    """
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size, padding):
        super(STResNet, self).__init__()
        self.in_conv = WNConv2d(in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x


class PatchGAN(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, args):
        """Constructs a basic PatchGAN convolutional discriminator.

        Each position in the output is a score of discriminator confidence that
        a 70x70 patch of the input is real.

        Args:
            args: Arguments passed in via the command line.
        """
        super(PatchGAN, self).__init__()

        norm_layer = get_norm_layer(args.norm_type)

        layers = []

        # Double channels for conditional GAN (concatenated src and tgt images)
        num_channels = args.num_channels

        layers += [nn.Conv2d(num_channels, args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(args.num_channels_d, 2 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(2 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(2 * args.num_channels_d, 4 * args.num_channels_d, args.kernel_size_d, stride=2, padding=1),
                   norm_layer(4 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(4 * args.num_channels_d, 8 * args.num_channels_d, args.kernel_size_d, stride=1, padding=1),
                   norm_layer(8 * args.num_channels_d),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(8 * args.num_channels_d, 1, args.kernel_size_d, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)
        init_model(self.model, init_method=args.initializer)

    def forward(self, input_):
        return self.model(input_)


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHECKERBOARD:
            norm_channels = in_channels
            out_channels = 2 * in_channels
            in_channels = 2 * in_channels + 1
        else:
            norm_channels = in_channels // 2
            out_channels = in_channels
            in_channels = in_channels
        self.st_norm = nn.BatchNorm2d(norm_channels, affine=False)
        self.st_net = STResNet(in_channels, mid_channels, out_channels,
                               num_blocks=num_blocks, kernel_size=3, padding=1)

        # Learnable scale and shift for s
        self.s_scale = nn.Parameter(torch.ones(1))
        self.s_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, sldj=None, rev=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_like(x, rev=self.reverse_mask)
            x_b = x * b
            x_b = 2. * self.st_norm(x_b)
            b = b.expand(x.size(0), -1, -1, -1)
            x_b = F.relu(torch.cat((x_b, -x_b, b), dim=1))
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift
            s = s * (1. - b)
            t = t * (1. - b)

            # Scale and translate
            if rev:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_norm(x_id)
            st = F.relu(torch.cat((st, -st), dim=1))
            st = self.st_net(st)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift

            # Scale and translate
            if rev:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, rev=False):

        if rev:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, rev=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, rev)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, rev=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, rev=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, rev)
                x = squeeze_2x2(x, rev=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, rev)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, rev)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, rev=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, rev)
                x = squeeze_2x2(x, rev=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, rev=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, rev)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, rev=True, alt_order=True)

        return x, sldj


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
        un_normalize_x (bool): Un-normalize inputs `x`: shift (-1, 1) to (0, 1)
            assuming we used `transforms.Normalize` with mean 0.5 and std 0.5.
        no_latent (bool): If True, assume both `x` and `z` are image distributions.
            So we should pre-process the same in both directions. E.g., True in CycleFlow.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8,
                 un_normalize_x=False, no_latent=False):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        self.un_normalize_x = un_normalize_x
        self.no_latent = no_latent

        # Get inner layers
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, rev=False):
        ldj = F.softplus(x) + F.softplus(-x) \
              - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        sldj = ldj.reshape(ldj.size(0), -1).sum(-1)

        x, sldj = self.flows(x, sldj, rev)

        return x, sldj

    # def _pre_process(self, x):
    #     """De-quantize and convert the input image `x` to logits.
    #
    #     Args:
    #         x (torch.Tensor): Input image.
    #
    #     Returns:
    #         y (torch.Tensor): Logits of `x`.
    #         ldj (torch.Tensor): Log-determinant of the Jacobian of the transform.
    #
    #     See Also:
    #         - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
    #         - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
    #     """
    #     if self.un_normalize_x:
    #         x = x * 0.5 + 0.5
    #
    #     # Expect inputs in [0, 1]
    #     if x.min() < 0 or x.max() > 1:
    #         raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
    #                          .format(x.min(), x.max()))
    #
    #     # De-quantize
    #     x = (x * 255. + torch.rand_like(x)) / 256.
    #
    #     # Convert to logits
    #     y = (2 * x - 1) * self.data_constraint  # [-0.9, 0.9]
    #     y = (y + 1) / 2                         # [0.05, 0.95]
    #     y = y.log() - (1. - y).log()            # logit
    #
    #     # Save log-determinant of Jacobian of initial transform
    #     ldj = F.softplus(y) + F.softplus(-y) \
    #         - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
    #     ldj = ldj.reshape(ldj.size(0), -1).sum(-1)
    #
    #     return y, ldj


class GANLoss(nn.Module):
    """Module for computing the GAN loss for the generator.

    When `use_least_squares` is turned on, we use mean squared error loss,
    otherwise we use the standard binary cross-entropy loss.

    Note: We use the convention that the discriminator predicts the probability
    that the target image is real. Therefore real corresponds to label 1.0."""
    def __init__(self, device, use_least_squares=False):
        super(GANLoss, self).__init__()
        self.loss_fn = nn.MSELoss() if use_least_squares else nn.BCELoss()
        self.real_label = None  # Label tensor passed to loss if target is real
        self.fake_label = None  # Label tensor passed to loss if target is fake
        self.device = device

    def _get_label_tensor(self, input_, is_tgt_real):
        # Create label tensor if needed
        if is_tgt_real and (self.real_label is None or self.real_label.numel() != input_.numel()):
            self.real_label = torch.ones_like(input_, device=self.device, requires_grad=False)
        elif not is_tgt_real and (self.fake_label is None or self.fake_label.numel() != input_.numel()):
            self.fake_label = torch.zeros_like(input_, device=self.device, requires_grad=False)

        return self.real_label if is_tgt_real else self.fake_label

    def __call__(self, input_, is_tgt_real):
        label = self._get_label_tensor(input_, is_tgt_real)
        return self.loss_fn(input_, label)

    def forward(self, input_):
        raise NotImplementedError('GANLoss should be called directly.')


class ImageBuffer(object):
    """Holds a buffer of old generated images for training. Stabilizes training
    by allowing us to feed a history of generated examples to the discriminator,
    so the discriminator cannot just focus on the newest examples.

    Based on ideas from Section 2.3 of the paper:
    "Learning from Simulated and Unsupervised Images through Adversarial Training"
    by Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Josh Susskind, Wenda Wang, Russ Webb
    (http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf)
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Size of the pool for mixing. Set to 0 to disable image mixer.
        """
        self.capacity = capacity
        self.buffer = []

    def sample(self, images):
        """Sample old images and mix new images into the buffer.

        Args:
            images: New example images to mix into the buffer.

        Returns:
            Tensor batch of images that are mixed from the buffer.
        """
        if self.capacity == 0:
            return images

        # Add to the pool
        mixed_images = []  # Mixture of pool and input images
        for new_img in images:
            new_img = torch.unsqueeze(new_img.data, 0)

            if len(self.buffer) < self.capacity:
                # Pool is still filling, so always add
                self.buffer.append(new_img)
                mixed_images.append(new_img)
            else:
                # Pool is filled, so mix into the pool with probability 1/2
                if random.uniform(0, 1) < 0.5:
                    mixed_images.append(new_img)
                else:
                    pool_img_idx = random.randint(0, len(self.buffer) - 1)
                    mixed_images.append(self.buffer[pool_img_idx].clone())
                    self.buffer[pool_img_idx] = new_img

        return torch.cat(mixed_images, 0)


class JacobianClampingLoss(nn.Module):
    """Module for adding Jacobian Clamping loss.

    See Also:
        https://arxiv.org/abs/1802.08768v2
    """
    def __init__(self, lambda_min=1., lambda_max=20.):
        super(JacobianClampingLoss, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, gz, gz_prime, z, z_prime):
        q = (gz - gz_prime).norm() / (z - z_prime).norm()
        l_max = (q.clamp(self.lambda_max, float('inf')) - self.lambda_max) ** 2
        l_min = (q.clamp(float('-inf'), self.lambda_min) - self.lambda_min) ** 2
        l_jc = l_max + l_min

        return l_jc


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj
        nll = -ll.mean()/np.prod(z.size()[1:])

        return nll


class Flow2Flow(nn.Module):
    """Flow2Flow Model
    Normalizing flows for unpaired image-to-image translation.
    Uses two normalizing flow models (RealNVP) for the generators,
    and two PatchGAN discriminators. The generators map to a shared
    intermediate latent space `Z` with simple prior `p_Z`, and the
    whole system optimizes a hybrid GAN-MLE objective.
    """
    def __init__(self, args):
        """
        Args:
            args: Configuration args passed in via the command line.
        """
        super(Flow2Flow, self).__init__()
        self.device = 'cuda'
        self.gpu_ids = args.gpu_ids
        self.is_training = args.is_training

        self.in_channels = args.num_channels
        self.out_channels = 4 ** (args.num_scales - 1) * self.in_channels

        # Set up RealNVP generators (g_src: X <-> Z, g_tgt: Y <-> Z)
        self.g_src = RealNVP(num_scales=args.num_scales,
                             in_channels=args.num_channels,
                             mid_channels=args.num_channels_g,
                             num_blocks=args.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        init_model(self.g_src, init_method=args.initializer)
        self.g_tgt = RealNVP(num_scales=args.num_scales,
                             in_channels=args.num_channels,
                             mid_channels=args.num_channels_g,
                             num_blocks=args.num_blocks,
                             un_normalize_x=True,
                             no_latent=False)
        init_model(self.g_tgt, init_method=args.initializer)

        if self.is_training:
            # Set up discriminators
            self.d_tgt = PatchGAN(args)  # Answers Q "is this tgt image real?"
            self.d_src = PatchGAN(args)  # Answers Q "is this src image real?"

            self._data_parallel()

            # Set up loss functions
            self.max_grad_norm = args.clip_gradient
            self.lambda_mle = args.lambda_mle
            self.mle_loss_fn = RealNVPLoss()
            self.gan_loss_fn = GANLoss(device=self.device, use_least_squares=True)

            self.clamp_jacobian = args.clamp_jacobian
            self.jc_loss_fn = JacobianClampingLoss(args.jc_lambda_min, args.jc_lambda_max)

            # Setup image mixers
            buffer_capacity = 50 if args.use_mixer else 0
            self.src2tgt_buffer = ImageBuffer(buffer_capacity)  # Buffer of generated tgt images
            self.tgt2src_buffer = ImageBuffer(buffer_capacity)  # Buffer of generated src images
        else:
            self._data_parallel()

        # Images in flow src -> lat -> tgt
        self.src = None
        self.src2lat = None
        self.src2tgt = None

        # Images in flow tgt -> lat -> src
        self.tgt = None
        self.tgt2lat = None
        self.tgt2src = None

        # Jacobian clamping tensors
        self.src_jc = None
        self.tgt_jc = None
        self.src2tgt_jc = None
        self.tgt2src_jc = None

        # Discriminator loss
        self.loss_d_tgt = None
        self.loss_d_src = None
        self.loss_d = None

        # Generator GAN loss
        self.loss_gan_src = None
        self.loss_gan_tgt = None
        self.loss_gan = None

        # Generator MLE loss
        self.loss_mle_src = None
        self.loss_mle_tgt = None
        self.loss_mle = None

        # Jacobian Clamping loss
        self.loss_jc_src = None
        self.loss_jc_tgt = None
        self.loss_jc = None

        # Generator total loss
        self.loss_g = None

    def forward(self):
        """No-op. We do the forward pass in `backward_g`."""
        pass

    def test(self):
        """Run a forward pass through the generator for test inference.
        Used during test inference only, as this throws away forward-pass values,
        which would be needed for backprop.
        Important: Call `set_inputs` prior to each successive call to `test`.
        """
        # Disable auto-grad because we will not backprop
        with torch.no_grad():
            src2lat, _ = self.g_src(self.src, rev=False)
            src2lat2tgt, _ = self.g_tgt(src2lat, rev=True)
            self.src2tgt = torch.tanh(src2lat2tgt)

            tgt2lat, _ = self.g_tgt(self.tgt, rev=False)
            tgt2lat2src, _ = self.g_src(tgt2lat, rev=True)
            self.tgt2src = torch.tanh(tgt2lat2src)

    def _forward_d(self, d, real_img, fake_img):
        """Forward  pass for one discriminator."""

        # Forward on real and fake images (detach fake to avoid backprop into generators)
        loss_real = self.gan_loss_fn(d(real_img), is_tgt_real=True)
        loss_fake = self.gan_loss_fn(d(fake_img.detach()), is_tgt_real=False)
        loss_d = 0.5 * (loss_real + loss_fake)

        return loss_d

    def backward_d(self, images_a, images_b):
        self.src = images_a
        self.tgt = images_b
        # Forward tgt discriminator
        src2tgt = self.src2tgt_buffer.sample(self.src2tgt)
        self.loss_d_tgt = self._forward_d(self.d_tgt, self.tgt, src2tgt)

        # Forward src discriminator
        tgt2src = self.tgt2src_buffer.sample(self.tgt2src)
        self.loss_d_src = self._forward_d(self.d_src, self.src, tgt2src)

        # Backprop
        return {"dis_loss": self.loss_d_tgt + self.loss_d_src}


    def backward_g(self, images_a, images_b):
        self.src = images_a
        self.tgt = images_b
        if self.clamp_jacobian:
            # Double batch size with perturbed inputs for Jacobian Clamping
            self._jc_preprocess()

        # Forward src -> lat: Get MLE loss
        self.src2lat, sldj_src2lat = self.g_src(self.src, rev=False)
        self.loss_mle_src = self.lambda_mle * self.mle_loss_fn(self.src2lat, sldj_src2lat)

        # Finish src -> lat -> tgt: Say target is real to invert loss
        self.src2tgt, _ = self.g_tgt(self.src2lat, rev=True)
        self.src2tgt = torch.tanh(self.src2tgt)

        # Forward tgt -> lat: Get MLE loss
        self.tgt2lat, sldj_tgt2lat = self.g_tgt(self.tgt, rev=False)
        self.loss_mle_tgt = self.lambda_mle * self.mle_loss_fn(self.tgt2lat, sldj_tgt2lat)

        # Finish tgt -> lat -> src: Say source is real to invert loss
        self.tgt2src, _ = self.g_src(self.tgt2lat, rev=True)
        self.tgt2src = torch.tanh(self.tgt2src)

        # Jacobian Clamping loss
        if self.clamp_jacobian:
            # Split inputs and outputs from Jacobian Clamping
            self._jc_postprocess()
            self.loss_jc_src = self.jc_loss_fn(self.src2tgt, self.src2tgt_jc, self.src, self.src_jc)
            self.loss_jc_tgt = self.jc_loss_fn(self.tgt2src, self.tgt2src_jc, self.tgt, self.tgt_jc)
            self.loss_jc = self.loss_jc_src + self.loss_jc_tgt
        else:
            self.loss_jc_src = self.loss_jc_tgt = self.loss_jc = 0.

        # GAN loss
        self.loss_gan_src = self.gan_loss_fn(self.d_tgt(self.src2tgt), is_tgt_real=True)
        self.loss_gan_tgt = self.gan_loss_fn(self.d_src(self.tgt2src), is_tgt_real=True)

        # Total losses
        self.loss_gan = self.loss_gan_src + self.loss_gan_tgt
        self.loss_mle = self.loss_mle_src + self.loss_mle_tgt

        # Backprop
        return {"gan_loss": self.loss_gan, "mle_loss": self.loss_mle, "jac_loss": self.loss_jc}

    def _data_parallel(self):
        self.g_src = nn.DataParallel(self.g_src, [self.gpu_ids]).to(self.device)
        self.g_tgt = nn.DataParallel(self.g_tgt, [self.gpu_ids]).to(self.device)
        if self.is_training:
            self.d_src = nn.DataParallel(self.d_src, [self.gpu_ids]).to(self.device)
            self.d_tgt = nn.DataParallel(self.d_tgt, [self.gpu_ids]).to(self.device)

    def _jc_preprocess(self):
        """Pre-process inputs for Jacobian Clamping. Doubles batch size.
        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        delta = torch.randn_like(self.src)
        src_jc = self.src + delta / delta.norm()
        src_jc.clamp_(-1, 1)
        self.src = torch.cat((self.src, src_jc), dim=0)

        delta = torch.randn_like(self.tgt)
        tgt_jc = self.tgt + delta / delta.norm()
        tgt_jc.clamp_(-1, 1)
        self.tgt = torch.cat((self.tgt, tgt_jc), dim=0)

    def _jc_postprocess(self):
        """Post-process outputs after Jacobian Clamping.
        Chunks `self.src` into `self.src` and `self.src_jc`,
        `self.src2tgt` into `self.src2tgt` and `self.src2tgt_jc`,
        and similarly for `self.tgt` and `self.tgt2src`.
        See Also:
            Algorithm 1 from https://arxiv.org/1802.08768v2
        """
        self.src, self.src_jc = self.src.chunk(2, dim=0)
        self.tgt, self.tgt_jc = self.tgt.chunk(2, dim=0)

        self.src2tgt, self.src2tgt_jc = self.src2tgt.chunk(2, dim=0)
        self.tgt2src, self.tgt2src_jc = self.tgt2src.chunk(2, dim=0)