import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.inn_subnets import *
from src.utils.dimensionality_calculations import calculate_downscale_dimensionality
from typing import Union, List, Callable


def append_multi_scale_inn_blocks(nodes: Ff.SequenceINN, num_scales: int, blocks_per_scale: Union[List, int],
                                  dimensions: List, subnets: Union[Callable, List[Callable]],
                                  downsampling_type: str = "haar", instant_downsampling: bool = False,
                                  conditional_blocks: bool = False, varying_kernel_size: bool = False,
                                  flatten: bool = False, clamping: float = 1.):
    """

    :param nodes: sequence inn to append the ms inn blocks to
    :param num_scales: number of scales
    :param blocks_per_scale: number of coupling blocks within one scale
    :param dimensions: dimensionality of input
    :param subnets: subnetworks within each coupling block of each scale
    :param downsampling_type:
    :param instant_downsampling: boolean whether there is downsampling before the first coupling block
    :param conditional_blocks:
    :param varying_kernel_size:
    :param flatten:
    :param clamping:
    """

    if isinstance(blocks_per_scale, int):
        blocks_per_scale = [blocks_per_scale for _ in range(num_scales)]
    elif not isinstance(blocks_per_scale, List):
        raise TypeError("The kwarg 'blocks_per_scale' has to be of type list or int!")

    if downsampling_type == "irevnet":
        downsampling_block = Fm.IRevNetDownsampling
    elif downsampling_type == "haar":
        downsampling_block = Fm.HaarDownsampling
    else:
        raise KeyError("Please specify a downsampling type from 'irevnet' and 'haar'!")

    if isinstance(subnets, Callable):
        subnets = [subnets for _ in range(num_scales)]
    elif isinstance(subnets, List):
        if len(subnets) != num_scales:
            raise RuntimeError(f"The amount of subnetworks ({len(subnets)}) in the subnets list does not match the number of scales ({num_scales})!")
    else:
        raise TypeError("The kwarg 'subnets' has to be of type list or callable!")

    orig_dims = dimensions
    condition, condition_shape = None, None
    for scale in range(num_scales):
        if instant_downsampling or scale > 0:
            nodes.append(downsampling_block)
            orig_dims = calculate_downscale_dimensionality(orig_dims, 2)
        for k in range(blocks_per_scale[scale]):
            if varying_kernel_size and k % 2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnets[scale]
            if conditional_blocks:
                condition = scale
                orig_dims[0] = 2
                condition_shape = orig_dims
            append_all_in_one_block(nodes,
                                    sub_network=subnet,
                                    condition=condition,
                                    condition_shape=condition_shape,
                                    clamping=clamping
                                    )

    if flatten:
        nodes.append(Fm.Flatten)


def append_all_in_one_block(nodes: Ff.SequenceINN, sub_network: Callable,
                            condition=None, condition_shape=None, clamping=1.):
    nodes.append(Fm.AllInOneBlock,
                 subnet_constructor=sub_network,
                 cond=condition,
                 cond_shape=condition_shape,
                 affine_clamping=clamping,
                 gin_block=False,
                 global_affine_init=.7,
                 global_affine_type="SOFTPLUS",
                 permute_soft=False,
                 learned_householder_permutation=0,
                 reverse_permutation=False)
    return nodes
