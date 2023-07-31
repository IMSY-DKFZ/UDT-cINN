import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.inn_subnets import subnet_alignflow_res_net
import torch


def calculate_downsampling_dim(dims):
    assert len(dims) == 3
    dim_2 = int(round(dims[2] / 2))
    dim_1 = int(round(dims[1] / 2))
    dim_0 = dims[0] * 4

    return dim_0, dim_1, dim_2


def build_inn(dimensions):

    nodes = [Ff.InputNode(*dimensions)]
    split_nodes = []

    num_layers = 2
    num_scales = 2

    for scale in range(num_scales):
        for k in range(num_layers):
            if scale > 0 and k == 0:
                nodes.append(Ff.Node(split_node.out1, Fm.IRevNetDownsampling, {}))
            else:
                nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.RNVPCouplingBlock,
                                 {"subnet_constructor": subnet_alignflow_res_net}))
        for k in range(num_layers):
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.RNVPCouplingBlock,
                                 {"subnet_constructor": subnet_alignflow_res_net}))

        if scale < num_scales - 1:
            split_node = Ff.Node(nodes[-1], Fm.Split, {})
            nodes.append(split_node)
            nodes.append(Ff.OutputNode(split_node.out0))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.RNVPCouplingBlock,
                             {"subnet_constructor": subnet_alignflow_res_net}))

    nodes.append(Ff.OutputNode(nodes[-1]))

    return Ff.ReversibleGraphNet(nodes, verbose=False)


if __name__ == "__main__":
    import torch.nn as nn

    size = (1, 16, 128, 256)
    r = torch.rand(size).cuda()
    net = build_inn(size[1:])
    net = net.cuda()

    print(torch.max(torch.abs(r - net(net(r)[0], rev=True)[0])))







