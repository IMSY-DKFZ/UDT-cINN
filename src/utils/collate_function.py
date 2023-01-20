import numpy as np
import torch


def collate(batch_dict):
    batch_size = len(batch_dict)

    image_a = torch.zeros([batch_size, *np.shape(batch_dict[0]["image_a"])])
    image_b = torch.zeros([batch_size, *np.shape(batch_dict[0]["image_b"])])

    seg_a, seg_b = list(), list()
    oxy_a, oxy_b = list(), list()

    for batch_idx in range(batch_size):
        image_a[batch_idx, :, :, :] = batch_dict[batch_idx]["image_a"]
        image_b[batch_idx, :, :, :] = batch_dict[batch_idx]["image_b"]
        seg_a.append(batch_dict[batch_idx]["seg_a"])
        seg_b.append(batch_dict[batch_idx]["seg_b"])
        oxy_a.append(batch_dict[batch_idx]["oxy_a"])
        oxy_b.append(batch_dict[batch_idx]["oxy_b"])

    return {"image_a": image_a.type(torch.float32), "image_b": image_b.type(torch.float32),
            "seg_a": seg_a, "seg_b": seg_b,
            "oxy_a": oxy_a, "oxy_b": oxy_b}
