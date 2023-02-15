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


def collate_hsi(batch):
    spectra_a = torch.stack([i['spectra_a'] for i in batch], dim=0)
    spectra_b = torch.stack([i['spectra_b'] for i in batch], dim=0)

    seg_a = torch.tensor([i['seg_a'] for i in batch])
    seg_b = torch.tensor([i['seg_b'] for i in batch])
    subjects_a = np.array([i['subjects_a'] for i in batch])
    subjects_b = np.array([i['subjects_b'] for i in batch])
    image_ids_a = np.array([i['image_ids_a'] for i in batch])
    image_ids_b = np.array([i['image_ids_b'] for i in batch])
    mapping = batch[0]['mapping']
    order = batch[0]['order']
    return {
        'spectra_a': spectra_a,
        'spectra_b': spectra_b,
        'seg_a': seg_a,
        'seg_b': seg_b,
        'subjects_a': subjects_a,
        'subjects_b': subjects_b,
        'image_ids_a': image_ids_a,
        'image_ids_b': image_ids_b,
        'mapping': mapping,
        'order': order
    }
