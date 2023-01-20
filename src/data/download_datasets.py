import torch
from torchvision.datasets import MNIST, USPS
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import yaml
from src.utils.normalizations import standardize


torch.manual_seed(42)
data_root = "/home/kris/Work/Data"
normalization_standardize = True
img_size = [32, 32]
datasets = {"MNIST": MNIST,
            "USPS": USPS}
train_val_split = [7/8, 1/8]


for dataset_key, dataset_value in datasets.items():
    dataset_path = os.path.join(data_root, dataset_key)
    os.makedirs(dataset_path, exist_ok=True)

    conf_path = os.path.join(dataset_path, "data_config.yaml")
    if os.path.exists(conf_path):
        os.remove(conf_path)
    with open(conf_path, "a") as data_conf_file:
        yaml.dump({"dataset_name": dataset_key,
                   "dims": [1, *img_size],
                   "standardize": normalization_standardize},
                  data_conf_file)

    for split_path in ["training", "validation", "test"]:
        os.makedirs(os.path.join(dataset_path, split_path), exist_ok=True)
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(img_size)])

    train_val_data = dataset_value(root=dataset_path,
                                   train=True,
                                   download=True,
                                   transform=transformations)
    train_val_dataloader = DataLoader(train_val_data, batch_size=1, shuffle=True)
    number_of_train_val_images = len(train_val_dataloader)

    for idx, (image, label) in enumerate(train_val_dataloader):
        array = np.array(image[0, :, :, :])
        array, _, _ = standardize(array, log=False)
        if idx/number_of_train_val_images < train_val_split[0]:
            split = "training"
        else:
            split = "validation"
        np.save(os.path.join(dataset_path, split, str(idx)), array)

    test_data = dataset_value(root=dataset_path,
                              train=False,
                              download=True,
                              transform=transformations)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    number_of_test_images = len(test_dataloader)

    for idx, (image, label) in enumerate(test_dataloader):
        array = np.array(image[0, :, :, :])
        array, _, _ = standardize(array, log=False)
        split = "test"
        np.save(os.path.join(dataset_path, split, str(idx)), array)
