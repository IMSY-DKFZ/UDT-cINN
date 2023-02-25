import torch
from abc import abstractmethod, ABC
from typing import Tuple, Dict
from omegaconf import DictConfig
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import os
from src.visualization import col_bar


class DomainAdaptationTrainerBasePA(pl.LightningModule, ABC):
    def __init__(self, experiment_config: DictConfig):
        super().__init__()

        self.config = experiment_config

        self.dimensions = self.config.data.dimensions
        self.channels = self.dimensions[0]
        self.dimensionality = np.product(self.dimensions)

    def model_name(self):
        return self._get_name()

    @staticmethod
    def aggregate_total_loss(losses_dict: Dict):
        total_loss = 0
        for loss_name, loss_value in losses_dict.items():
            total_loss += loss_value
            losses_dict[loss_name] = loss_value.detach()
        losses_dict["loss"] = total_loss
        return losses_dict

    def log_losses(self, losses_dict: Dict):
        for loss_key, loss_value in losses_dict.items():
            self.log(loss_key, loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.config.batch_size)

    def get_images(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images_a = batch["image_a"]
        images_b = batch["image_b"]
        images_a, images_b = images_a.cuda(non_blocking=True), images_b.cuda(non_blocking=True)
        if self.config.condition == "segmentation":
            seg_a = torch.from_numpy(np.array(batch["seg_a"])).type(torch.float32)
            ret_data = (images_a, seg_a), images_b
        else:
            ret_data = images_a, images_b
        return ret_data

    def recon_criterion(self, model_recon, target_recon):
        """

        :param model_recon:
        :param target_recon:
        :return:
        """
        if self.config.recon_criterion == "mse":
            recon_error = torch.mean((model_recon - target_recon)**2)
        elif self.config.recon_criterion == "abs":
            recon_error = torch.mean(torch.abs(model_recon - target_recon))
        else:
            raise KeyError("Please use one of the implemented reconstruction criteria "
                           "'abs' or 'mse' in config.recon_criterion!")
        return recon_error

    def get_label_conditions(self, labels, n_labels: int, labels_size=None):
        if isinstance(labels, int) and labels == 0:
            one_hot_seg_shape = list(labels_size)
            if len(one_hot_seg_shape) == 4:
                one_hot_seg_shape.pop(1)

            if self.config.real_labels == "constant":
                labels = torch.ones(size=one_hot_seg_shape) / n_labels
            elif self.config.real_labels == "noise":
                labels = torch.rand(size=one_hot_seg_shape)

            # random_segmentation = np.random.choice(range(n_labels), size=one_hot_seg_shape)
            #
            # labels = torch.from_numpy(random_segmentation).type(torch.float32)

            one_hot_seg = labels.type(torch.float32)

        else:
            if self.config.label_noise:
                one_hot_seg = torch.stack(
                    [(labels == label) + torch.rand_like(labels) * self.config.label_noise_level for label in
                     range(n_labels)],
                    dim=1
                )
            else:
                one_hot_seg = torch.stack([(labels == label) for label in range(n_labels)], dim=1)

        one_hot_seg /= torch.linalg.norm(one_hot_seg, dim=1, keepdim=True, ord=1)

        assert one_hot_seg.size()[1] == n_labels

        return one_hot_seg

    @abstractmethod
    def forward(self, inp, mode="a", *args, **kwargs):
        """
        Forward pass of the domain adaptation model.
        Each model must implement this method as it defines how the images from each domain are mapped into the latent
        space and then to the other domain

        :param inp:
        :param mode:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def translate_image(self, image, input_domain="a"):
        """

        :param image:
        :param input_domain:
        :return:
        """
        pass

    def test_step(self, batch, batch_idx, *args, **kwargs):
        path = os.path.join(self.config.save_path, "testing")
        generated_image_data_path = os.path.join(path, "generated_image_data")
        os.makedirs(generated_image_data_path, exist_ok=True)

        images_a, images_b = self.get_images(batch)
        if len(images_a) == 5 or len(images_a) == 5:
            images_a, images_b = torch.squeeze(images_a, dim=0), torch.squeeze(images_b, dim=0)

        images_ab = self.translate_image(images_a, input_domain="a")
        images_ba = self.translate_image(images_b, input_domain="b")

        images_a = images_a[0].cpu().numpy() if isinstance(images_a, tuple) else images_a.cpu().numpy()
        images_b = images_b[0].cpu().numpy() if isinstance(images_b, tuple) else images_b.cpu().numpy()
        images_ab = images_ab[0].cpu().numpy() if isinstance(images_ab, tuple) else images_ab.cpu().numpy()
        images_ba = images_ba[0].cpu().numpy() if isinstance(images_ba, tuple) else images_ba.cpu().numpy()

        if self.config.normalization not in ["None", "none"]:
            if self.config.normalization == "standardize":
                images_a = images_a * self.config.data.std_a + self.config.data.mean_a
                images_ba = images_ba * self.config.data.std_a + self.config.data.mean_a

                images_b = images_b * self.config.data.std_b + self.config.data.mean_b
                images_ab = images_ab * self.config.data.std_b + self.config.data.mean_b

        np.savez(os.path.join(generated_image_data_path, f"test_batch_{batch_idx}"),
                 images_a=images_a,
                 images_b=images_b,
                 images_ab=images_ab,
                 images_ba=images_ba,
                 seg_a=batch["seg_a"],
                 seg_b=batch["seg_b"],
                 oxy_a=batch["oxy_a"],
                 oxy_b=batch["oxy_b"],
                 )

        if True:
            generated_images_path = os.path.join(path, "generated_images")
            os.makedirs(generated_images_path, exist_ok=True)
            plt.figure(figsize=(6, 6))
            plt.subplot(2, 2, 1)
            plt.title("Domain A")
            img_a = plt.imshow(images_a[0, 0, :, :])
            col_bar(img_a)
            plt.subplot(2, 2, 2)
            plt.title("Domain A to Domain B")
            img_ab = plt.imshow(images_ab[0, 0, :, :])
            col_bar(img_ab)
            plt.subplot(2, 2, 3)
            plt.title("Domain B")
            img_b = plt.imshow(images_b[0, 0, :, :])
            col_bar(img_b)
            plt.subplot(2, 2, 4)
            plt.title("Domain B to Domain A")
            img_ba = plt.imshow(images_ba[0, 0, :, :])
            col_bar(img_ba)
            plt.savefig(os.path.join(generated_images_path, f"test_batch_{batch_idx}.png"))
            plt.close()
