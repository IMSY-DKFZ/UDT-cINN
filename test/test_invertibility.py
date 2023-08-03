import torch
import unittest
from numpy.testing import assert_array_almost_equal
from pathlib import Path
import pytorch_lightning as pl

from src.utils.config_io import load_config
from src.trainers import GanCondinitionalDomainAdaptationINNHSI, GanCondinitionalDomainAdaptationINN


here = Path(__file__).parent


class InnInvertibilityTest(unittest.TestCase):

    def setUp(self) -> None:

        self.config = load_config(str(here.parent / "src" / "configs" / "test_conf.yaml"))
        self.powers_of_two = [2**power for power in range(2, 8)]
        self.models = {
            "cINN": {
                "model": GanCondinitionalDomainAdaptationINN,
                "data_dimensions": 3
            },
            "cINN_HSI": {
                "model": GanCondinitionalDomainAdaptationINNHSI,
                "data_dimensions": 1
            }
        }

    def test_model_invertibility(self):
        for model_name, model_specs in self.models.items():
            model = model_specs["model"]
            input_shape = [torch.randint(low=2, high=len(self.powers_of_two), size=(1,)).item() for dim in range(model_specs["data_dimensions"])]
            input_shape = [self.powers_of_two[num] for num in input_shape]
            self.config.data["dimensions"] = input_shape if len(input_shape) > 1 else input_shape[0]
            self.batch_size = self.config.batch_size

            conv_inn = model(self.config)
            print(conv_inn.model_name())
            conv_inn = conv_inn.cuda()

            rand_shape = [self.batch_size]
            rand_shape += input_shape

            for i in range(10):
                x_a = torch.randn(rand_shape).cuda()
                x_b = torch.randn(rand_shape).cuda()

                for mode, image in zip(["a", "b"], [x_a, x_b]):
                    orig_image, inv_image = conv_inn.sample_inverted_image(image, mode=mode, visualize=False)
                    assert_array_almost_equal(orig_image, inv_image, decimal=1)

