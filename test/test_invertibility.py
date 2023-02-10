from src.utils.config_io import load_config
from src.trainers import GanCondinitionalDomainAdaptationINNHSI, GanCondinitionalDomainAdaptationINN
import torch
import unittest
from numpy.testing import assert_array_almost_equal


class InnInvertibilityTest(unittest.TestCase):

    def setUp(self) -> None:

        self.config = load_config("/home/kris/Work/Repositories/miccai23/src/configs/test_conf.yaml")

        self.model_list = [
            GanCondinitionalDomainAdaptationINN,
            GanCondinitionalDomainAdaptationINNHSI,
        ]

        self.input_shape_list = [
            [2, 128, 128],
            100,
        ]

    def test_model_invertibility(self):
        for model, input_shape in zip(self.model_list, self.input_shape_list):
            self.config.data["dimensions"] = input_shape
            self.batch_size = self.config.batch_size

            conv_inn = model(self.config)
            print(conv_inn.model_name())
            conv_inn = conv_inn.cuda()

            rand_shape = [self.batch_size]
            rand_shape += input_shape if isinstance(input_shape, list) else [input_shape]

            for i in range(10):
                x_a = torch.randn(rand_shape).cuda()
                x_b = torch.randn(rand_shape).cuda()

                for mode, image in zip(["a", "b"], [x_a, x_b]):
                    orig_image, inv_image = conv_inn.sample_inverted_image(image, mode=mode, visualize=False)
                    assert_array_almost_equal(orig_image, inv_image, decimal=1)

