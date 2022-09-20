import torch
import warnings
from enum import IntEnum
from skimage import io
import numpy as np
from distutils.version import LooseVersion
import torch.nn.functional as F

from .utils import *

from common import geometry

default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
}


class FaceAlignment:
    def __init__(self, device='cuda', softmax_temperature=0.1, heatmap_to_xy_scale_factor=1.15):
        """
        A simplified and efficient modification of the face alignment algorithm.

        The most important modifications:
        - support only 2d landmarks
        - face detection removed (as we always have the face in the middle of the picture)
        - slow numpy heatmap to x, y conversion algorithm is replaced by a vectorized one
          similar to the keypoint detector.

        :param softmax_temperature: softmax temperature for heatmap to x, y conversion.
        :param heatmap_to_xy_scale_factor: empirical scale factor, needed to fit the results of the vectorized
        heatmap to x, y conversion.

        """
        self.device = device
        self._softmax_temperature = softmax_temperature
        self._heatmap_to_xy_scale_factor = heatmap_to_xy_scale_factor

        if LooseVersion(torch.__version__) < LooseVersion('1.5.0'):
            raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        network_size = 4  # LARGE

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Initialise the face aligment networks
        network_name = '2DFAN-' + str(network_size)
        self.face_alignment_net = torch.jit.load(load_file_from_url(default_model_urls[network_name]))
        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        self._grid = geometry.make_coordinate_grid2((64, 64), device=device)

    def get_landmarks(self, input):

        prediction = self.face_alignment_net(input)
        prediction_shape = prediction.shape
        heatmap = prediction.reshape(prediction_shape[0], prediction_shape[1], -1)
        heatmap = F.softmax(heatmap / self._softmax_temperature, dim=2)
        heatmap = heatmap.reshape(*prediction_shape)

        heatmap = heatmap.unsqueeze(-1)
        landmarks = (heatmap * self._grid).sum(dim=(2, 3))
        landmarks = landmarks * self._heatmap_to_xy_scale_factor

        return landmarks
