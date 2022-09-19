import torch
import warnings
from enum import IntEnum
from skimage import io
import numpy as np
from distutils.version import LooseVersion

from .utils import *



default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
}


class FaceAlignment:
    def __init__(self, device='cuda', flip_input=False, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.verbose = verbose

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

    def get_landmarks(self, input):
        heatmaps = self.face_alignment_net(input)
        if self.flip_input:
            heatmaps += flip(self.face_alignment_net(flip(input)), is_label=True)
        heatmaps = heatmaps.detach().cpu().numpy()

        pts = get_preds_fromhm(heatmaps)
        pts = torch.from_numpy(pts) * 4

        return pts
