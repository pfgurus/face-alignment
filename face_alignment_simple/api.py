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

    def get_landmarks_simple(self, image_or_path):

        image = get_image(image_or_path)

        inp = image
        scale = 1
        center = torch.tensor((128.0, 128.0))

        inp = torch.from_numpy(inp.transpose(
            (2, 0, 1))).float()

        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        out = self.face_alignment_net(inp).detach()
        if self.flip_input:
            out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
        out = out.cpu().numpy()

        pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

        return pts
