import collections
import glob
import importlib
import logging
import os
import shutil

import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed


from . import geometry

logger = logging.getLogger(__name__)


def make_red_green_image(image):
    """
    Convert a 1 channel image of positive and negative numbers into a color image
    for visualization. Red corresponds to negative numbers, green to positive ones.
    :param image: one channel image (H, W) or (H, W, 1).
    :return: a 3-channel image in RGB format.
    """
    if np.ndim == 2:
        image = np.squeeze(image, axis=2)
    pos = (image >= 0) * image
    neg = (image <= 0) * image
    blue = np.zeros(image.shape, dtype=image.dtype)
    rg = np.stack([-neg, pos, blue], axis=-1)
    return rg


def make_red_green(t):
    """
    Convert a tensor of positive and negative numbers into a color image for visualization.
    Red corresponds to negative numbers, green to positive ones.
    :param t: tensor (B, H, W)
    :return: tensor (B, 3, H, W) in RGB format.
    """
    with torch.no_grad():
        pos = (t >= 0) * t
        neg = (t <= 0) * t
        blue = torch.zeros_like(t)
        rg = torch.stack([-neg, pos, blue], axis=1)
    return rg


def show_flow(flow, scale=10):
    """
    Create image tensors for flow visualization.
    :param flow: a tensor (B, H, W, 2).
    :param scale: scale factor
    :return: flow_x, flow_y images (B, 3, H, W).
    """
    with torch.no_grad():
        flow_vis = []
        for i in range(flow.shape[-1]):
            flow_vis.append(make_red_green(flow[..., i] * scale).clip(0, 1))
    return tuple(flow_vis)


def tensor_to_images(t, red_green=None):
    """
    Convert a tensor to images for visualization.
    :param t: a tensor (B, C, H, W) or (C, H, W)
    :param red_green: if C == 1, show negative values in red and positive in green. None: autodetect.
    :return: a list of HWC, 3 channel images.
    """
    if type(t) == torch.Tensor:
        t = t.detach().cpu().numpy()

    if t.ndim < 3 or t.ndim > 4:
        raise ValueError('Unsupported tensor shape')

    if t.ndim == 3:
        t = np.expand_dims(t, 0)

    t = np.ascontiguousarray(np.moveaxis(t, 1, -1))
    result = []
    for i in range(len(t)):
        image = t[i]
        if image.shape[2] == 1:
            image = np.squeeze(image)
        if image.ndim == 2:
            if red_green is None and image.min() < 0:
                red_green = True
            if red_green:
                image = make_red_green_image(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result.append(image)
    return result


def show_tensor(name, t, red_green=None, wait_key_time=1):
    """
    Show  tensor in a window.
    :param name: window name
    :param t: a tensor (B, C, H, W) or (C, H, W)
    :param red_green: if C == 1, show negative values in red and positive in green. None: autodetect.
    :param wait_key_time: a time in ms to wait for a keypress. 0 - wait forever.
    :return:
    """
    images = tensor_to_images(t, red_green)
    for i, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(name+f'-{i}', image)
    cv2.waitKey(wait_key_time)


def batch_to_device(batch, device):
    """
    Move all tensors in a batch to a device, preserve other data.
    :param batch: a dictonary of key, data pairs. Data can be a single value, list or tuple.
    :param device: torch device.
    :return: a dictionary with all data tensors moved to the device.
    """
    new_batch = {}
    for k, v in batch.items():
        new_v = v
        if type(v) == torch.Tensor:
            new_v = v.to(device)
        elif type(v) == list or type(v) == tuple:
            new_v = []
            for e in v:
                new_e = e
                if type(e) == torch.Tensor:
                    new_e = e.to(device)
                new_v.append(new_e)
            if type(v) == tuple:
                new_v = tuple(new_v)
        new_batch[k] = new_v
    return new_batch


def set_device(default=None):
    """
    Set default device to create models and tensors on.
    :param default 'cpu', 'cuda:N' or None to autodetect.
    """
    if (default is None and torch.cuda.is_available()) or (default is not None and default.startswith('cuda')):
        if default is None:
            device = torch.device('cuda', 0)
        else:
            device = torch.device(default)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    return device


def ddp_setup(rank, devices, master_addr, master_port):
    # Each process will only see its GPU under id 0, do this setting before any other CUDA call.
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{devices[rank]}'
    logger.info(f"Rank {rank} uses physical device {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not torch.cuda.is_available():
        raise RuntimeError('Only CUDA training is supported')

    # Initialize process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    backend = 'nccl' if distributed.is_nccl_available() else 'gloo'
    logger.info(f'Rank {rank} uses backend {backend}')
    world_size = len(devices)
    distributed.init_process_group(backend, rank=rank, world_size=world_size)

    set_device()


def ddp_cleanup():
    distributed.destroy_process_group()


def get_checkpoint_file(checkpoint_path):
    """
    Get the checkpoint file from a file or directory path
    :param checkpoint_path: path to the checkpoint file. If it is a directory, return the last checkpoint.
    """
    if os.path.isdir(checkpoint_path):
        checkpoints = glob.glob(checkpoint_path + '/*.pth.tar')
        if not checkpoints:
            raise ValueError(f'No checkpoints in {checkpoint_path}')
        checkpoints.sort()
        checkpoint_file = os.path.normpath(checkpoints[-1])
    else:
        checkpoint_file = checkpoint_path

    return checkpoint_file


def load_config(config_file=None, checkpoint_file=None):
    """
    Loads configuration from a config or a checkpoint.
    If only the config_file is specified, no checkpoint will be loaded.
    If only the checkpoint_file is specified, the config is loaded from the checkpoint.
    If both are specified, the config from the config_file is taken.

    :param config_file: path to the config file.
    :param checkpoint_file: path to the checkpoint file. If it is a directory, load the last checkpoint.
    :return config dictionary.
    """

    if config_file:
        logger.info(f'Loading config from {config_file}')
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif checkpoint_file:
        checkpoint_file = get_checkpoint_file(checkpoint_file)
        logger.info(f'Loading checkpoint {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        logger.info(f'Loading config from checkpoint {checkpoint_file}')
        config = checkpoint['config']
    else:
        raise ValueError('At least one of the config_file and checkpoint_file must be specified')

    return config


def load_checkpoint(checkpoint_file, map_location=None):
    """
    Loads a checkpoint.
    :param checkpoint_file: path to the checkpoint file. If it is a directory, load the last checkpoint.
    :param map_location: see torch.load().
    :return A tuple (epoch, time, checkpoint).
    """

    checkpoint_file = get_checkpoint_file(checkpoint_file)

    logger.info(f'Loading checkpoint {checkpoint_file}')
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    epoch = checkpoint['epoch'] + 1  # Switch to the next epoch.
    time = checkpoint['time'] if 'time' in checkpoint else None

    return epoch, time, checkpoint


def load_submodule(submodule, key_prefix, checkpoint):
    """
    Load a sub-module (a model which is an attribute of another model) from a checkpoint.
    :param submodule: submodule.
    :param checkpoint: checkpoint.
    :param key_prefix: dictionary key prefix to remove.
    """
    submodule_state = collections.OrderedDict()
    for k, v in checkpoint['modules'][0].items():
        if k.startswith(key_prefix + '.'):
            new_key = k[len(key_prefix + '.'):]
            submodule_state[new_key] = v
    submodule.load_state_dict(submodule_state)


def save_checkpoint(path, epoch, time, config, modules, **kwargs):
    """
    Save a checkpoint at the end of the epoch.
    :param path: path to checkpoint file.
    :param epoch: finished epoch.
    :param time: time training parameter.
    :param config: configuration.
    :param modules: an iterable of modules (e.g. models, optimizers) to save.
    :param kwargs: additional data to save in the checkpoint dictionary.
    """
    data = {
        'epoch': epoch,
        'time': time,
        'config': config,
        'modules': [m.state_dict() for m in modules]
    }
    for k, v in kwargs.items():
        data[k] = v
    torch.save(data, path)


def create_object(full_class_name, *args, **kwargs):
    """
    Create an object specified by class name and arguments.
    :param full_class_name: full class name. Either a string like 'package.subpackage.module.Class' or
    a tuple ('package.subpackage.module', 'Class.Subclass').
    :return: object instance.
    """
    if type(full_class_name) == str:
        parts = full_class_name.split('.')
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
    else:
        module_name = full_class_name[0]
        class_name = full_class_name[1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    obj = cls(*args, **kwargs)
    return obj


def count_parameters(obj):
    """
    Count parameters in the optimizer or model.
    :param obj: optimizer or module.
    :return: the number of parameters.
    """
    def count_in_list(l):
        return sum([p.numel() for p in l])

    if isinstance(obj, torch.optim.Optimizer):
        count = 0
        for pg in obj.param_groups:
            count += count_in_list(pg['params'])
    else:
        count = count_in_list(obj.parameters())

    return count


def compute_parameter_checksum(obj):
    """
    Compute a simple checksum of all parameters in the optimizer or model.
    :param obj: optimizer or module.
    :return: the sum of all parameters.
    """
    def sum_in_list(l):
        return sum([p.abs().sum() for p in l])

    if isinstance(obj, torch.optim.Optimizer):
        s = 0
        for pg in obj.param_groups:
            s += sum_in_list(pg['params'])
    else:
        s = sum_in_list(obj.parameters())

    return s


def compute_grad_stats(obj):
    """
    Compute the mean and max abs grad for all parameters in the optimizer or model.
    :param obj: optimizer or module.
    :return: mean, max abs(grad).
    """

    class Stats:
        max_grad = 0
        sum_grad = 0
        count = 0

    def compute_for_list(l, stats):
        for p in l:
            if p.grad is not None:
                a = p.grad.abs()
                stats.max_grad = max(a.max(), stats.max_grad)
                stats.sum_grad += a.sum()
                stats.count += a.numel()

    stats = Stats()

    if isinstance(obj, torch.optim.Optimizer):
        for pg in obj.param_groups:
            compute_for_list(pg['params'], stats)
    else:
        compute_for_list(obj.parameters(), stats)

    mean_grad = 0 if stats.count == 0 else stats.sum_grad / stats.count

    return mean_grad, stats.max_grad


def max_abs_param(module):
    """
    Max of abs of the parameters of a module.
    :param module: module.
    :return: the max abs value.
    """
    m = 0
    for p in module.parameters():
        m = max(m, p.abs().max().item())
    return m


def draw_axes(image, r, tx, ty, size=100, thickness=1, colors=((0, 0, 255), (0, 255, 0), (255, 0, 0))):
    """
    Draws axes of the standard right-handed CS: x, y as on the screen, z axis looking away from the observer.
    :param image: image
    :param r: rotation matrix for post-multiplication.
    :param tx: translation x in pixels
    :param ty: translation y in pixels
    :param size: axis length as a number or a tuple
    :param colors: color for axes x, y, z.
    :param thickness: line thickness.
    """
    if type(size) in (int, float):
        size = (size, ) * 3
    axes = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
    origin = np.array((tx, ty, 0), dtype=np.float32)
    axes = np.dot(axes, r) * np.array(size).reshape(3, 1) + origin

    o = tuple(origin[:2].astype(int))
    # Draw z-axis last.
    for ai in range(3):
        a = tuple(axes[ai, :2].astype(int))
        cv2.line(image, o, a, colors[ai], thickness)

    return image


def draw_gaze_vector(image, gaze_vector, color=(1, 1, 0), size=100, thickness=1):
    """
    Draws a gaze vector on an image.
    :param image: image as numpy array.
    :param size: scaling factor
    :param color: color
    :param thickness: line thickness.
    """
    gaze_vector = gaze_vector * size
    center = image.shape[1] // 2, image.shape[0] // 2
    end = int(center[0] + gaze_vector[0]), int(center[1] + gaze_vector[1])
    half_color = tuple([c / 2 for c in color])
    cv2.circle(image, center, 3, half_color, thickness)
    cv2.line(image, center, end, color, thickness)


class VideoReader:
    """ Read video from a video file, camera, pictures, etc. """
    def __init__(self, video_path, roi=None, loop=False, flip=False):
        """
        Create object
        :param video_path: path to a video. Pass an integer number to open a camera by its ID.
        """
        self._video = None
        self._frames = None
        self._frame_index = 0
        self._loop = loop
        self._roi = roi
        self._video_path = video_path
        self._flip = flip
        if os.path.isdir(video_path):
            self._frames = glob.glob(video_path + '/**/*.png', recursive=True)
        elif os.path.splitext(video_path)[1].lower() in ['.png', '.jpg', '.jpeg']:
            self._frames = [video_path]
        else:
            if self._video_path.isnumeric():
                self._video_path = int(self._video_path)  # Camera id
                self._loop = False
            logger.info(f'Opening video capture {self._video_path} ...')
            self._video = cv2.VideoCapture(self._video_path)
            logger.info(f'Opened video capture {self._video_path}')

    @property
    def frame_index(self):
        """ Returns the index of the next frame returned by read_frame(). In loop mode will jump down skipping 0. """
        return self._frame_index

    def read_frame(self):
        """
        Read a frame.
        :return: an image in BGR format, or None in case of end of file, etc.
        """
        if self._frames is not None:
            if self._frame_index >= len(self._frames):
                if not self._loop:
                    return None
                self._frame_index = 0
            frame_path = self._frames[self._frame_index]
            self._frame_index += 1
            frame = cv2.imread(frame_path)
        else:
            ret, frame = self._video.read()
            self._frame_index += 1
            if not ret and self._loop:
                self._video = cv2.VideoCapture(self._video_path)
                ret, frame = self._video.read()
                self._frame_index = 1
            if not ret:
                frame = None

        if self._flip:
            frame = np.ascontiguousarray(np.flip(frame, axis=1))

        if self._roi is not None:
            frame = frame[self._roi[1]:self._roi[1]+self._roi[3], self._roi[0]:self._roi[0]+self._roi[2]]

        return frame


def rand_uniform(b, e, shape, device):
    """ Generate uniform random numbers in range [b, e). """
    return torch.rand(shape, device=device) * (e - b) + b


def interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    """
    A wrapper for torch.functional.interpolate() with convenient default parameter settings
    without generating warnings.
    """
    if size == input.shape[-2:] and scale_factor is None:
        return input
    if scale_factor == 1 and size is None:
        return input

    if mode in ("nearest", "area"):
        align_corners = None
    if size is None:
        recompute_scale_factor = False
    else:
        recompute_scale_factor = None
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                         recompute_scale_factor=recompute_scale_factor)


def almost_equal(x, y, eps=1e-5):
    return (x - y).abs().max() < eps


def make_rng_seed(base_seed, epoch, rank):
    """
    Make an RNG seed. It changes with epoch to ensure reproducibility after a checkpoint restart.
    :param base_seed: seed from the configuration.
    :param epoch: epoch.
    :param rank: DDP process rank.
    :return: new RNG seed.
    """
    num_epochs = 1024
    num_ranks = 64
    num_dataloader_workers = 64
    seed = base_seed * num_epochs * num_ranks * num_dataloader_workers + \
           epoch * num_ranks * num_dataloader_workers + \
           rank * num_dataloader_workers

    return seed


def make_clean_directory(path):
    """
    Creates an empty directory.

    If it exists, delete its content.
    If the directory is opened in Windows Explorer, may throw PermissionError,
    although the directory is usually cleaned. The caller may catch this exception to avoid program termination.
    :param path: path to the directory.
    """
    need_create = True
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        need_create = False
    elif os.path.isfile(path):
        os.remove(path)
    if need_create:
        os.makedirs(path)