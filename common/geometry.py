""" Geometrical and coordinate transforms, rotations, etc. """

import math

import numpy as np
from scipy.spatial import transform
import torch
from torch.nn import functional as F


def make_transform2(scale=1, angle=0, tx=0, ty=0):
    """
    Rotate, scale (about origin) and then translate 2d points.
    :return a 3x3 matrix transformation matrix.
    """
    ca = np.cos(angle) * scale
    sa = np.sin(angle) * scale
    t = np.array([ca, -sa, tx, sa, ca, ty, 0, 0, 1], dtype=np.float64).reshape(3, 3)
    return t


def angles_to_r3(ax, ay, az):
    """
    Make a 3D rotation matrices for post-multiplication from angles.
    See https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations

    All input tensors are in batches.

    :param ax: rotation angle around x-axis in radians.
    :param ay: rotation angle around y-axis in radians.
    :param az: rotation angle around z-axis in radians.
    :return: rx, ry, rz
    """
    if ax.ndim != 2 or ax.shape[1] != 1:
        raise ValueError(f'Wrong ax shape, expected (b, 1), got {ax.shape}')
    if ay.ndim != 2 or ay.shape[1] != 1:
        raise ValueError(f'Wrong ax shape, expected (b, 1), got {ay.shape}')
    if az.ndim != 2 or az.shape[1] != 1:
        raise ValueError(f'Wrong ax shape, expected (b, 1), got {az.shape}')

    b = ax.shape[0]
    device = ax.device
    rx = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
    ry = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
    rz = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)

    ax = ax.squeeze(-1)
    ay = ay.squeeze(-1)
    az = az.squeeze(-1)

    rx[:, 1, 1] = torch.cos(ax)
    rx[:, 1, 2] = -torch.sin(ax)
    rx[:, 2, 1] = torch.sin(ax)
    rx[:, 2, 2] = torch.cos(ax)

    ry[:, 0, 0] = torch.cos(ay)
    ry[:, 0, 2] = torch.sin(ay)
    ry[:, 2, 0] = -torch.sin(ay)
    ry[:, 2, 2] = torch.cos(ay)

    rz[:, 0, 0] = torch.cos(az)
    rz[:, 0, 1] = -torch.sin(az)
    rz[:, 1, 0] = torch.sin(az)
    rz[:, 1, 1] = torch.cos(az)

    # Transpose for post-multiplication
    return rx.transpose(1, 2), ry.transpose(1, 2), rz.transpose(1, 2)


def h3(r=None, t=None, s=None):
    """
    Make a 3D homogeneous transform matrix for post-multiplication:

    R        0
             0
             0
    tx ty tz 1

    All arguments must have a batch dimension.
    :param: s: (B, 1) matrix for uniform scaling.
    :param: r: is one of the following:
        - (B, 3, 3) rotation matrix, may also scale, shear, etc.
        - (B, 6) r6 matrix. If scale is None, r6 is converted to scale and rotation, otherwise to rotation only.
    :param: t: if one of the following:
        - (B, 2) or (B, 1, 2) 2D tx, ty translation (tz=0 is implied)
        - (B, 3) or (B, 1, 3) 3D tx, ty, tz translation.
    :return: a (B, 4, 4) matrix.
    """
    b = 1
    if s is not None:
        b = max(b, s.shape[0])
        device = s.device
    elif r is not None:
        b = max(b, r.shape[0])
        device = r.device
    elif t is not None:
        b = max(b, t.shape[0])
        device = t.device
    else:
        raise ValueError('At least one of the arguments must be specified.')

    h = torch.eye(4, device=device).repeat(b, 1, 1)
    if r is not None:
        if r.ndim == 2:
            if r.shape[1] != 6:
                raise ValueError(f'Expected (B, 6) r6 matrix, got {r.shape}')
            if s is None:
                # Scale is not specified, extract from r6
                s = r6_to_s(r)
            r = r6_to_r(r)
        elif r.ndim != 3:
            raise ValueError(f'Expected (B, 3, 3) rotation matrix, got {r.shape}')
        h[:, :3, :3] = r

    if t is not None:
        if t.ndim == 3:
            if t.shape[1] == 1:
                t = t.squeeze(1)
            else:
                raise ValueError(f'Expected (B, 1, 2) or (B, 1, 3) translation, got {t.shape}')
        elif t.ndim != 2:
            raise ValueError(f'Expected (B, 2) or (B, 3) translation, got {t.shape}')

        if t.shape[1] == 2:
            h[:, 3, :2] = t
        elif t.shape[1] == 3:
            h[:, 3, :3] = t
        else:
            raise ValueError(f'Expected (B, 2) or (B, 3) translation, got {t.shape}')
    if s is not None:
        h[:, :3, :3] *= s.unsqueeze(-1)

    return h


def make_coordinate_grid2(t, device=None, align_corners=False):
    """
    Create a 2d coordinate grid [-1,1] x [-1,1] of the size given by x.
    :param t: input tensor of size (..., H, W) or a tuple (H, W).
    :param align_corners: if True, the point (-1, -1) of the normalized CS in in the center of the pixel (0, 0),
    otherwise in its top left corner. See also:
        torch.nn.functional.grid_sample()
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
    :return a tensor (H, W, 2) with x, y coordinates.
    """

    if type(t) == torch.Tensor:
        h, w = t.shape[-2:]
        device = t.device
    else:
        h, w = t

    x, y = torch.meshgrid(
        torch.arange(float(w), device=device),
        torch.arange(float(h), device=device), indexing='xy')

    lh = 1 if align_corners else 1 - 1 / h
    lw = 1 if align_corners else 1 - 1 / w

    x = lw * (2 * (x / (w - 1)) - 1).unsqueeze_(2)
    y = lh * (2 * (y / (h - 1)) - 1).unsqueeze_(2)

    xy = torch.cat((x, y), 2)

    return xy


def rotation_diff(r1, r2, degrees=False):
    """
    Computes the absolute angular difference between two rotation matrices d = |r1 - r2|.
    :param r1: rotation matrix 1 (3, 3) or (B, 3, 3) as numpy array
    :param r2: rotation matrix 2 (3, 3) or (B, 3, 3) as numpy array
    :param degrees: if True, the angles are in degrees, otherwise in radians.
    :return: angular difference
    """
    dr = np.matmul(r1, np.linalg.inv(r2))
    da = np.linalg.norm(transform.Rotation.from_matrix(dr).as_rotvec(), axis=-1)
    if degrees:
        da = np.rad2deg(da)

    return da


def r_to_r6(r):
    """
    Converts rotation matrices to 6D. If the rotation matrix is scaled, the r6 will also be scaled.

    The r6 representation consists of the first 2 columns of the rotation matrix:
    r = [Xx Xy Xz]
        [Yx Yy Yz]
        [Zx Zy Zz]
    r6 = [Xx Xy Yx Yy Zx Zy]

    The rows of the rotation matrix are the rotated coordinate axis. This matrix is to post-multiply vectors.
    E.g. for X axis: [1 0 0] r = [Xx Xy Xz].

    See also: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d
    :param r: a tensor (..., 3, 3).
    :return r6 representation (..., 6) of the rotation and scale.
    """
    return r[..., :, :2].clone().reshape(*r.size()[:-2], 6)


def r6_to_r(r6):
    """
    Converts 6D rotation representation  to rotation matrix.
    :param r6: a tensor (..., 6): 6D rotation representation (first two rows of a rotation matrix).
    :return rotation matrix.
    """
    a1, a2 = r6[..., 0::2], r6[..., 1::2]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def r6_to_s(r6):
    """
    Extract scale from a 6D rotation representation.
    :param r6: a tensor (..., 6): 6D rotation representation (first two rows of a rotation matrix).
    :return scale factor (..., 1).
    """
    # A uniform handling of all r6 elements to extract the scale.
    return torch.linalg.norm(r6, dim=-1, keepdim=True) / math.sqrt(2)


def pixel_to_norm2(pixel_xy, hw, align_corners=False):
    """
    Convert 2D pixel coordinates to normalized coordinates in range [-1, 1].
    Attention: non-uniform scaling if h != w.
    :param pixel_xy: pixel coordinates [..., 2] of x, y points.
    :param hw: (H, W) sizes
    :param align_corners: if True, the point (-1, -1) of the normalized CS in in the center of the pixel (0, 0),
    otherwise in its top left corner. See also:
        torch.nn.functional.grid_sample()
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
    :return: a tensor [..., 2] with normalized coordinates.
    """
    if len(hw) != 2:
        raise ValueError(f'Expected spatial size (H, W), got {hw}')
    hw = pixel_xy.new_tensor(hw[-1::-1])
    if align_corners:
        n = pixel_xy / (hw - 1) * 2 - 1
    else:
        n = (2 * pixel_xy + 1) / hw - 1

    return n


def norm_to_pixel2(norm_xy, hw, align_corners=False):
    """
    Convert 2D normalized coordinates in range [-1, 1] to pixel coordinates.
    Attention: non-uniform scaling if h != w.
    :param norm_xy: a tensor [..., 2] with normalized x, y coordinates.
    :param hw: (H, W) sizes
    :param align_corners: if True, the point (-1, -1) of the normalized CS in in the center of the pixel (0, 0),
    otherwise in its top left corner. See also:
        torch.nn.functional.grid_sample()
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
    :return: a tensor [..., 2] with pixel coordinates.
    """
    if len(hw) != 2:
        raise ValueError(f'Expected spatial size (H, W), got {hw}')
    hw = norm_xy.new_tensor(hw[-1::-1])
    if align_corners:
        p = (norm_xy + 1) / 2 * (hw - 1)
    else:
        p = ((norm_xy + 1) * hw - 1) / 2
    return p


def line_line_intersection3(p0, p1):
    """
    For n lines passing through points p0, p1 find the intersection point (if exists)
    or the point closest to all the lines.
    :param p0: (n, 2) numpy array of points
    :param p1: (n, 2) numpy array of points
    :return: a tuple (closest_point, projections), where
    - closest_point is a (1, 3) closest point
    - projections is a (n, 3) array of the projection of the closest point to the lines.

    See: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_three_dimensions
    """
    n = p0.shape[0]
    d = p1 - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    # Compute a normal vector to d by zeroing out the smallest of x, y, z coordinates and swapping the other two, e.g.
    # (1, 2, 3) -> (0, -3, 2)
    min_c = np.argmin(np.abs(d), axis=1)
    swap_c = np.array((0, 1, 2), dtype=np.int64)
    swap_c = np.tile(swap_c, (n, 1))
    swap_c = swap_c[swap_c - min_c.reshape(n, 1) != 0].reshape(n, 2)

    n1 = d.copy()
    i0 = np.arange(n)
    n1[i0, min_c] = 0
    n1[i0, swap_c[:, 0]] = -d[i0, swap_c[:, 1]]
    n1[i0, swap_c[:, 1]] = d[i0, swap_c[:, 0]]
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    assert (n1 * d).sum() < 1e-10

    # Compute another normal vector by a cross product.
    n2 = np.cross(n1, d)
    assert (n2 * d).sum() < 1e-10

    A = np.concatenate((n1, n2), axis=0)
    b = (A * np.tile(p0, (2, 1))).sum(1)

    closest_point = np.linalg.lstsq(A, b, rcond=None)[0].reshape(1, 3)
    projections = (d * (closest_point - p0)).sum(1, keepdims=True) * d + p0

    return closest_point, projections


def gaze2_to_3(gaze2, z_factor=1):
    """
    Converts 2D gaze vector (x, y) to a 3D gaze vector (x, y, z)
    :param gaze2: a tensor (B, 2) of 2D gaze vector
    :param z_factor a multiplier for z-coordinate, usually 1 or -1.
    :return: a tensor (B, 3) of 3D gaze vectors, each vector's norm is one.
    """
    s = torch.sum(gaze2**2, -1, keepdim=True)
    # As gaze2 can be a prediction of a DNN, s can be > 1. Therefore normalize it to avoid NAN in sqrt().
    s = s.clamp(max=1)
    z = (1 - s).sqrt() * z_factor
    gaze3 = torch.cat((gaze2, z), -1)
    return gaze3


def inverse_rt(transf):
    """
    Inverts a rotation and translation matrix using the fact that r.inverse() == r.transpose().
    if transf consist of r, t, then:
    x1 = x @ r + t
    inverse transform: r.T, - t @ r.T
    proof:
    x1 @ r.T - t @ r.T = (x @ r + t) @ r.T - t @ r.T = x @ r @ r.T  + t @ r.T - t @ r.T = x @ I + 0 = x.

    :param transf: a (B, 4, 4) matrix containing rotation and translation.
    :return: the inverse of the transf.
    """
    inv = torch.zeros_like(transf)
    inv_r = transf[:, :3, :3].transpose(1, 2)
    inv[:, :3, :3] = inv_r
    inv[:, 3:, :3] = -transf[:, 3:, :3] @ inv_r
    inv[:, 3, 3] = 1
    return inv