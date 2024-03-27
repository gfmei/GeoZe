# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import collections
from collections.abc import Sequence

import numpy as np
import torch
from scipy.linalg import expm, norm


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def fnv_hash_vec(arr):
    '''
    FNV64-1A
    '''
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    '''
    Ravel the coordinates after subtracting the min coordinates.
    '''
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    set_ignore_label_when_collision=False,
                    return_index=False,
                    hash_type='fnv',
                    quantization_size=1):
    r'''Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.
    Args:
        coords (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a matrix of size
        :math:`N \times D` where :math:`N` is the number of points in the
        :math:`D` dimensional space.

        feats (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a matrix of size
        :math:`N \times D_F` where :math:`N` is the number of points and
        :math:`D_F` is the dimension of the features.

        labels (:attr:`numpy.ndarray`, optional): labels associated to eah coordinates.
        ignore_label (:attr:`int`, optional): the int value of the IGNORE LABEL.
        set_ignore_label_when_collision (:attr:`bool`, optional): use the `ignore_label`
        when at least two points fall into the same cell.
        return_index (:attr:`bool`, optional): True if you want the indices of the
        quantized coordinates. False by default.

        hash_type (:attr:`str`, optional): Hash function used for quantization. Either
        `ravel` or `fnv`. `ravel` by default.
        quantization_size (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyper_rectangle of the grid cell.
    note:
        Please check `examples/indoor.py` for the usage.
    '''
    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert hash_type in [
        'ravel', 'fnv'
    ], "Invalid hash_type. Either ravel, or fnv allowed. You put hash_type=" + hash_type
    assert coords.ndim == 2, "The coordinates must be a 2D matrix. The shape of the input is " + str(coords.shape)
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(
            quantization_size
        ) == dimension, "Quantization size and coordinates size mismatch."
        quantization_size = [i for i in quantization_size]
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [quantization_size for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coords)
    else:
        key = fnv_hash_vec(discrete_coords)

    if use_label:
        _, inds, counts = np.unique(key, return_index=True, return_counts=True)
        filtered_labels = labels[inds]
        if set_ignore_label_when_collision:
            filtered_labels[counts > 1] = ignore_label
        if return_index:
            return inds, filtered_labels
        else:
            return discrete_coords[inds], feats[inds], filtered_labels
    else:
        _, inds, inds_reverse = np.unique(key, return_index=True, return_inverse=True)
        if return_index:
            return inds, inds_reverse
        else:
            if use_feat:
                return discrete_coords[inds], feats[inds]
            else:
                return discrete_coords[inds]


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer(object):

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        '''
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          scale_augmentation_bound: None or (0.9, 1.1)
          rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
          translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.abc.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        # rigid_transformation = M_t @ rigid_transformation
        coords_aug = np.floor(coords_aug - min_coords)

        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds]

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]

        return coords_aug, feats, labels, np.array(inds_reconstruct)


def collation_fn(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)
    '''
    coords, feats, labels = list(zip(*batch))
    for i, coord in enumerate(coords):
        coord[:, 0] *= i
    return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)
    '''
    coords, xyz, feats, labels, feat_3d, spts, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    accmulate_points_num = 0
    for i, coord in enumerate(coords):
        coord[:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return [torch.cat(coords), torch.cat(xyz), torch.cat(feats), torch.cat(labels), torch.cat(feat_3d), torch.cat(spts),
            torch.cat(mask), torch.cat(inds_recons)]


def num_to_natural(group_ids, void_number=-1):
    """
    Reassigns non-void group IDs in the input array to a continuous sequence of natural numbers.
    Keeps the void_number unchanged. The new sequence starts from 0 if void_number is -1,
    or from 1 if void_number is 0.

    Parameters:
    - group_ids: np.array, original array of group IDs.
    - void_number: int, the value in the array to be ignored. Should be either -1 or 0.

    Returns:
    - np.array with renumbered group IDs.

    Raises:
    - Exception if void_number is not -1 or 0.
    """
    if void_number not in [-1, 0]:
        raise Exception("void_number must be -1 or 0")

    if np.all(group_ids == void_number):
        return group_ids

    array = group_ids.copy()
    unique_values = np.unique(array[array != void_number])

    # Start mapping from 0 or 1 based on void_number.
    start_mapping = 0 if void_number == -1 else 1
    mapping = {val: i + start_mapping for i, val in enumerate(unique_values)}

    # Apply mapping, skipping the void_number.
    for original, new in mapping.items():
        array[array == original] = new

    return array
