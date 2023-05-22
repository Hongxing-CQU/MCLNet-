import math
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from common.math.random import uniform_2_sphere
import common.math.se3 as se3
import common.math.so3 as so3


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class RandomCrop:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample


class RandomTransformSE3_euler:
    def __init__(self, rot_mag=45.0, trans_mag=0.5):
        self.angle = rot_mag
        self.trans = trans_mag

    def generate_transform(self):
        anglex = np.random.uniform() * np.pi * self.angle / 180.
        angley = np.random.uniform() * np.pi * self.angle / 180.
        anglez = np.random.uniform() * np.pi * self.angle / 180.

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R = Rx.dot(Ry).dot(Rz)
        t = np.random.uniform(-self.trans, self.trans, size=3)
        euler = np.array([anglez, angley, anglex])

        return R, t, euler

    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        R, t, euler = self.generate_transform()

        T_gt = np.concatenate([R, t[:, None]], axis=-1)
        T_gt = np.concatenate([T_gt, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)

        sample['T'] = T_gt
        sample['euler'] = euler


        if 'points' in sample:
            sample['points'][:, 0:3] = sample['points'][:, 0:3] @ R.T + t[None, :]
        else:
            sample['points_ref'][:, 0:3] = sample['points_ref'][:, 0:3] @ R.T + t[None, :]
            # transform normals
            if sample['points_ref'].shape == 6:
                sample['points_ref'][:, 3:6] = sample['points_ref'][:, 3:6] @ R.T

        return sample


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            sample['points_ref'] = np.random.permutation(sample['points_ref'])
            sample['points_src'] = np.random.permutation(sample['points_src'])
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class Dict2DcpList:
    """Converts dictionary of tensors into a list of tensors compatible with Deep Closest Point"""
    def __call__(self, sample):

        target = sample['points_src'][:, :3].transpose().copy()
        src = sample['points_ref'][:, :3].transpose().copy()

        rotation_ab = sample['transform_gt'][:3, :3].transpose().copy()
        translation_ab = -rotation_ab @ sample['transform_gt'][:3, 3].copy()

        rotation_ba = sample['transform_gt'][:3, :3].copy()
        translation_ba = sample['transform_gt'][:3, 3].copy()

        euler_ab = Rotation.from_dcm(rotation_ab).as_euler('zyx').copy()
        euler_ba = Rotation.from_dcm(rotation_ba).as_euler('xyz').copy()

        return src, target, \
               rotation_ab, translation_ab, rotation_ba, translation_ba, \
               euler_ab, euler_ba


class Dict2PointnetLKList:
    """Converts dictionary of tensors into a list of tensors compatible with PointNet LK"""
    def __call__(self, sample):

        if 'points' in sample:
            # Train Classifier (pretraining)
            return sample['points'][:, :3], sample['label']
        else:
            # Train PointNetLK
            transform_gt_4x4 = np.concatenate([sample['transform_gt'],
                                               np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
            return sample['points_src'][:, :3], sample['points_ref'][:, :3], transform_gt_4x4



def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "crop":
        train_transforms = [Resampler(num_points),
                            SplitSourceRef(),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            RandomCrop(partial_p_keep),
                            ShufflePoints()]

        test_transforms = [Resampler(num_points),
                           SplitSourceRef(),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           RandomCrop(partial_p_keep),
                           ShufflePoints()]

    elif noise_type == "noise_crop":
        train_transforms = [Resampler(num_points),
                            SplitSourceRef(),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            RandomCrop(partial_p_keep),
                            RandomJitter(),
                            ShufflePoints()]

        test_transforms = [Resampler(num_points),
                           SplitSourceRef(),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           RandomCrop(partial_p_keep),
                           RandomJitter(),
                           ShufflePoints()]

    elif noise_type == "crop_plus":
        train_transforms = [SplitSourceRef(),
                            RandomCrop(partial_p_keep),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Resampler(num_points),
                            ShufflePoints()]

        test_transforms = [SplitSourceRef(),
                           RandomCrop(partial_p_keep),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Resampler(num_points),
                           ShufflePoints()]
    elif noise_type == "noise_crop_plus":
        train_transforms = [SplitSourceRef(),
                            RandomCrop(partial_p_keep),
                            RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Resampler(num_points),
                            RandomJitter(),
                            ShufflePoints()]

        test_transforms = [SplitSourceRef(),
                           RandomCrop(partial_p_keep),
                           RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Resampler(num_points),
                           RandomJitter(),
                           ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms










