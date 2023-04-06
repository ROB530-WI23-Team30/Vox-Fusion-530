#!/usr/bin/env python3
import os.path as osp

import numpy as np
from glob import glob
from torch.utils.data import Dataset
import cv2
import torch


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False, max_depth=-1) -> None:
        self.data_path = data_path
        self.num_imgs = len(glob(osp.join(data_path, "rgb", "*.png")))
        self.use_gt = use_gt
        self.max_depth = max_depth
        self.K = self.load_intrinsic()
        self.gt_pose = self.load_gt_pose()

    def get_init_pose(self):
        if self.gt_pose is not None:
            return self.gt_pose[0, ...].reshape(4, 4)
        else:
            return np.eye(4)

    def load_intrinsic(self):
        return np.array([[481.2, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]])

    def load_gt_pose(self):
        gt_file = osp.join(self.data_path, "groundtruth.txt")
        pose_data = np.loadtxt(gt_file, delimiter=" ", dtype=np.unicode_)
        pose_vecs = pose_data[:, 1:]
        poses = [
            DataLoader.pose_matrix_from_quaternion(p).reshape(-1) for p in pose_vecs
        ]
        assert len(poses) == self.num_imgs, "number of poses and images mismatch"
        poses = np.array(poses, dtype=np.float64)
        return poses

    @staticmethod
    def pose_matrix_from_quaternion(pvec: np.ndarray) -> np.ndarray:
        """convert 4x4 pose matrix to (t, q)
        :param pvec: 7-dim vector (x, y, z, qx, qy, qz, qw)
        :return: 4x4 pose matrix
        """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def load_image(self, index):
        rgb = cv2.imread(osp.join(self.data_path, "rgb", "{:04d}.png".format(index)))
        depth: np.ndarray = cv2.imread(
            osp.join(self.data_path, "depth", "{:04d}.png".format(index)), -1
        )
        depth = depth / 5000.0
        # if self.max_depth > 0:
        #     depth[depth > self.max_depth] = 0
        return (rgb / 255.0), depth

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        img, depth = self.load_image(index)
        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        pose = self.gt_pose[index] if self.use_gt else None
        return index, img, depth, self.K, pose


if __name__ == "__main__":
    import sys

    loader = DataLoader(sys.argv[1])
    num_img = len(loader)
    print(f"loader length: {len(loader)}")
    for i in range(num_img):
        index, img, depth, K, _ = loader[i]
        print(K)
        print(index, img.shape)
        print(index, depth.shape)
        cv2.imshow("img", img.numpy())
        cv2.imshow("depth", depth.numpy())
        print(f"depth: {depth.reshape(-1)[:10]}")
        cv2.waitKey(1)
