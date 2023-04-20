"""
Evaluate tracking performance

Usage:
python eval_track.py /path/to/pose /path/to/gt_pose
"""
import numpy as np
import sys


def align(rec_xyz, gt_xyz):
    # align the two point clouds, using the method of Horn (closed-form)
    # rec_xyz: N x 3
    # gt_xyz: N x 3

    # compute centroids
    rec_centroid = np.mean(rec_xyz, axis=0)
    gt_centroid = np.mean(gt_xyz, axis=0)

    # subtract centroids
    rec_xyz = rec_xyz - rec_centroid
    gt_xyz = gt_xyz - gt_centroid

    # compute covariance matrix
    H = np.matmul(rec_xyz.T, gt_xyz)

    # compute SVD
    U, S, Vt = np.linalg.svd(H)

    # compute rotation matrix
    R = np.matmul(Vt.T, U.T)

    # compute translation
    t = gt_centroid - np.matmul(R, rec_centroid)

    # compute alignment error
    rec_xyz_aligned = np.matmul(rec_xyz, R) + t
    error = np.linalg.norm(rec_xyz_aligned - gt_xyz, axis=1)

    return R, t, error


def plot_traj(ax, traj, style, color, label):
    x = traj[:, 0]
    y = traj[:, 1]
    ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(poses_dir_rec, poses_dir_gt, if_verbose=True, if_plot=True):
    rec_xyz = poses_dir_rec[:, :3, 3] / poses_dir_gt[0, 3, 3]
    gt_xyz = poses_dir_gt[:, :3, 3]
    rec_xyz = rec_xyz.reshape(-1, 3)
    gt_xyz = gt_xyz.reshape(-1, 3)

    R, t, error = align(rec_xyz, gt_xyz)
    print("R: ", R)
    print("t: ", t)

    rec_xyz_aligned = np.matmul(rec_xyz, R) + t
    t = rec_xyz[0, :] - gt_xyz[0, :]
    rec_xyz_aligned = rec_xyz - t
    if if_verbose:
        # compute tran_err in verbose mode including max, min, mean, median, std, rmse
        tran_err = np.linalg.norm(rec_xyz_aligned - gt_xyz, axis=1)
        print("ATE: ", np.mean(tran_err))
        print("Max: ", np.max(tran_err))
        print("Min: ", np.min(tran_err))
        print("Median: ", np.median(tran_err))
        print("Std: ", np.std(tran_err))
        print("RMSE: ", np.sqrt(np.mean(tran_err**2)))

    if if_plot:
        # plot the trajectory
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(ax, rec_xyz_aligned, "-", "r", "rec")
        plot_traj(ax, gt_xyz, "-", "b", "gt")
        ax.legend()
        plt.savefig("traj.png")


if __name__ == "__main__":
    poses_dir_rec = sys.argv[0]
    poses_dir_gt = sys.argv[1]
    poses_rec = np.load(poses_dir_rec)
    poses_gt = np.loadtxt(poses_dir_gt)

    poses_gt = poses_gt.reshape(-1, 4, 4)
    evaluate_ate(poses_rec, poses_gt)
