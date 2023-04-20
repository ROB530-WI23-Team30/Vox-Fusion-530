"""
Evaluation on mesh reconstruction quality

Usage:
python eval_mesh.py --rec_file /path/to/mesh.ply --gt_file /path/to/gt.ply
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import argparse


def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def get_align_transformation(gt_point_cloud, predicted_point_cloud):
    trans_init = np.eye(4)
    threshold = 0.1

    reg_p2p = o3d.pipelines.registration.registration_icp(
        gt_point_cloud,
        predicted_point_cloud,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation = reg_p2p.transformation
    return transformation


def chamfer_distance(p1, p2):
    gen_points_kd_tree = KDTree(p1)
    p1_to_p2, _ = gen_points_kd_tree.query(p2)
    gen_points_kd_tree = KDTree(p2)
    p2_to_p1, _ = gen_points_kd_tree.query(p1)
    return np.mean(p1_to_p2) + np.mean(p2_to_p1)


def f_score(p1, p2, threshold):
    gen_points_kd_tree = KDTree(p1)
    p1_to_p2, _ = gen_points_kd_tree.query(p2)
    gen_points_kd_tree = KDTree(p2)
    p2_to_p1, _ = gen_points_kd_tree.query(p1)
    precision = np.sum(np.array(p1_to_p2) < threshold) / len(p1)
    recall = np.sum(np.array(p2_to_p1) < threshold) / len(p2)

    if precision + recall == 0:
        return 0

    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


def completion(gt_points, predicted_points):
    gt_points_kd_tree = KDTree(predicted_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def completion_ratio(gt_points, predicted_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(predicted_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the reconstructed point cloud"
    )
    # Example usage:
    # gt_file = "Datasets/Replica/office4_mesh.ply"
    # rec_file = "output/Replica/office4/mesh/final_mesh_eval_rec.ply"
    parser.add_argument(
        "--rec_file", type=str, help="Path to the reconstructed point cloud"
    )
    parser.add_argument(
        "--gt_file", type=str, help="Path to the ground truth point cloud"
    )
    args = parser.parse_args()

    # Import the ground truth and predicted point clouds
    ground_truth_points = load_ply(args.gt_file)
    predicted_points = load_ply(args.rec_file)
    # Get the transformation matrix to align the predicted point cloud
    # to the ground truth point cloud
    transformation = get_align_transformation(ground_truth_points, predicted_points)
    predicted_points.transform(transformation)

    ground_truth_points = np.asarray(ground_truth_points.points)
    predicted_points = np.asarray(predicted_points.points)

    # Calculate the chamfer distance and f-score
    threshold = 0.1
    cd = chamfer_distance(ground_truth_points, predicted_points)
    fs = f_score(ground_truth_points, predicted_points, threshold)

    print("Chamfer Distance: ", cd)
    print("F-Score: ", fs)

    comp = completion(ground_truth_points, predicted_points)
    comp_ratio = completion_ratio(ground_truth_points, predicted_points)
    print("Completion: ", comp)
    print("Completion Ratio: ", comp_ratio)
