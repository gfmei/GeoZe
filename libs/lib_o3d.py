import open3d as o3d
import numpy as np
import torch


def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.0
    return redirected_normals


def to_o3d_pcd(pcd, normal=None, radius=-1.0):
    if torch.is_tensor(pcd):
        pcd = pcd.cpu().numpy()
    if torch.is_tensor(normal):
        normal = normal.cpu().numpy()
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    if radius > 0 and normal is None:
        pcd_.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=33))
    if normal is not None:
        pcd_.normals = o3d.utility.Vector3dVector(normal)
    return pcd_


def estimate_normals(points, radius=0.05, k=33):
    pcd = to_o3d_pcd(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k))
    point_normals = np.asarray(pcd.normals)
    # view_point = points.mean(axis=0)
    # point_normals = normal_redirect(points, point_normals, view_point)
    return point_normals


def estimate_geo_feature(points, normals=None, voxel_size=0.05):
    radius_normal = voxel_size
    pcd = to_o3d_pcd(points, normals, radius=radius_normal)
    radius_feature = 2.5*voxel_size
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return np.asarray(pcd.normals), np.asarray(pcd_fpfh.data).T


def batch_geo_feature(tensors, voxel_size=0.05):
    bs = len(tensors)
    features, normals = [], []
    for i in range(bs):
        tensor = tensors[i]
        normal, feature = estimate_geo_feature(tensor, voxel_size=voxel_size)
        normals.append(torch.from_numpy(normal).to(tensors))
        features.append(torch.from_numpy(feature).to(tensors))
    return torch.stack(normals), torch.stack(features, dim=0)
