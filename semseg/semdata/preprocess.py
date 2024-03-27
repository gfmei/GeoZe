import copy
import glob
import open3d as o3d
import os
import sys

import plyfile
import torch
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('./'))
from libs.lib_o3d import estimate_geo_feature
from libs.lib_vis import creat_labeled_point_cloud
import segmentator
from semseg.utils.data_utils import vertex_normal

def num_to_natural(group_ids, void_number=-1):
    """
    code credit: SAM3D
    """
    if (void_number == -1):
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        # map ith(start from 0) group_id to i
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif (void_number == 0):
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")

    return array


CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '.aggregation.json'

# Map relevant classes to {0,1,...,19}, and ignored classes to 255
remapper = np.ones(150) * (255)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        return vertices, faces


def point_indices_from_group(seg_indices, group):
    group_segments = np.array(group['segments'])
    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return point_idx


def process_one_scene_scannet(fn, out_dir, voxel_size=0.05):
    '''Process one ScanNet scene and save processed data as a .pth file.'''
    label_fn = fn[:-3] + 'labels.ply'
    vertices, faces = read_plymesh(fn)
    # seg_indices = segmentator.segment_mesh(torch.from_numpy(vertices.astype(np.float32)),
    #                                        torch.from_numpy(faces).long(), kThresh=0.01, segMinVerts=200).numpy()
    coords = vertices[:, :3]
    colors = vertices[:, 3:6] / 127.5 - 1
    labels_data = plyfile.PlyData().read(label_fn)
    labels = remapper[np.array(labels_data.elements[0]['label'])]
    normals = vertex_normal(coords, faces)
    print('estimate the fpfh features')
    _, fpfhs = estimate_geo_feature(coords, normals, voxel_size=voxel_size)
    print('estimate the fpfh features end')
    output_file_name = os.path.basename(fn)[:-4] + '.pth'
    # segments_file = copy.deepcopy(fn).replace('.ply', SEGMENTS_FILE_PFIX)
    # Load Aggregations file
    # with open(aggregations_file) as f:
    #     aggregation = json.load(f)
    #     seg_groups = np.array(aggregation['segGroups'])
    mesh_file = fn
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    _vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    _faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    seg_indices = segmentator.segment_mesh(_vertices, _faces, kThresh=0.02, segMinVerts=50).numpy()
    # creat_labeled_point_cloud(coords, seg_indices, 'pred')
    # print(seg_indices.shape, normals.shape)
    # 0 / 0
    save_dict = {'coord': coords, 'color': colors, 'label': labels, 'normal': normals, 'fpfh': fpfhs, 'spt_ids': seg_indices}
    output_file_path = os.path.join(out_dir, output_file_name)
    torch.save(save_dict, output_file_path)
    print(fn, label_fn)
    print('finished one scene')


def process_txt(filename):
    '''Read lines from a text file and strip newlines.'''
    with open(filename) as file:
        lines = [line.rstrip() for line in file.readlines()]
    return lines


def prepare_and_process_scenes(split='val', out_dir_base='/data/disk1/data/scannet/scannet_3d_new',
                               in_path_base='/data/disk1/data/scannet/scans'):
    '''Prepare and process ScanNet scenes based on a split.'''
    out_dir = os.path.join(out_dir_base, split)
    in_path = in_path_base
    scene_list = process_txt(f'../meta/scannet/scannetv2_{split}.txt')

    os.makedirs(out_dir, exist_ok=True)

    files = [glob.glob(os.path.join(in_path, scene, '*_vh_clean_2.ply'))[0] for scene in scene_list]
    assert all(os.path.exists(f) for f in files), "Some scene files could not be found."

    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     # Load scene paths
    #     print('Processing scenes...')
    #     _ = list(executor.map(process_one_scene_scannet, files, repeat(out_dir)))

    for fn in files:
        process_one_scene_scannet(fn, out_dir)


# Example of how to call the main function
if __name__ == "__main__":
    split = 'val'  # Choose between 'train' | 'val'
    out_dir = f'/data/disk1/data/scannet/scannet_3d_new'
    in_path = '/data/disk1/data/scannet/scans'  # Original ScanNet data directory
    prepare_and_process_scenes(split, out_dir, in_path)