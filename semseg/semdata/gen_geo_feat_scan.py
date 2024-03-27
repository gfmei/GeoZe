import os
from glob import glob
from os.path import join

import torch

from libs.lib_o3d import estimate_geo_feature


def gen_geo_feat_scan(datapath_prefix, split, voxel_size=0.05, out_root=None):
    data_paths = sorted(glob(join(datapath_prefix, split, '*_vh_clean_2.pth')))
    if len(data_paths) == 0:
        raise Exception('0 file is loaded in the point loader.')
    for data_path in data_paths:
        data = torch.load(data_path)
        # print(data)
        locs_in, _, labels_in = data
        normals, fpfhs = estimate_geo_feature(locs_in, None, voxel_size=voxel_size)
        print('estimate the fpfh features end')
        save_dict = {'normal': normals, 'fpfh': fpfhs}
        output_file_name = os.path.basename(data_path)[:-4] + '.pth'
        output_file_path = os.path.join(out_root, output_file_name)
        torch.save(save_dict, output_file_path)
        print(output_file_path)
        print('finished one scene')


if __name__ == '__main__':
    data_root = '/data/disk1/data/scannet/scannet_3d'
    split = 'val'
    out_dir = '/data/disk1/data/scannet/scannet_3d/scannet_3d_geo'
    out_dir = os.path.join(out_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    gen_geo_feat_scan(data_root, split, out_root=out_dir)
