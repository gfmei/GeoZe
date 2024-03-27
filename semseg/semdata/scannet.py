import copy
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from semseg.utils.data_utils import Voxelizer, collation_fn_eval_all, num_to_natural


class ScanNet(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, root, split, voxel_size=0.05, prefix_feat='scannet_multiview_openseg', seg_min_verts=20,
                 k_neighbors=12, thres=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.split = split
        if split is None:
            split = ''
        self.data_dir = root
        self.prefix_feat = prefix_feat
        self.seg_min_verts = seg_min_verts
        self.k_neighbors = k_neighbors
        self.thres = thres

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if split == 'train':  # for training set, export a chunk of point cloud
            self.n_split_points = 20000
            self.num_rand_file_per_scene = 5
            self.scannet_file_list = self.read_files('../semseg/meta/scannet/scannetv2_train.txt')
        else:  # for the validation set, export the entire point cloud instead of chunks
            self.n_split_points = 2000000
            self.num_rand_file_per_scene = 1
            self.scannet_file_list = self.read_files('../semseg/meta/scannet/scannetv2_val.txt')

        self.voxel_size = voxel_size

    def __getitem__(self, index_long):
        index = index_long % len(self.scannet_file_list)
        scene_id = self.scannet_file_list[index]
        scene_name = scene_id + '_vh_clean_2.pth'
        # data_root_3d = join(self.data_dir, 'scannet_3d', self.split, scene_name)
        # load 3D data (point cloud)
        # raw_points, raw_colors, raw_labels = torch.load(data_root_3d)
        data_root_geo = join(self.data_dir, 'scannet_3d_new', self.split, scene_name)
        load_pcd = torch.load(data_root_geo)
        raw_points = load_pcd['coord']
        raw_colors = load_pcd['color']
        raw_labels = load_pcd['label']
        raw_sptids = load_pcd['spt_ids']
        raw_normals = load_pcd['normal']
        raw_fpfhs = load_pcd['fpfh']
        raw_labels[raw_labels == -100] = 255
        cat_raw_labels = np.concatenate((raw_labels.reshape(-1, 1), raw_sptids.reshape(-1, 1)), axis=1)
        # load 3D features
        nn_occur = np.random.randint(5)
        try:
            feature_root = join(self.data_dir, self.prefix_feat, scene_id + '_%d.pt' % nn_occur)
            processed_data = torch.load(feature_root)
        except Exception as e:
            feature_root = join(self.data_dir, self.prefix_feat, scene_id + '_0.pt')
            processed_data = torch.load(feature_root)
        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            if isinstance(mask_chunk, np.ndarray):  # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != 'train':  # val or test set
                feat_3d_new = torch.zeros((raw_points.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(mask_chunk)  # every point needs to be evaluted
        elif len(processed_data.keys()) > 2:  # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data[
                'mask_full']
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True  # mask out points without feature assigned
        else:
            raise NotImplementedError

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]
        feats_in = np.concatenate((raw_colors, raw_normals, raw_fpfhs), axis=1)
        # calculate the corresponding point features after voxelization
        if self.split == 'train' and flag_mask_merge:
            locs, feat_lo, cat_labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                raw_points, feats_in, cat_raw_labels, return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        elif self.split == 'train' and not flag_mask_merge:  # legacy, for old processed features
            feat_3d = feat_3d[mask]  # get features for visible points
            locs, feat_lo, cat_labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                raw_points, feats_in, cat_raw_labels, return_ind=True)
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point clouds
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1
            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        else:
            # original_locs = raw_points.copy()
            locs, feat_lo, cat_labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                raw_points[mask_chunk], feats_in[mask_chunk], cat_raw_labels[mask_chunk], return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]
            # original_locs = original_locs[vox_ind]
        feat_lo = torch.from_numpy(feat_lo).float()
        locs = torch.from_numpy(locs).float()
        cat_labels = cat_labels
        spts_idx = torch.from_numpy(num_to_natural(cat_labels[:, 1])).long()
        coords = copy.deepcopy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        labels = torch.from_numpy(cat_raw_labels)
        # labels = cat_labels[:, 0]

        return coords, locs, feat_lo, labels, feat_3d, spts_idx, mask, torch.from_numpy(inds_reconstruct).long()

    def __len__(self):
        return len(self.scannet_file_list)

    @staticmethod
    def read_files(file):
        f = open(file)
        lines = f.readlines()
        name_list = [line.split('.')[0].strip() for line in lines]
        f.close()
        return name_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader, Dataset
    from libs.lib_vis import creat_labeled_point_cloud, get_colored_point_cloud_from_soft_labels

    data_root = '/data/disk1/data/scannet'
    # data_root = '/storage2/TEV/datasets/ScanNet'
    test_data = ScanNet(data_root, split='val')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collation_fn_eval_all)

    for i, data in enumerate(test_dataloader):
        print([x.shape for x in data])
        xyz = data[1]
        points = np.asarray(xyz)
        spts_ids = np.asarray(data[5])
        gt_ids = np.asarray(data[3])
        # mu_xyz, mu_feats, idx = down_sample(xyz.unsqueeze(0), data[2].unsqueeze(0), int(1.5*len(np.unique(spts_ids))))
        # soft_labels = connect_clustering(data[0], mu_xyz, [data[1][:, :, :3], data[1][:, :, 3:6]],
        #                            [mu_feats[:, :, :3], mu_feats[:, :, 3:6]],
        #                            [1.0, 1.0], ['cos', 'eu'], iters=30, idx=None)[0]
        normals = F.normalize(data[2][:, :3], dim=-1)
        colors = data[2][:, 3:6]
        # soft_labels = connect_clustering(xyz.unsqueeze(0), mu_xyz, [
        #     normals.unsqueeze(0), colors.unsqueeze(0)],
        #                                  [mu_feats[:, :, :3], mu_feats[:, :, 3:6]],
        #                                  [1.0, 1.0], ['cos', 'eu'], iters=30, idx=None)[0]
        creat_labeled_point_cloud(points, spts_ids, 'pred')
        creat_labeled_point_cloud(points, gt_ids, 'gt')
        # get_colored_point_cloud_from_soft_labels(points, soft_labels[0].cpu().numpy(), 'clus')

        break
