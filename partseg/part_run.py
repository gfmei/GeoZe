import json
import os
import sys
import time

import torch
import random
import warnings
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]      # repo root, then partseg/ -- runs from anywhere
from shapenet import ShapeNetPart
from libs.lib_o3d import batch_geo_feature
from partmodel import post_search
from partmodel.post_search import search_prompt, search_vweight
from rendering.prejection import RealisticProjection
from partseg.partclip import clip
try:
    from libs.lib_vis import get_colored_image_pca_sep
except Exception:
    get_colored_image_pca_sep = None

warnings.filterwarnings("ignore")

PC_NUM = 2048

TRANS = -1.5

params = {'vit_b16': {'maxpoolz': 5, 'maxpoolxy': 11, 'maxpoolpadz': 2, 'maxpoolpadxy': 5,
                      'convz': 5, 'convxy': 5, 'convsigmaxy': 1, 'convsigmaz': 2, 'convpadz': 2, 'convpadxy': 2,
                      'imgbias': 0., 'depth_bias': 0.3, 'obj_ratio': 0.7, 'bg_clr': 0.0,
                      'resolution': 224, 'depth': 112}}
net = 'vit_b16'

cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
          'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
          'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}


class Extractor(torch.nn.Module):
    def __init__(self, model):
        super(Extractor, self).__init__()

        self.model = model
        self.pc_views = RealisticProjection(params[net])
        self.get_img = self.pc_views.get_img
        self.params_dict = params[net]

    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc)

        _, x = self.model.encode_image(img)
        x = x / x.norm(dim=-1, keepdim=True)
        B, L, C = x.shape
        if is_save:
            feats = torch.nn.functional.interpolate(x.reshape(B, 14, 14, C).permute(0, 3, 1, 2), size=(224, 224),
                                                    mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            for i in range(len(img)):
                # Normalize the depth values to the desired range (0 to 255 in this example)
                normalized_depth = cv2.normalize(img[i][0].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)

                # Convert the depth matrix to an 8-bit unsigned integer (uint8) image
                depth_image = np.uint8(normalized_depth)
                feat = feats[i]
                get_colored_image_pca_sep(feat.cpu().numpy(), i)
                cv2.imwrite(f'saved_depth_{i}.png', depth_image)

        x = x.reshape(B, 14, 14, C).permute(0, 3, 1, 2)
        # print(B, L, C, x.shape, is_seen.shape, point_loc_in_img.shape)
        return is_seen, point_loc_in_img, x


def extract_feature_maps(model_name, data_path, class_choice, device):
    output_path = 'output/{}/'.format(model_name.replace('/', '_'))
    mode = 'test'

    save_path = os.path.join(output_path, class_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, "{}_features.pt".format(mode))):
        return

    model, _ = clip.load(model_name, device=device)
    model.to(device)

    segmentor = Extractor(model)
    segmentor = segmentor.to(device)
    segmentor.eval()

    print('\nStart to extract and save feature maps of class {}...'.format(class_choice))
    test_loader = DataLoader(ShapeNetPart(data_path, partition=mode, num_points=PC_NUM, class_choice=class_choice),
                             batch_size=1, shuffle=False, drop_last=False)
    feat_store, label_store, pc_store = [], [], []
    normal_store, fpfh_store = [], []
    ifseen_store, pointloc_store = [], []
    for data in tqdm(test_loader):
        pc, cat, label = data
        pc, label = pc.cuda(), label.cuda()
        with torch.no_grad():
            is_seen, point_loc_in_img, feat = segmentor(pc)
            normal, fpfh = batch_geo_feature(pc, voxel_size=0.05)
            pc_store.append(pc)
            normal_store.append(normal)
            fpfh_store.append(fpfh)
            feat_store.append(feat[None, :, :, :])
            label_store.append(label.squeeze()[None, :])
            ifseen_store.append(is_seen[None, :, :])
            pointloc_store.append(point_loc_in_img[None, :, :, :])

    pc_store = torch.cat(pc_store, dim=0)
    normal_store = torch.cat(normal_store, dim=0)
    fpfh_store = torch.cat(fpfh_store, dim=0)
    feat_store = torch.cat(feat_store, dim=0)
    label_store = torch.cat(label_store, dim=0)
    ifseen_store = torch.cat(ifseen_store, dim=0)
    pointloc_store = torch.cat(pointloc_store, dim=0)

    # save features for post-search
    print('Save feature and labels: ============================')
    torch.save(pc_store, os.path.join(save_path, "{}_pc.pt".format(mode)))
    torch.save(normal_store, os.path.join(save_path, "{}_normal.pt".format(mode)))
    torch.save(fpfh_store, os.path.join(save_path, "{}_fpfh.pt".format(mode)))
    torch.save(feat_store, os.path.join(save_path, "{}_features.pt".format(mode)))
    torch.save(label_store, os.path.join(save_path, "{}_labels.pt".format(mode)))
    torch.save(ifseen_store, os.path.join(save_path, "{}_ifseen.pt".format(mode)))
    torch.save(pointloc_store, os.path.join(save_path, "{}_pointloc.pt".format(mode)))


def main(args):
    """Cache the multi-view CLIP features, then evaluate the ORIGINAL GeoZe aggregation.

    The features are the expensive part and are written once per category under `output/`;
    everything downstream reads them back.  For SimpleGeoZe -- the method this repo ships, and
    the faster and more accurate of the two -- use `simple_geoze.py`.
    """
    random.seed(0)
    torch.manual_seed(0)
    post_search.TOKEN_LAYOUT = args.layout
    classes = list(cat2id) if args.classchoice == 'all' else args.classchoice.split(',')
    img_size = (params[net]['resolution'], params[net]['resolution'])

    results, t0 = {}, time.time()
    for class_choice in classes:
        extract_feature_maps(args.modelname, args.datasetpath, class_choice, args.device)
        if args.extract_only:
            continue
        r = search_prompt(class_choice, args.modelname, only_evaluate=not args.search,
                          img_size=img_size)
        if args.search:                              # prompt search, then view-weight search
            search_vweight(class_choice, args.modelname, r)
            continue
        results[class_choice] = r
        print(f"  {class_choice:12s} n={r['n']:4d}  Acc {r['acc']:6.2f}  IoU {r['iou']:6.2f}  "
              f"{r['ms']:7.2f} ms/shape", flush=True)

    if not results:
        return
    cls = float(np.mean([r['iou'] for r in results.values()]))
    inst = float(np.concatenate([r['shape_ious'] for r in results.values()]).mean() * 100)
    print(f'\nRESULT geoze  class-mIoU={cls:.2f}  instance-mIoU={inst:.2f}  '
          f"Acc={np.mean([r['acc'] for r in results.values()]):.2f}  "
          f"{np.mean([r['ms'] for r in results.values()]):.2f} ms/shape  "
          f'({len(results)} classes, {time.time() - t0:.0f}s)', flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'class_miou': cls, 'instance_miou': inst, 'args': vars(args),
                   'by_class': {c: r['iou'] for c, r in results.items()}},
                  open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='ViT-B/16')
    parser.add_argument('--classchoice', default='all', help='a category, a comma list, or all')
    parser.add_argument('--datasetpath', default='/data/disk1/data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--extract_only', action='store_true', help='cache features and stop')
    parser.add_argument('--search', action='store_true', help='prompt + view-weight search')
    parser.add_argument('--layout', default='repo', choices=['repo', 'tokens'],
                        help="cached feature-map layout; see post_search.view_tokens")
    parser.add_argument('--out', default='')
    main(parser.parse_args())
