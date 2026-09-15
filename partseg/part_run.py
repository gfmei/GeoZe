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

sys.path.append(os.path.abspath('../'))
from shapenet import ShapeNetPart
from libs.lib_o3d import batch_geo_feature
from partmodel.best_param import best_param_v2
from partmodel.eval_v2 import evaluate, summarize
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


V2_KEYS = ('part', 'n_sp', 'knn', 'w_x', 'w_n', 'w_g', 'w_v', 'min_size', 'th_f', 'th_n',
           'rounds', 'alpha', 'gamma0', 'n_ev', 'embed', 'n_land', 'refine', 'land', 'seed',
           'sparse_iters', 'vccs_voxel', 'vccs_w_s', 'vccs_w_f', 'vccs_seed',
           'vccs_boundary')


def main(args):
    """Evaluate one category, a comma-separated list, or `all` (16 categories, PointCLIP V2 protocol).

    --model geoze              the original GeoZe aggregation (partmodel/partgeoze.py)
    --model v2                 PartGeoZe v2   (partmodel/partgeozev2.py)
    --baseline point|meanpool  no aggregation / mean pooling over the v2 superpoints
    --baseline oracle          majority ground-truth part per superpoint: the partition's ceiling
    """
    random.seed(0)
    torch.manual_seed(0)
    post_search.TOKEN_LAYOUT = args.layout      # the GeoZe path reads the module-level setting
    classes = list(cat2id) if args.classchoice == 'all' else args.classchoice.split(',')
    img_size = (params[net]['resolution'], params[net]['resolution'])

    p = dict(best_param_v2)
    for k in V2_KEYS:
        if getattr(args, k, None) is not None:
            p[k] = getattr(args, k)
    p['center'] = not args.no_center
    p['split'] = not args.no_split
    p['self_tune'] = not args.no_self_tune
    p['ortho'] = args.ortho
    mode = args.baseline or args.model
    tag = args.tag or (mode if mode != 'v2' else 'v2' + ('/' + p['part'] if p['part'] != 'spectral' else ''))
    tag += ('' if args.layout == 'repo' else '/tokens') + ('' if args.prompt == 'repo' else '/simple')

    results, t0 = {}, time.time()
    for class_choice in classes:
        # extract and save feature maps, labels, point locations (no-op when cached)
        extract_feature_maps(args.modelname, args.datasetpath, class_choice, args.device)
        if mode == 'geoze':
            r = search_prompt(class_choice, args.modelname, only_evaluate=not args.search, img_size=img_size)
            if args.search:                          # prompt search, then view-weight search
                search_vweight(class_choice, args.modelname, r)
                continue
        else:
            r = evaluate(class_choice, args.modelname, mode, p, device=args.device, img_size=img_size,
                         limit=args.limit, layout=args.layout, prompt=args.prompt)
        results[class_choice] = r
        print(f"  {class_choice:12s} n={r['n']:4d}  Acc {r['acc']:6.2f}  IoU {r['iou']:6.2f}  "
              f"{r['ms']:7.2f} ms/shape  {r['regions']:5.1f} regions", flush=True)

    if not results:
        return
    sm = summarize(results)
    print(f"\nRESULT shapenetpart/{tag}  class-mIoU={sm['class_miou']:.2f}  "
          f"instance-mIoU={sm['instance_miou']:.2f}  Acc={sm['acc']:.2f}  "
          f"{sm['ms']:.2f} ms/shape  ({len(results)} classes, {time.time() - t0:.0f}s)", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump({'tag': tag, 'summary': sm, 'param': p if mode != 'geoze' else None, 'args': vars(args),
                   'classes': {c: {k: v for k, v in r.items() if k != 'shape_ious'} for c, r in results.items()},
                   'shape_ious': {c: r['shape_ious'] for c, r in results.items()}},
                  open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='ViT-B/16')
    parser.add_argument('--classchoice', default='table', help='a category, a comma list, or all')
    parser.add_argument('--datasetpath', default='/data/disk1/data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', default='v2', choices=['geoze', 'v2'])
    parser.add_argument('--baseline', default='', choices=['', 'point', 'meanpool', 'oracle'])
    parser.add_argument('--search', action='store_true', help='GeoZe only: prompt + view-weight search')
    parser.add_argument('--no_center', action='store_true', help='v2: merge on raw instead of centred cosines')
    parser.add_argument('--no_split', action='store_true', help='v2: keep spectral clusters as they are')
    parser.add_argument('--limit', type=int, default=0, help='shapes per class (0 = all)')
    parser.add_argument('--layout', default='repo', choices=['repo', 'tokens'],
                        help="cached feature-map layout: the shipped reshape, or true CLIP patch tokens")
    parser.add_argument('--prompt', default='repo', choices=['repo', 'simple'],
                        help='the searched per-part sentences, or `a {part} of a {category}`')
    parser.add_argument('--tag', default='')
    parser.add_argument('--out', default='', help='json with per-class and per-shape results')
    parser.add_argument('--part', default=None,
                        choices=[None, 'kmeans', 'spectral', 'vccs', 'vccs_gpu', 'fps'])
    parser.add_argument('--vccs_seed', default=None, choices=[None, 'fps', 'grid', 'zcurve'])
    parser.add_argument('--embed', default=None,
                        choices=[None, 'sparse', 'dense', 'nystrom', 'lobpcg'])
    parser.add_argument('--no_self_tune', action='store_true', help='drop local scaling of the position cue')
    parser.add_argument('--ortho', action='store_true', help='Nystrom: orthogonalised extension')
    parser.add_argument('--land', default=None, choices=[None, 'curve', 'fps'])
    parser.add_argument('--seed', default=None, choices=[None, 'curve', 'fps'])
    for k in ('th_f', 'th_n', 'alpha', 'gamma0', 'w_x', 'w_n', 'w_g', 'w_v',
              'vccs_voxel', 'vccs_w_s', 'vccs_w_f', 'vccs_boundary'):
        parser.add_argument(f'--{k}', type=float, default=None)
    for k in ('n_sp', 'knn', 'rounds', 'min_size', 'n_ev', 'n_land', 'refine',
              'sparse_iters'):
        parser.add_argument(f'--{k}', type=int, default=None)
    main(parser.parse_args())
