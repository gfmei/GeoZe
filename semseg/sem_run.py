import argparse
import logging
import os
import random
import sys
import urllib

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm


sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))
from semseg.utils.label_constants import SCANNET_LABELS_20, MATTERPORT_LABELS_21, MATTERPORT_LABELS_40, \
    MATTERPORT_LABELS_80, MATTERPORT_LABELS_160, NUSCENES_LABELS_16, NUSCENES_LABELS_DETAILS, MAPPING_NUSCENES_DETAILS
from semmodel.semgeoze import SemGeoze
from semseg.utils.util import get_palette, extract_text_feature
from semseg.utils.data_utils import collation_fn_eval_all
from semseg.semdata.scannet import ScanNet
from semseg.utils import config, metric


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                        default='config/scannet/ratchet_openseg.yaml',
                        help='config file')
    parser.add_argument('--save_folder', type=str, default='./results', help='config file')
    # parser.add_argument('--last_name', type=str, default='scannet_3d', help='data set name')
    parser.add_argument('--vis_pred', type=bool,
                        default=True,
                        help='config file')
    parser.add_argument('--vis_gt', type=bool,
                        default=True,
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/test_ours_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'  # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
    else:  # an arbitrary semdata, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    if hasattr(args, 'map_nuscenes_details'):
        labelset = list(NUSCENES_LABELS_DETAILS)
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette


def main():
    '''Main function.'''

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s'
        % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)

    # ####################### Data Loader ####################### #
    val_data = ScanNet(root=args.data_root, prefix_feat=args.fused_feature_type, voxel_size=args.voxel_size,
                       split=args.split, seg_min_verts=20, k_neighbors=50, thres=0.01)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                             drop_last=False, collate_fn=collation_fn_eval_all,
                                             sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.last_name
    # if hasattr(args, 'labelset'):
    #     # if the labelset is specified
    #     labelset_name = args.labelset
    transformer = SemGeoze(dim=768, sigma_d=0.01, sigma_a=0.001, sigma_e=0.001, angle_k=10, voxel_size=args.voxel_size,
                           n_pts=2000).cuda()
    transformer.eval()

    evaluate(transformer, val_loader, labelset_name)


def evaluate(transformer, val_data_loader, labelset_name='scannet_3d'):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy:  # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    # save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True

    text_features, labelset, mapper, palette = precompute_text_related_properties(labelset_name)

    with torch.no_grad():
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                logger.info(
                    "\nEvaluation {} out of {} runs...\n".format(rep_i + 1, args.test_repeats))

            # repeat the evaluation process
            if rep_i > 0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []

            for i, (coords, xyz, feat, label, feat_3d, spts_idx, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                if torch.all(coords[:, 0] != 0):
                    class UnsupportedBatchSizeError(Exception):
                        def __init__(self, message="Batches with more than one point cloud are not supported yet"):
                            self.message = message
                            super().__init__(self.message)

                    raise UnsupportedBatchSizeError()

                org_feats = feat_3d.cuda(non_blocking=True)
                text_features = F.normalize(text_features, dim=-1)

                if feature_type == 'fusionssss':
                    # NPAttn part
                    org_feats = org_feats.squeeze_().half()
                    predictions, protos, fpfhs = transformer(coords.unsqueeze(0).to(torch.float32),
                                                             org_feats.unsqueeze(0).to(torch.float32),
                                                             text_features.to(torch.float32))
                    # predictions = predictions.squeeze()[inds_reverse, :]
                    predictions = F.normalize(predictions, dim=-1)
                    pred = (predictions.half() @ text_features.half().t())  # * (fpfhs.half() @ protos.half().t())
                    del org_feats, predictions, protos, fpfhs

                else:
                    org_feats = F.normalize(org_feats[inds_reverse, :], dim=-1)
                    pred = org_feats.half() @ text_features.half().t()
                    del org_feats
                # pred = predictions.half() @ protos.half().t()
                # pred = sinkhorn(1 - pred.unsqueeze(0), p=None, q=probs)[0].squeeze()
                logits_pred = torch.max(pred, 1)[1].detach_().cpu()
                if mark_no_feature_to_unknown:
                    # some points do not have 2D features from 2D feature fusion.
                    # Directly assign 'unknown' label to those points during inference.
                    logits_pred[~mask[inds_reverse]] = len(labelset) - 1
                torch.cuda.empty_cache()

                # special case for nuScenes, evaluation points are only a subset of input
                if 'nuscenes' in labelset_name:
                    label_mask = (label != 255)
                    label = label[label_mask]
                    logits_pred = logits_pred[label_mask]
                    pred = pred[label_mask]
                if eval_iou:
                    if mark_no_feature_to_unknown:
                        if "nuscenes" in labelset_name:  # special case
                            masks.append(mask[inds_reverse][label_mask])
                        else:
                            masks.append(mask[inds_reverse])

                    if args.test_repeats == 1:
                        # save directly the logits
                        preds.append(logits_pred)
                    else:
                        # only save the dot-product results, for repeat prediction
                        preds.append(pred.cpu())
                    gts.append(label.cpu())
            accumu_iou = 0.0
            if eval_iou:
                gt = torch.cat(gts)
                pred = torch.cat(preds)

                pred_logit = pred
                if args.test_repeats > 1:
                    pred_logit = pred.float().max(1)[1]

                if mapper is not None:
                    pred_logit = mapper[pred_logit]

                if mark_no_feature_to_unknown:
                    mask = torch.cat(masks)
                    pred_logit[~mask] = 256

                if args.test_repeats == 1:
                    accumu_iou = metric.evaluate(pred_logit.numpy(), gt.numpy(), dataset=labelset_name, stdout=True)
                elif args.test_repeats > 1:
                    store = pred + store
                    store_logit = store.float().max(1)[1]
                    if mapper is not None:
                        store_logit = mapper[store_logit]

                    if mark_no_feature_to_unknown:
                        store_logit[~mask] = 256
                    accumu_iou = metric.evaluate(store_logit.numpy(), gt.numpy(), stdout=True, dataset=labelset_name)
        return accumu_iou


if __name__ == '__main__':
    main()
    # sh run/eval_ATTN.sh exp/scannet_fused_fts_ATTN config/scannet/ours_openseg_pretrained.yaml fusion
