"""Class names, prompts and searched hyper-parameters for zero-shot scene segmentation.

Mirrors partseg/partmodel/best_param.py: everything tuned once and then frozen at evaluation
time lives here, so sem_run.py stays free of magic numbers.

Two datasets, and they need DIFFERENT text towers.  A VLM's fused 3D features live in whatever
text space its 2D backbone was trained against, so the classifier table has to come from that
same tower or the cosines are meaningless:

    ScanNet   LSeg    512-d  ->  CLIP ViT-B/32
    nuScenes  OpenSeg 768-d  ->  CLIP ViT-L/14
"""

# --------------------------------------------------------------------------- #
#  ScanNet v2, 20 classes                                                      #
# --------------------------------------------------------------------------- #
SCANNET20 = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture',
]

# --------------------------------------------------------------------------- #
#  nuScenes lidarseg, 16 classes                                               #
# --------------------------------------------------------------------------- #
NUSCENES16 = [
    'barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person',
    'traffic cone', 'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation',
]

# OpenScene does NOT classify nuScenes against the 16 eval names: it encodes a 43-entry
# DETAILED vocabulary and maps the argmax back.  'manmade' and 'other flat' are not things a
# VLM has ever seen captioned, whereas 'building' / 'pole' / 'curb' are.  Measured on 31 val
# scans, region mean pooling: 16 names -> 0.1843 mIoU, 43 details -> 0.2691 (+8.5), OA
# 0.354 -> 0.609.  Do not "simplify" this back to the 16 names.
NUSCENES_DETAILS = [
    'barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer',
    'crane', 'dump truck', 'motorcycle', 'person', 'pedestrian', 'traffic cone', 'trailer',
    'semi trailer', 'cargo container', 'shipping container', 'freight container', 'truck',
    'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn',
    'meadow', 'turf', 'sod', 'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk',
    'bush', 'shrub', 'plant', 'flower', 'woods',
]
NUSCENES_DETAILS_MAP = [
    0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9,
    10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
]

CLASS_NAMES = {'scannet20': SCANNET20, 'nuscenes16': NUSCENES16}

# OpenScene renames ScanNet's last class before encoding (run/evaluate.py); nuScenes is used
# verbatim.
LABELSET = {
    'scannet20': SCANNET20[:-1] + ['other'],
    'nuscenes16': list(NUSCENES_DETAILS),
}

# argmax index -> eval class id.  None means the table is already in eval-class order.
LABEL_MAP = {'scannet20': None, 'nuscenes16': list(NUSCENES_DETAILS_MAP)}

best_prompt = {
    'scannet20': ['a {} in a scene'],
    'nuscenes16': ['a {} in a scene'],
}

text_encoder = {
    'scannet20': 'openai/clip-vit-base-patch32',     # LSeg  -> 512-d
    'nuscenes16': 'openai/clip-vit-large-patch14-336',  # OpenSeg -> 768-d (the @336 tower)
}

# `th_f` is the semantic admissibility of a merge, `th_n` the geometric one; `rounds` is the
# depth of the mutual-best-match hierarchy.  See semgeozev2.HierMerge.
#
# The nuScenes distances are an order of magnitude larger than ScanNet's because the scene is a
# ~200 m LiDAR scan rather than a 5 m room, and `w_c=0` because there is no colour.
best_param = {
    'scannet20': dict(
        n_anchor=128,          # landmark queries per region for the consensus estimator
        q_gamma=1.0,           # sharpness of the structural confidence q_i
        th_f=0.85, th_n=0.30,  # merge admissibility
        rounds=10,             # hierarchy depth
        knn=8,                 # point neighbours used to build the region graph
        min_votes=4,           # multi-curve agreement needed to accept a neighbour pair
        curve_voxel=0.02, curve_max_dist=0.3,
        gamma0=0.0,            # inter-region residual strength (0 = off; see NOTE below)
        w_c=1.0,               # colour cue is available
    ),
    'nuscenes16': dict(
        n_anchor=128,
        q_gamma=1.0,
        th_f=0.85, th_n=0.30,
        rounds=10,
        knn=8,
        min_votes=4,
        curve_voxel=0.20, curve_max_dist=3.0,   # outdoor: ~0.5 m point spacing at range
        gamma0=0.0,
        w_c=0.0,               # no colour on LiDAR -> drop the cue instead of feeding zeros
    ),
}

# NOTE  gamma0 defaults to 0.  Inter-region attention measured +0.0015 mIoU on ScanNet full val,
# inside the noise, because its normal-based boundary gate sits at 0.908 (wide open) — adjacent
# segments usually DO have agreeing normals whether or not they are the same object.  It is kept
# implemented and switchable (--gamma0) rather than deleted, since it is part of the method and
# may matter on partitions with sharper boundaries.
