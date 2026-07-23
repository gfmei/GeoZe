"""Semantic-segmentation metrics shared by the scene datasets."""
import numpy as np


class ConfusionMatrix:
    """mIoU / mAcc / OA over `n` classes.

    Classes with NO ground-truth point are excluded from the means.  This matters on scene
    subsets: counting an absent class as IoU 0 drags mIoU down by (#absent / n) and makes
    subset numbers incomparable with full-split ones.
    """

    def __init__(self, n=20):
        self.n = n
        self.m = np.zeros((n, n), np.int64)

    def add(self, gt, pred):
        v = (gt >= 0) & (gt < self.n)
        self.m += np.bincount(gt[v] * self.n + pred[v], minlength=self.n ** 2).reshape(self.n, self.n)

    def scores(self):
        tp = np.diag(self.m).astype(np.float64)
        present = self.m.sum(1) > 0
        iou = tp / np.maximum(self.m.sum(1) + self.m.sum(0) - tp, 1)
        acc = tp / np.maximum(self.m.sum(1), 1)
        return {'miou': float(iou[present].mean()) if present.any() else 0.0,
                'macc': float(acc[present].mean()) if present.any() else 0.0,
                'oa': float(tp.sum() / max(self.m.sum(), 1)),
                'iou': iou, 'present': present}
