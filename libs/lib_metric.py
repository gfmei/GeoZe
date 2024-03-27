import numpy as np

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, eva=False):
    label = label.squeeze()
    shape_ious = []
    category = {}
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[int(label[shape_idx])]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[int(label[0])])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        if label[shape_idx] not in category:
            category[int(label[shape_idx])] = [shape_ious[-1]]
        else:
            category[label[shape_idx]].append(shape_ious[-1])
    if eva:
        return shape_ious, category
    else:
        return shape_ious
