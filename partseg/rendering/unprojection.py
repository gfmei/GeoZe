import torch


def vanilla_upprojection(img_feats, is_seen, point_loc, img_size=(64, 64), n_points=2048, vweights=None):
    device = img_feats.device
    b, nv, hw, c = img_feats.size(0), img_feats.size(1), img_feats.size(2), img_feats.size(3)
    img_feats = img_feats.reshape(b * nv, hw, c)
    point_loc = point_loc.reshape(b * nv, -1, 2)  # (b * nv, hw, 2)
    is_seen = is_seen.reshape(b * nv, -1, 1)  # (b * nv, hw, 1)

    # upsample to the original image size
    upsample = torch.nn.Upsample(size=img_size, mode='bilinear')  # nearest, bilinear
    avgpool = torch.nn.AvgPool2d(6, 1, 0)
    padding = torch.nn.ReplicationPad2d((2, 3, 2, 3))

    img_feats = img_feats.half().permute(0, 2, 1).reshape(-1, c, int(hw**0.5), int(hw**0.5))
    img_feats = avgpool(padding(img_feats))
    output = upsample(img_feats)

    # back-projecting to each points
    nbatch = torch.repeat_interleave(torch.arange(0, nv * b)[:, None], n_points).view(-1, ).to(device).long()
    yy = point_loc[:, :, 0].view(-1).long()
    xx = point_loc[:, :, 1].view(-1).long()

    point_feats = output[nbatch, :, yy, xx]
    point_feats = point_feats.view(b, nv, n_points, -1)
    is_seen = is_seen.reshape(b, nv, n_points, 1)

    # points features is the weighted mean of pixel features
    if vweights is None:
        point_feats = torch.mean(point_feats * is_seen, dim=1)
    else:
        vweights = vweights.view(1, -1, 1, 1)
        point_feats = torch.mean(point_feats * vweights * is_seen, dim=1)

    return point_feats, is_seen, point_loc
