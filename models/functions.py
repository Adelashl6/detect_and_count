from __future__ import division
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F


def vis_detections(im_path, class_det, w=None):
    im = cv2.imread(im_path)
    for det in class_det:
        bbox = det[:4]
        score = det[4]
        cv2.rectangle(im,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                      (127, 255, 0), 1)
        cv2.putText(im,
                    '{:.3f}'.format(score),
                    (int(bbox[0]), int(bbox[1] - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), thickness=1, lineType=8)

    if w is not None:
        cv2.imwrite(w, im)


def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt).clamp(min=0)  # [rows, 2]
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])

    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])
    union = area1 + area2 - overlap

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious


# def soft_argmax(maps, masks, temperature=None):
#     N, H, W = maps.size()
#     # Create coordinates grid
#     x_range = torch.arange(W, dtype=torch.float32, device=maps.device) + 0.5
#     y_range = torch.arange(H, dtype=torch.float32, device=maps.device) + 0.5
#     y_grid, x_grid = torch.meshgrid(y_range, x_range)
#     x_grid = x_grid.flatten()
#     y_grid = y_grid.flatten()
#
#     map_flats = maps.view(N, -1)
#     if temperature:
#         map_flats = F.softmax(map_flats * temperature, dim=-1)
#
#     mask_flats = masks.view(N, -1)
#     map_flats *= (mask_flats > 0).float()
#     map_flats /= torch.sum(map_flats, dim=-1, keepdim=True)
#
#     # Compute the expectation of the coordinates
#     expected_x = torch.sum(x_grid[None, :] * map_flats, -1)
#     expected_y = torch.sum(y_grid[None, :] * map_flats, -1)
#     return expected_x, expected_y


def get_targets(fboxes,
                hboxes,
                iboxes,
                featmap_size,
                cfg):
    assert fboxes.shape[0] == hboxes.shape[0] == iboxes.shape[0]
    batch_size = fboxes.shape[0]

    cls_list, reg_list, off_list, den_list = [], [], [], []
    region_mask_list, post_prob_list = [], []

    for idx in range(batch_size):
        fboxes_per_img = fboxes[idx]
        hboxes_per_img = hboxes[idx]
        iboxes_per_img = iboxes[idx]

        fboxes_per_img = fboxes_per_img[fboxes_per_img[:, -1] != -1]
        hboxes_per_img = hboxes_per_img[hboxes_per_img[:, -1] != -1]
        iboxes_per_img = iboxes_per_img[iboxes_per_img[:, -1] != -1]

        targets = get_target_single(
            fboxes_per_img, hboxes_per_img, iboxes_per_img, featmap_size, cfg)
        cls, reg, off, den, region_mask, post_prob = targets

        cls_list.append(cls)
        reg_list.append(reg)
        off_list.append(off)
        den_list.append(den)
        region_mask_list.append(region_mask)
        post_prob_list.append(post_prob)

    cls_targets = torch.stack(cls_list, dim=0)
    reg_targets = torch.stack(reg_list, dim=0)
    off_targets = torch.stack(off_list, dim=0)
    den_targets = torch.stack(den_list, dim=0)

    return (cls_targets, reg_targets, off_targets, den_targets,
            region_mask_list, post_prob_list)


def get_target_single(fboxes, hboxes, iboxes, feat_size, cfg):
    full_boxes = fboxes / cfg.down
    head_boxes = hboxes / cfg.down
    ignore_boxes = iboxes / cfg.down

    num_full = fboxes.shape[0]
    num_ignore = iboxes.shape[0]

    feat_h, feat_w = feat_size
    reg_map = fboxes.new_zeros((2, feat_h, feat_w))
    off_map = fboxes.new_zeros((2, feat_h, feat_w))
    mask = fboxes.new_zeros((feat_h, feat_w))
    region_mask = fboxes.new_zeros((num_full, feat_h, feat_w))

    x_range = torch.arange(feat_w, dtype=torch.float32, device=cfg.device) + 0.5
    y_range = torch.arange(feat_h, dtype=torch.float32, device=cfg.device) + 0.5
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    x_grid = x_grid.flatten().unsqueeze(0)
    y_grid = y_grid.flatten().unsqueeze(0)

    # full
    full_xs = 0.5 * full_boxes[:, 0] + 0.5 * full_boxes[:, 2]
    full_ys = 0.5 * full_boxes[:, 1] + 0.5 * full_boxes[:, 3]
    full_x_dis = -2 * torch.matmul(full_xs[:, None], x_grid) + \
                 full_xs[:, None] * full_xs[:, None] + x_grid * x_grid
    full_y_dis = -2 * torch.matmul(full_ys[:, None], y_grid) + \
                 full_ys[:, None] * full_ys[:, None] + y_grid * y_grid
    x_sigma = full_boxes[:, 2] - full_boxes[:, 0]
    y_sigma = full_boxes[:, 3] - full_boxes[:, 1]
    x_sigma = ((x_sigma[:, None] - 1) * 0.5 - 1) * 0.3 + 0.8
    y_sigma = ((y_sigma[:, None] - 1) * 0.5 - 1) * 0.3 + 0.8
    full_l2_dis = full_x_dis / (2.0 * x_sigma ** 2) + \
                  full_y_dis / (2.0 * y_sigma ** 2)
    cls_map = torch.exp(-full_l2_dis).view(-1, feat_h, feat_w)

    # head
    head_xs = 0.5 * head_boxes[:, 0] + 0.5 * head_boxes[:, 2]
    head_ys = 0.5 * head_boxes[:, 1] + 0.5 * head_boxes[:, 3]
    head_x_dis = -2 * torch.matmul(head_xs[:, None], x_grid) + \
                 head_xs[:, None] * head_xs[:, None] + x_grid * x_grid
    head_y_dis = -2 * torch.matmul(head_ys[:, None], y_grid) + \
                 head_ys[:, None] * head_ys[:, None] + y_grid * y_grid
    head_l2_dis = (head_x_dis + head_y_dis) / (2.0 * cfg.sigma ** 2)
    den_map = torch.exp(-head_l2_dis).view(-1, feat_h, feat_w)
    post_prob = F.softmax(-head_l2_dis, dim=0).view(-1, feat_h, feat_w)

    for ind in range(num_full):
        x1, y1, x2, y2, is_target = full_boxes[ind, :]
        argmax = torch.argmax(cls_map[ind])
        c_x = argmax % feat_w
        c_y = argmax / feat_w

        if is_target == 0:
            mask = mask.zero_()
            cls_map[ind] = cls_map[ind] * mask
            den_map[ind] = den_map[ind] * mask
            continue

        mask = mask.zero_()
        mask[y1.ceil().int():y2.int() + 1,
             x1.ceil().int():x2.int() + 1] = 1
        cls_map[ind] = cls_map[ind] * mask
        cls_map[ind, c_y, c_x] = 1
        reg_map[0,
                max(0, c_y - cfg.radius):c_y + cfg.radius + 1,
                max(0, c_x - cfg.radius):c_x + cfg.radius + 1] = torch.log(y2 - y1)
        reg_map[1,
        max(0, c_y - cfg.radius):c_y + cfg.radius + 1,
        max(0, c_x - cfg.radius):c_x + cfg.radius + 1] = torch.log(x2 - x1)
        off_map[0, c_y, c_x] = 0.5 * y1 + 0.5 * y2 - c_y - 0.5
        off_map[1, c_y, c_x] = 0.5 * x1 + 0.5 * x2 - c_x - 0.5
        region_mask[ind,
                    max(0, c_y - cfg.radius):c_y + cfg.radius + 1,
                    max(0, c_x - cfg.radius):c_x + cfg.radius + 1] = 0.1
        region_mask[ind, c_y, c_x] = 1

        mask = mask.zero_()
        mask[y1.ceil().int():(0.8 * y1 + 0.2 * y2).int() + 1,
             (0.75 * x1 + 0.25 * x2).ceil().int():(0.25 * x1 + 0.75 * x2).int() + 1] = 1
        post_prob[ind] = post_prob[ind] * mask

    if num_full > 0:
        cls_map, _ = cls_map.max(0, keepdim=True)
    else:
        cls_map = cls_map.sum(0, keepdim=True)
    den_map = den_map.sum(0, keepdim=True)

    for ind in range(num_ignore):
        x1, y1, x2, y2 = ignore_boxes[ind, :-1]
        ig_map = cls_map[0,
                         y1.ceil().int():y2.int() + 1,
                         x1.ceil().int():x2.int() + 1]
        ig_map[ig_map <= 0] = -1
    return cls_map, reg_map, off_map, den_map, region_mask, post_prob


def gaussian_blur(hm, kernel=3):    # TODO need to test kernel size
    border = (kernel - 1) // 2
    height = hm.shape[0]
    width = hm.shape[1]

    origin_max = np.max(hm)
    dr = np.zeros((height + 2 * border, width + 2 * border))
    dr[border: -border, border: -border] = hm.copy()
    dr = cv2.GaussianBlur(dr, (kernel, kernel), 0.5)
    hm= dr[border: -border, border: -border].copy()
    hm *= origin_max / np.max(hm)
    return hm


def get_results(cls, reg, offset, density, size,
                ratio=0.41, score=0.1, down=4, nms_thresh=0.5):
    cls = np.squeeze(cls)
    # cls = gaussian_blur(cls)
    reg = np.squeeze(np.exp(reg), axis=0)
    offset_y = offset[0, 0, :, :]
    offset_x = offset[0, 1, :, :]
    density = np.squeeze(density)

    y_c, x_c = np.where(cls > score)
    counts = density.sum()

    bboxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = reg[0, y_c[i], x_c[i]]
            w = ratio * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            x1 = np.clip(down * (x_c[i] + o_x + 0.5 - w / 2), 0, size[1])
            x2 = np.clip(down * (x_c[i] + o_x + 0.5 + w / 2), 0, size[1])
            y1 = np.clip(down * (y_c[i] + o_y + 0.5 - h / 2), 0, size[0])
            y2 = np.clip(down * (y_c[i] + o_y + 0.5 + h / 2), 0, size[0])
            hx1 = (0.75 * x1 + 0.25 * x2) / down
            hx2 = (0.25 * x1 + 0.75 * x2) / down
            hy1 = y1 / down
            hy2 = (0.8 * y1 + 0.2 * y2) / down
            s = cls[y_c[i], x_c[i]]
            # bboxs.append([x1, y1, x2, y2, s])
            bboxs.append([x1, y1, x2, y2, s, hx1, hy1, hx2, hy2])
        bboxs = np.asarray(bboxs, dtype=np.float32)
        # bboxs, _ = nms(bboxs, iou_thr=nms_thresh, device_id=0)
        # keep = cnms(bboxs, density)
        keep = nms(bboxs, density)
        bboxs = bboxs[keep, :5]

    return bboxs, counts


def get_head_count(bboxes, density):
    counts = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = int(math.ceil(x1))
        x2 = int(x2) + 1
        y1 = int(math.ceil(y1))
        y2 = int(y2) + 1
        counts.append(np.sum(density[y1:y2, x1:x2]))
    counts = np.array(counts)
    return counts


def cnms(dets, density, thr=0.5, cthr=0.2, alpha=0.85):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    hx1, hy1, hx2, hy2 = dets[:, 5], dets[:, 6], dets[:, 7], dets[:, 8]
    counts = get_head_count(dets[:, 5:], density)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        if counts[i] <= cthr:
            order = order[1:]
            continue
        keep.append(i)
        
        # full body box iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)

        # head body count
        xx1 = np.maximum(hx1[i], hx1[order[1:]])
        yy1 = np.maximum(hy1[i], hy1[order[1:]])
        xx2 = np.minimum(hx2[i], hx2[order[1:]])
        yy2 = np.minimum(hy2[i], hy2[order[1:]])

        inter = get_head_count(np.stack((xx1, yy1, xx2, yy2), axis=-1), density)
        unions = counts[i] + counts[order[1:]] - inter

        # remove duplicate box
        iou_thr = max(thr, alpha * counts[i] / (counts[i] + 1))
        # iou_thr = thr
        count_thr = 2 * counts[i]

        cond1 = ious <= iou_thr
        cond2 = unions >= count_thr
        inds = np.where(cond1 | cond2)[0]
        order = order[inds + 1]

    return keep


def nms(dets, density, thr=0.5, cthr=0.2, alpha=0.85):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    hx1, hy1, hx2, hy2 = dets[:, 5], dets[:, 6], dets[:, 7], dets[:, 8]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # full body box iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)

        cond1 = ious <= thr
        inds = np.where(cond1)[0]
        order = order[inds + 1]

    return keep