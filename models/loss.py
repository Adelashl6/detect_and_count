import torch
import torch.nn as nn

# from opr import PrRoIPool


class cls_loss(nn.Module):
    def __init__(self):
        super(cls_loss, self).__init__()
        self.gamma = 2.0
        self.alpha = 4.0

    def forward(self, pred, target, factor=1.0):
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        ign_inds = target.lt(0).float()

        pos_weights = pos_inds * torch.pow(1 - pred, self.gamma)
        neg_weights = (neg_inds - ign_inds) * \
                      torch.pow(1 - target, self.alpha) * \
                      torch.pow(pred, self.gamma)
        pos_loss = -torch.log(pred) * pos_weights
        neg_loss = -torch.log(1 - pred) * neg_weights

        loss = pos_loss + neg_loss
        loss = loss.sum() / max(1, pos_inds.sum())
        return loss * factor


class reg_loss(nn.Module):
    def __init__(self):
        super(reg_loss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')
        #self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, weight, factor=1.0):
        pred_norm = pred / (target + 1e-8)
        target_norm = target / (target + 1e-8)
        loss = weight * self.smoothl1(pred_norm, target_norm)
        loss = loss.sum() / max(1.0, weight.sum())
        return loss * factor


class off_loss(nn.Module):
    def __init__(self):
        super(off_loss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target, weight, factor=1.0):
        loss = weight * self.smoothl1(pred, target)
        loss = loss.sum() / max(1.0, weight.sum())
        return loss * factor


class den_loss(nn.Module):
    def __init__(self):
        super(den_loss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, factor=1.0):
        loss = self.mse(pred, target)
        loss = loss.sum() / target.size(0)
        return loss * factor


class dnc_loss(nn.Module):
    def __init__(self, size, down, ratio=0.41):
        super(dnc_loss, self).__init__()
        self.size = size
        self.down = down
        self.ratio = ratio
        self.prroi = PrRoIPool(out_size=1, spatial_scale=1.0)

    def forward(self, cls_preds, reg_preds, off_preds, den_preds,
                fboxes, hboxes, ctr_masks, post_probs, factor=1.0):

        batch_size, _, H, W = cls_preds.size()

        loss = 0
        for i in range(batch_size):
            cls = cls_preds[i]
            reg = reg_preds[i]
            off = off_preds[i]
            den = den_preds[i]
            ctr_mask = ctr_masks[i]
            post_prob = post_probs[i]

            fboxes_per_img = fboxes[i]
            hboxes_per_img = hboxes[i]
            fboxes_per_img = fboxes_per_img[fboxes_per_img[:, -1] != -1]
            hboxes_per_img = hboxes_per_img[hboxes_per_img[:, -1] != -1]

            mask_full = fboxes_per_img[:, -1] > 0
            num_valid_full = torch.nonzero(mask_full).numel()
            mask_head = hboxes_per_img[:, -1] > 0
            num_valid_head = torch.nonzero(mask_head).numel()
            if num_valid_head > 0:
                # h = (gts[:, 3] - gts[:, 1]) / self.down
                # w = (gts[:, 2] - gts[:, 0]) / self.down
                #
                # score_map = cls * ctr_mask
                # x_c, y_c = soft_argmax(score_map, ctr_mask, temperature=None)
                # c2h = 0.4 * h
                # x_h = x_c.clamp(min=0, max=feat_size[1] - 1)
                # y_h = (y_c - c2h).clamp(min=0, max=feat_size[0] - 1)
                #
                # x1 = (x_h + x_o + 0.5 - w / 2).clamp(min=0, max=feat_size[1])
                # y1 = (y_h + y_o + 0.5 - h / 2).clamp(min=0, max=feat_size[0])
                # x2 = (x_h + x_o + 0.5 - w / 2).clamp(min=0, max=feat_size[1])
                # y2 = (y_h + y_o + 0.5 - h / 2).clamp(min=0, max=feat_size[0])
                # bboxes = torch.stack([x1, y1, x2, y2], -1)

                bboxes = hboxes_per_img[mask_head, :-1] / self.down

                area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                rois = bboxes.new_zeros((num_valid_head, 5))
                rois[:, 1:] = bboxes
                roi_density = self.prroi(den[None, :] * post_prob, rois)
                counts = [
                    area[i] * roi_density[i, i].squeeze()
                    for i in range(num_valid_head)
                ]
                counts = torch.stack(counts)
                gt_counts = counts.new_ones((num_valid_head,))
                loss += torch.abs(counts - gt_counts).sum() / num_valid_head
            else:
                loss += 0 * den.sum()
        return loss * factor


# class c2d_loss(nn.Module):
#     def __init__(self, size, down, ratio=0.41):
#         super(c2d_loss, self).__init__()
#         self.size = size
#         self.down = down
#         self.ratio = ratio
#
#     def forward(self, cls_preds, reg_preds, offset_preds, density_preds,
#                 gt_bboxes, ctr_masks, post_probs, factor=1.0):
#
#         batch_size = gt_bboxes.shape[0]
#         feat_size = reg_preds.size()[-2:]
#
#         loss = 0
#         for i in range(batch_size):
#             density = density_preds[i]
#             post_prob = post_probs[i]
#
#             gts = gt_bboxes[i]
#             gts = gts[gts[:, 3] != -1]
#
#             num_full = gts.shape[0]
#             if num_full > 0:
#                 h = gts[:, 3] - gts[:, 1]
#                 w = gts[:, 2] - gts[:, 0]
#
#                 score_map = density * post_prob
#                 x_h, y_h = soft_argmax(score_map, post_prob, temperature=None)
#                 h2c = 0.4 * h / self.down
#                 x_c = x_h.clamp(min=0, max=feat_size[1] - 1)
#                 y_c = (y_h + h2c).clamp(min=0, max=feat_size[0] - 1)
#
#                 x1 = (self.down * x_c - w / 2).clamp(min=0, max=self.size[1])
#                 y1 = (self.down * y_c - h / 2).clamp(min=0, max=self.size[0])
#                 x2 = (self.down * x_c + w / 2).clamp(min=0, max=self.size[1])
#                 y2 = (self.down * y_c + h / 2).clamp(min=0, max=self.size[0])
#                 bboxes = torch.stack([x1, y1, x2, y2], -1)
#
#                 ious = bbox_overlaps(bboxes, gts).clamp(min=1e-6)
#                 loss += -ious.log().sum() / num_full
#             else:
#                 loss += 0 * density.sum()
#         return loss * factor


# class d2c_loss(nn.Module):
#     def __init__(self, size, down, ratio=0.41):
#         super(d2c_loss, self).__init__()
#         self.size = size
#         self.down = down
#         self.ratio = ratio
#         self.prroi = PrRoIPool(out_size=1, spatial_scale=1.0)
#
#     def forward(self, cls_preds, reg_preds, offset_preds, density_preds,
#                 gt_bboxes, ctr_masks, post_probs, factor=1.0):
#
#         batch_size = gt_bboxes.shape[0]
#         feat_size = reg_preds.size()[-2:]
#
#         loss = 0
#         for i in range(batch_size):
#             cls = cls_preds[i]
#             density = density_preds[i]
#             ctr_mask = ctr_masks[i]
#             post_prob = post_probs[i]
#
#             gts = gt_bboxes[i]
#             gts = gts[gts[:, 3] != -1]
#
#             num_full = gts.shape[0]
#             if num_full > 0:
#                 h = (gts[:, 3] - gts[:, 1]) / self.down
#                 w = (gts[:, 2] - gts[:, 0]) / self.down
#
#                 score_map = cls * ctr_mask
#                 x_c, y_c = soft_argmax(score_map, ctr_mask, temperature=None)
#                 c2h = 0.4 * h
#                 x_h = x_c.clamp(min=0, max=feat_size[1] - 1)
#                 y_h = (y_c - c2h).clamp(min=0, max=feat_size[0] - 1)
#
#                 x1 = (x_h - w / 2).clamp(min=0, max=feat_size[1])
#                 y1 = (y_h - h / 2).clamp(min=0, max=feat_size[0])
#                 x2 = (x_h + w / 2).clamp(min=0, max=feat_size[1])
#                 y2 = (y_h + h / 2).clamp(min=0, max=feat_size[0])
#                 bboxes = torch.stack([x1, y1, x2, y2], -1)
#
#                 area = (x2 - x1) * (y2 - y1)
#                 rois = bboxes.new_zeros((num_full, 5))
#                 rois[:, 1:] = bboxes
#                 roi_density = self.prroi(density[None, :] * post_prob, rois)
#                 counts = [
#                     area[i] * roi_density[i, i].squeeze()
#                     for i in range(num_full)
#                 ]
#                 counts = torch.stack(counts)
#                 gt_counts = counts.new_ones((num_full,))
#                 loss += torch.abs(counts - gt_counts).sum() / num_full
#             else:
#                 loss += 0 * density.sum()
#         return loss * factor