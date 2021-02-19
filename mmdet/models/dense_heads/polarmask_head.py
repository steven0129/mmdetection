import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, cross_class_nms
# from mmdet.ops import ModulatedDeformConvPack

from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob
from IPython import embed
import cv2
import numpy as np
import math
import onnx
import time

INF = 1e8


@HEADS.register_module
class PolarMask_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 use_dcn=False,
                 mask_nms=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_mask=dict(type='MaskIOULoss'),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 center_sample=True,
                 use_mask_center=True,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(PolarMask_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_mask = build_loss(loss_mask)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        # xez add for polarmask
        self.use_dcn = use_dcn
        self.mask_nms = mask_nms

        # debug vis img
        self.vis_num = 1000
        self.count = 0

        # test
        self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi

        # Polar Target Generation Config
        self.center_sample = center_sample
        self.use_mask_center = use_mask_center
        self.radius = 1.5
        self.strides = strides
        self.strides = strides
        self.regress_ranges = regress_ranges

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.polar_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.polar_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.polar_mask = nn.Conv2d(self.feat_channels, 36, 3, padding=1)
        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scales_mask = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        if not self.use_dcn:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.mask_convs:
                normal_init(m.conv, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
        normal_init(self.polar_mask, std=0.01)
        normal_init(self.polar_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_bbox, self.scales_mask)

    def forward_single(self, x, scale_bbox, scale_mask):
        cls_feat = x
        reg_feat = x
        mask_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)
        centerness = self.polar_centerness(cls_feat)

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_pred = scale_mask(self.polar_mask(mask_feat)).float().exp()

        return cls_score, centerness, mask_pred

    @force_fp32(apply_to=('cls_scores', 'mask_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             centernesses,
             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_masks,
             gt_bboxes_ignore=None,
             extra_data=None):
        assert len(cls_scores) == len(centernesses) == len(mask_preds)
        labels, _, mask_targets, all_level_points, normal_set = self.polar_target(cls_scores, gt_bboxes, gt_masks, gt_labels)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        for idx, cls_score in enumerate(cls_scores):
            new_cls_score = []
            for valid_idx in normal_set:
                new_cls_score.append(cls_score[valid_idx].unsqueeze(0))

            cls_scores[idx] = torch.cat(new_cls_score)

        for idx, centerness in enumerate(centernesses):
            new_centerness = []
            for valid_idx in normal_set:
                new_centerness.append(centerness[valid_idx].unsqueeze(0))

            centernesses[idx] = torch.cat(new_centerness)

        for idx, mask_pred in enumerate(mask_preds):
            new_mask_pred = []
            for valid_idx in normal_set:
                new_mask_pred.append(mask_pred[valid_idx].unsqueeze(0))

            mask_preds[idx] = torch.cat(new_mask_pred)
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_mask_preds = [
            mask_pred.permute(0, 2, 3, 1).reshape(-1, 36)
            for mask_pred in mask_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 80]
        flatten_mask_preds = torch.cat(flatten_mask_preds)  # [num_pixel, 36]
        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]

        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 36]
        flatten_points = torch.cat([points.repeat(num_imgs, 1)
                                    for points in all_level_points])  # [num_pixel,2]
        pos_inds = (flatten_labels != (self.num_classes - 1)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        pos_centerness = flatten_centerness[pos_inds]
        pos_mask_preds = flatten_mask_preds[pos_inds]

        if num_pos > 0:
            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_mask_targets = pos_mask_targets.cuda()
            pos_centerness_targets = self.polar_centerness_target(pos_mask_targets)
            pos_centerness_targets = pos_centerness_targets.cuda()

            pos_points = flatten_points[pos_inds]
            loss_mask = self.loss_mask(pos_mask_preds,
                                       pos_mask_targets,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_mask = pos_mask_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_mask=loss_mask,
            loss_centerness=loss_centerness)

    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points, regress_ranges, num_points_per_level):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        #xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)   #feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]

        #mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []

        #第一步 先算重心  return [num_gt, 2]
        for mask in gt_masks:
            single_cnt = self.get_single_centerpoint(mask)
            if single_cnt != None:
                cnt, contour = single_cnt
                contour = contour[0]
                contour = torch.Tensor(contour).float().cuda()
                y, x = cnt
                mask_centers.append([x,y])
                mask_contours.append(contour)
            else:
                return None
        mask_centers = torch.Tensor(mask_centers).float().cuda()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        #加入center sample
        if self.center_sample:
            strides = [8, 16, 32, 64, 128]
            if self.use_mask_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(gt_bboxes,
                                                             mask_centers,
                                                             strides,
                                                             num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
            else:
                raise NotImplementedError('Please use mask center sampling...')
                exit()
        
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]

        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes - 1

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = (labels != self.num_classes - 1).nonzero().reshape(-1)
        mask_targets = torch.zeros(num_points, 36).float()

        pos_mask_ids = min_area_inds[pos_inds]
        for p,id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]

            dists, coords = self.get_36_coordinates(x, y, pos_mask_contour)
            mask_targets[p] = dists

        return labels, bbox_targets, mask_targets

    def get_single_centerpoint(self, mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        try:
            count = contour[0][:, 0, :]
        except IndexError:
            print('[PolarMask_Head] Index error after cv2.findContours() and remove this data')
            return None
        
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]

        # max_points = 360
        # if len(contour[0]) > max_points:
        #     compress_rate = len(contour[0]) // max_points
        #     contour[0] = contour[0][::compress_rate, ...]
        return center, contour

    def get_mask_sample_region(self, gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        #no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level,n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        #生成36个角度
        new_coordinate = {}
        for i in range(0, 360, 10):
            if i in angle:
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                new_coordinate[i] = d


        distances = torch.zeros(36)

        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a//10] = 1e-6
            else:
                distances[a//10] = new_coordinate[a]
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist

        return distances, new_coordinate

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points.cuda()

    def polar_target(self, cls_scores, gt_bboxes, gt_masks, gt_labels):
        num_batch = cls_scores[0].size(0)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, cls_scores[0].dtype, cls_scores[0].device)
        num_points_per_level = [i.size()[0] for i in all_level_points]
        expanded_regress_ranges = [
            all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                all_level_points[i]) for i in range(num_levels)
        ]

        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)

        labels_list = []
        bbox_targets_list = []
        mask_targets_list = []
        broken_batch_idx = []

        for batch_idx in range(num_batch):
            gt_bbox = gt_bboxes[batch_idx]
            gt_mask = gt_masks[batch_idx]
            gt_label = gt_labels[batch_idx]

            polar_target_single_result = self.polar_target_single(
                gt_bbox, gt_mask, gt_label, concat_points, concat_regress_ranges, num_points_per_level
            )

            if polar_target_single_result != None:
                _labels, _bbox_targets, _mask_targets = polar_target_single_result
                labels_list.append(_labels)
                bbox_targets_list.append(_bbox_targets)
                mask_targets_list.append(_mask_targets)
            else:
                labels_list.append(torch.zeros([sum(num_points_per_level)]))
                bbox_targets_list.append(torch.zeros([sum(num_points_per_level), 4]))
                mask_targets_list.append(torch.zeros([sum(num_points_per_level), 36]))
                broken_batch_idx.append(batch_idx)
        
        labels_list = list(map(lambda x: x.cuda(), labels_list))
        bbox_targets_list = list(map(lambda x: x.cuda(), bbox_targets_list))
        mask_targets_list = list(map(lambda x: x.cuda(), mask_targets_list))

        batch_set = set(range(cls_scores[0].size(0)))
        broken_set = set(broken_batch_idx)
        normal_set = batch_set - broken_set

        # split to per img, per level
        labels_list_lvl = []
        bbox_targets_list_lvl = []
        mask_targets_list_lvl = []

        for idx in normal_set:
            labels_list_lvl.append(labels_list[idx].split(num_points_per_level, 0))
            bbox_targets_list_lvl.append(bbox_targets_list[idx].split(num_points_per_level, 0))
            mask_targets_list_lvl.append(mask_targets_list[idx].split(num_points_per_level, 0))

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_mask_targets = []

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list_lvl]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list_lvl]))
            concat_lvl_mask_targets.append(
                torch.cat(
                    [mask_targets[i] for mask_targets in mask_targets_list_lvl]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets, all_level_points, normal_set

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets) + 1e-8

    @force_fp32(apply_to=('cls_scores', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   centernesses,
                   mask_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        num_levels = len(cls_scores)

        cls_scores = [cls_score.cuda() for cls_score in cls_scores]
        centernesses = [centerness.cuda() for centerness in centernesses]
        mask_preds = [mask_pred.cuda() for mask_pred in mask_preds]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, mask_preds[0].dtype,
                                      mask_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                mask_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list,
                                                mask_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          mask_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(mlvl_points)
        mlvl_scores = []
        mlvl_masks = []
        mlvl_centerness = []
        for cls_score, mask_pred, centerness, points in zip(
                cls_scores, mask_preds, centernesses, mlvl_points):
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, 36)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            masks = distance2mask(points, mask_pred, self.angles, max_shape=img_shape)

            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_masks.append(masks)

        mlvl_masks = torch.cat(mlvl_masks)
        
        try:
            scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(1).repeat(1, 36)
            _mlvl_masks = mlvl_masks / scale_factor
        except:
            _mlvl_masks = mlvl_masks / mlvl_masks.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)  # background class index is after all class index
        mlvl_centerness = torch.cat(mlvl_centerness)

        centerness_factor = 0.5  # mask centerness is smaller than origin centerness, so add a constant is important or the score will be too low.
        '''1 mask->min_bbox->nms, performance same to origin box'''
        a = _mlvl_masks
        _mlvl_bboxes = torch.stack([a[:, 0].min(1)[0],a[:, 1].min(1)[0],a[:, 0].max(1)[0],a[:, 1].max(1)[0]],-1)
        det_bboxes, det_labels, keep = cross_class_nms(
            _mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness + centerness_factor,
            return_inds=True
        )

        det_masks = _mlvl_masks[keep]

        return det_bboxes, det_labels, det_masks


# test
def distance2mask(points, distances, angles, max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, 36)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :].repeat(num_points, 1)
    cos = cos[None, :].repeat(num_points, 1)

    x = distances * sin + c_x
    y = distances * cos + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
    return res



