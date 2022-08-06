# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from ...core.utils import flip_tensor
from .single_stage import SingleStageDetector

import os
import json
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
import numpy as np
import cv2


def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt

@DETECTORS.register_module()
class E2EC(SingleStageDetector):
    """Implementation of E2EC
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(E2EC, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_masks,
                      gt_labels,
                      data_input,
                      gt_bboxes_ignore=None):
            """
            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                    Typically these should be mean centered and std scaled.
                img_metas (list[dict]): A List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    :class:`mmdet.datasets.pipelines.Collect`.
                gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                    image in [tl_x, tl_y, br_x, br_y] format.
                gt_labels (list[Tensor]): Class indices corresponding to each box
                gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                    boxes can be ignored when computing the loss.

            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            """
            super(SingleStageDetector, self).forward_train(img, img_metas)
            x = self.extract_feat(img)

            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_masks,
                                                  gt_labels, data_input, gt_bboxes_ignore)
            return losses

    def det_eval(self, img, img_metas, output):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach()
        if len(py) == 0:
            return
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()

        height, width, _ = img_metas['ori_shape']
        center = np.array([width / 2., height / 2.], dtype=np.float32)
        scale = None #coco
        test_scale = None
        test_rescale = None

        scale = np.array([width, height])
        x = 32
        if test_rescale is not None:
            input_w, input_h = int((width / test_rescale + x - 1) // x * x), \
                               int((height / test_rescale + x - 1) // x * x)
        else:
            if test_scale is None:
                input_w = (int(width / 1.) | (x - 1)) + 1
                input_h = (int(height / 1.) | (x - 1)) + 1
            else:
                scale = max(width, height) * 1.0
                scale = np.array([scale, scale])
                input_w, input_h = test_scale
        center = np.array([width // 2, height // 2])

        center = torch.tensor(center)
        scale = torch.tensor(scale)
        center = center.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        if len(box) == 0:
            return

        trans_output_inv = get_affine_transform(center, scale, 0, [input_w, input_h], inv=1)

        det_bboxes = []
        for i in range(len(label)):
            box_ = affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            bscore = float('{:.2f}'.format(score[i]))
            box_.append(bscore)
            det_bboxes.append(box_)

        det_bboxes = torch.tensor(det_bboxes)
        det_labels = torch.tensor(label)

        # det_bboxes, det_labels = self.bbox_head._bboxes_nms(det_bboxes, det_labels, self.test_cfg)

        return [(det_bboxes, det_labels)]

    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, wh_preds, offset_preds = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(wh_preds) == len(
                offset_preds) == 1

            # Feature map averaging
            center_heatmap_preds[0] = (center_heatmap_preds[0][0:1] +
                                       flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            wh_preds[0] = (wh_preds[0][0:1] +
                           flip_tensor(wh_preds[0][1:2], flip_direction)) / 2

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                wh_preds, [offset_preds[0][0:1]],
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        output = self.bbox_head.test_pipe(feat)
        results_list = self.det_eval(img, img_metas, output)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = imgs.shape[0]
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            img_metas[0]['batch_input_shape'] = tuple(imgs.shape[-2:])

            return self.simple_test(imgs, img_metas[0], **kwargs)
        else:
            # TODO: support test-time augmentation
            assert NotImplementedError