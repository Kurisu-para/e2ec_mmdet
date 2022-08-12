# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmcv.runner import force_fp32, auto_fp16

from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor

from ...core.utils import flip_tensor
from .single_stage import SingleStageDetector
import pycocotools.mask as mask_utils
import numpy as np
import cv2
import random
from shapely.geometry import Polygon
from args import coco as cfg

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

def polygons_to_mask(height, width, polygons):
    mask = np.zeros((height, width), dtype=np.int32)
    polygons = np.array(polygons, dtype=np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # shape = polygons.shape
    # polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, [polygons], 1) # 非int32 会报错
    return mask

def mask2result(masks, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        masks (torch.Tensor | np.ndarray): shape (n, img_h, img_w)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): mask results of each class
    """
    if masks.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [masks[labels == i, :] for i in range(num_classes)]

@DETECTORS.register_module()
class E2EC(SingleStageInstanceSegmentor):
    """Implementation of E2EC
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(E2EC, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

        self._cfg = cfg

    @auto_fp16(apply_to=('data_input["inp"]','data_input["ct_hm"]','data_input["wh"]','data_input["ct_cls"]','data_input["ct_ind"]',
                         'data_input["img_gt_polys"]','data_input["can_gt_polys"]','data_input["keyPointsMask"]'))
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
            x = self.extract_feat(data_input['inp'])
            losses = self.mask_head.forward_train(x, img_metas, gt_bboxes, gt_masks,
                                                  gt_labels, data_input, gt_bboxes_ignore)
            return losses

    @force_fp32(apply_to=('output["detection"]', 'output["py"]'))
    def det_eval(self, meta, output):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach()
        if len(py) == 0:
            return
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()

        center = meta['center'][0].detach().cpu().numpy()
        scale = meta['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return

        b, c, h, w = meta['inp'].shape
        trans_output_inv = get_affine_transform(center, scale, 0, [w, h], inv=1)

        det_bboxes = []
        for i in range(len(label)):
            box_ = affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            bscore = float('{:.2f}'.format(score[i]))
            box_.append(bscore)
            det_bboxes.append(box_)

        det_bboxes = np.array(det_bboxes)
        det_labels = label

        return [(det_bboxes, det_labels)]

    @force_fp32(apply_to=('output["detection"]', 'output["py"]'))
    def seg_eval(self, meta, output, img_metas):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy()

        if len(py) == 0:
            return

        center = meta['center'][0].detach().cpu().numpy()
        scale = meta['scale'][0].detach().cpu().numpy()

        b, c, h, w = meta['inp'].shape
        trans_output_inv = get_affine_transform(center, scale, 0, [w, h], inv=1)
        ori_h, ori_w, ori_c = img_metas['ori_shape']
        py = [affine_transform(py_, trans_output_inv) for py_ in py]
        # rles = coco_poly_to_rle(py, ori_h, ori_w)
        masks = [polygons_to_mask(ori_h, ori_w, p) for p in py]
        seg_masks = np.array(masks)
        seg_labels = label

        return [(seg_masks, seg_labels)]

    # def simple_test(self, img, img_metas, meta, rescale=False): #纯detection
    #     """Test function without test-time augmentation.
    #
    #     Args:
    #         img (torch.Tensor): Images with shape (N, C, H, W).
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.
    #
    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     feat = self.extract_feat(img)
    #     output = self.mask_head.test_pipe(feat)
    #     results_list = self.det_eval(meta, output)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in results_list
    #     ]
    #     return bbox_results

    def simple_test(self, img, img_metas, meta, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (B, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list(tuple): Formatted bbox and mask results of multiple \
                images. The outer list corresponds to each image. \
                Each tuple contains two type of results of single image:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        feat = self.extract_feat(img)
        output = self.mask_head.test_pipe(feat)
        results_list = self.det_eval(meta, output)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.mask_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        mask_results_list = self.seg_eval(meta, output, img_metas)
        mask_results = [
            mask2result(seg_masks, seg_labels, self.mask_head.num_classes)
            for seg_masks, seg_labels in mask_results_list
        ]
        format_results_list = []
        for bbox_result, mask_result in zip(bbox_results, mask_results):
            format_results_list.append((bbox_result, mask_result))

        return format_results_list

    @auto_fp16(apply_to=('imgs', ))
    def forward_test(self, imgs, img_metas, meta, **kwargs):
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

            return self.simple_test(imgs, img_metas[0], meta, **kwargs)
        else:
            # TODO: support test-time augmentation
            assert NotImplementedError