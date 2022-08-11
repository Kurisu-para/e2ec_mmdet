import os.path as osp
import mmcv
import numpy as np
import cv2
import random
import math
import copy
import pycocotools.mask as maskUtils
from shapely.geometry import Polygon
from torch.utils.data.dataloader import default_collate
import torch

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

from args import coco as cfg


class Douglas:
    D = 3

    def sample(self, poly):
        mask = np.zeros((poly.shape[0],), dtype=int)
        mask[0] = 1
        endPoint = poly[0: 1, :] + poly[-1:, :]
        endPoint /= 2
        poly_append = np.concatenate([poly, endPoint], axis=0)
        self.compress(0, poly.shape[0], poly_append, mask)
        return mask

    def compress(self, idx1, idx2, poly, mask):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        if (n == m + 1):
            return
        d = abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / math.sqrt(
            math.pow(A, 2) + math.pow(B, 2) + 1e-4)
        max_idx = np.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        if dmax > self.D:
            mask[max_idx] = 1
            self.compress(idx1, max_idx, poly, mask)
            self.compress(max_idx, idx2, poly, mask)


@PIPELINES.register_module()
class Contour:
    def __init__(self):
        self._cfg = cfg
        self.d = Douglas()

    def get_border(self, border, size):
        i = 1
        while np.any(size - border // i <= border // i):
            i *= 2
        return border // i

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_affine_transform(self,
                             center,
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
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def lighting_(self, data_rng, image, alphastd, eigval, eigvec):
        alpha = data_rng.normal(scale=alphastd, size=(3,))
        image += np.dot(eigvec, eigval * alpha)

    def blend_(self, alpha, image1, image2):
        image1 *= alpha
        image2 *= (1 - alpha)
        image1 += image2

    def saturation_(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        self.blend_(alpha, image, gs[:, :, None])

    def brightness_(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image *= alpha

    def contrast_(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        self.blend_(alpha, image, gs_mean)

    def color_aug(self, data_rng, image, eig_val, eig_vec):
        functions = [self.brightness_, self.contrast_, self.saturation_]
        random.shuffle(functions)
        gs = self.grayscale(image)
        gs_mean = gs.mean()
        for f in functions:
            f(data_rng, image, gs, gs_mean, 0.4)
        self.lighting_(data_rng, image, 0.1, eig_val, eig_vec)

    def augment(self, img, split, _data_rng, _eig_val, _eig_vec, mean, std,
                down_ratio, input_h, input_w, scale_range, scale=None, test_rescale=None, test_scale=None):
        height, width = img.shape[0], img.shape[1]
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if scale is None:
            scale = max(img.shape[0], img.shape[1]) * 1.0
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        flipped = False
        if split == 'train':
            scale = scale * (random.random() * (scale_range[1] - scale_range[0]) + scale_range[0])
            x, y = center
            w_border = self.get_border(width / 4, scale[0]) + 1
            h_border = self.get_border(height / 4, scale[0]) + 1
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, width - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, height - 1))

            if random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        if split != 'train':
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

        trans_input = self.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        orig_img = inp.copy()
        inp = (inp.astype(np.float32) / 255.)
        if split == 'train':
            self.color_aug(_data_rng, inp, _eig_val, _eig_vec)

        inp = (inp - mean) / std
        inp = inp.transpose(2, 0, 1)

        output_h, output_w = input_h // down_ratio, input_w // down_ratio
        trans_output = self.get_affine_transform(center, scale, 0, [output_w, output_h])
        inp_out_hw = (input_h, input_w, output_h, output_w)

        return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw

    def affine_transform(self, pt, t):
        """pt: [n, 2]"""
        new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
        return new_pt

    def handle_break_point(self, poly, axis, number, outside_border):
        if len(poly) == 0:
            return []

        if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
            return []

        break_points = np.argwhere(
            outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
        if len(break_points) == 0:
            return poly

        new_poly = []
        if not outside_border(poly[break_points[0], axis], number):
            new_poly.append(poly[:break_points[0]])

        for i in range(len(break_points)):
            current_poly = poly[break_points[i]]
            next_poly = poly[break_points[i] + 1]
            mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (
                    next_poly[axis] - current_poly[axis])

            if outside_border(poly[break_points[i], axis], number):
                if mid_poly[axis] != next_poly[axis]:
                    new_poly.append([mid_poly])
                next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
                new_poly.append(poly[break_points[i] + 1:next_point])
            else:
                new_poly.append([poly[break_points[i]]])
                if mid_poly[axis] != current_poly[axis]:
                    new_poly.append([mid_poly])

        if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
            current_poly = poly[-1]
            next_poly = poly[0]
            mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (
                    next_poly[axis] - current_poly[axis])
            new_poly.append([mid_poly])

        return np.concatenate(new_poly)

    def transform_polys(self, polys, trans_output, output_h, output_w):
        new_polys = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = self.affine_transform(poly, trans_output)
            poly = self.handle_break_point(poly, 0, 0, lambda x, y: x < y)
            poly = self.handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
            poly = self.handle_break_point(poly, 1, 0, lambda x, y: x < y)
            poly = self.handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
            if len(poly) == 0:
                continue
            if len(np.unique(poly, axis=0)) <= 2:
                continue
            new_polys.append(poly)
        return new_polys

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = self.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def filter_tiny_polys(self, polys):
        return [poly for poly in polys if Polygon(poly).area > 5]

    def get_cw_polys(self, polys):
        return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = self.filter_tiny_polys(instance)
            polys = self.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        if b3 ** 2 - 4 * a3 * c3 < 0:
            r3 = min(r1, r2)
        else:
            sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
            r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian2D(self, shape, sigma=(1, 1), rho=0):
        if not isinstance(sigma, tuple):
            sigma = (sigma, sigma)
        sigma_x, sigma_y = sigma

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
        h = np.exp(-energy / (2 * (1 - rho * rho)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)  # 四舍五入

        h, w = y_max - y_min, x_max - x_min
        radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        self.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])  # y*128+x

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def uniformsample(self, pgtnp_px2, newpnum):
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    def four_idx(self, img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0]
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]

    def get_img_gt(self, img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align]
            if begin > end:
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        r = np.concatenate(r, axis=0)
        return img_gt_poly[r, :]

    def img_poly_to_can_poly(self, img_poly):
        x_min, y_min = np.min(img_poly, axis=0)
        can_poly = img_poly - np.array([x_min, y_min])
        return can_poly

    def get_keypoints_mask(self, img_gt_poly):
        key_mask = self.d.sample(img_gt_poly)
        return key_mask

    def prepare_evolution(self, poly, img_gt_polys, can_gt_polys, keyPointsMask):
        img_gt_poly = self.uniformsample(poly, len(poly) * self._cfg.data.points_per_poly)
        idx = self.four_idx(img_gt_poly)
        img_gt_poly = self.get_img_gt(img_gt_poly, idx)
        can_gt_poly = self.img_poly_to_can_poly(img_gt_poly)
        key_mask = self.get_keypoints_mask(img_gt_poly)
        keyPointsMask.append(key_mask)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def __call__(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_labels = results['ann_info']['labels']
        gt_masks = results['ann_info']['masks']
        gt_contour_masks = copy.deepcopy(gt_masks)

        img = results['img']
        assert img.shape[1] == w and img.shape[0] == h
        width, height = img.shape[1], img.shape[0]

        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance] for instance in gt_contour_masks]
        cls_ids = gt_labels.tolist()

        image_id = results['img_info']['id']

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            self.augment(
                img, 'train',
                self._cfg.data.data_rng, self._cfg.data.eig_val, self._cfg.data.eig_vec,
                self._cfg.data.mean, self._cfg.data.std, self._cfg.commen.down_ratio,
                self._cfg.data.input_h, self._cfg.data.input_w, self._cfg.data.scale_range,
                self._cfg.data.scale, self._cfg.test.test_rescale, self._cfg.data.test_scale
            )

        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)

        data_input = {}

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([self._cfg.model.heads['ct_hm'], output_h, output_w], dtype=np.float32)
        ct_cls = []
        wh = []
        ct_ind = []

        # segmentation
        img_gt_polys = []
        keyPointsMask = []
        can_gt_polys = []

        for i, instance_poly in enumerate(instance_polys):
            cls_id = cls_ids[i]
            for j, poly in enumerate(instance_poly):  # 一个instance_ploy的所有poly，因为有断开的
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_evolution(poly, img_gt_polys, can_gt_polys, keyPointsMask)

        data_input.update({'inp': inp})
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys}
        data_input.update(detection)
        data_input.update(evolution)
        data_input.update({'keypoints_mask': keyPointsMask})
        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': image_id, 'ann': image_id, 'ct_num': ct_num}  # ann
        data_input.update({'meta': meta})
        results['data_input'] = data_input

        return results