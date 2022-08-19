import numpy as np
import cv2
import random
import importlib
from shapely.geometry import Polygon
from mmdet.datasets import PIPELINES

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

def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std,
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
        w_border = get_border(width/4, scale[0]) + 1
        h_border = get_border(height/4, scale[0]) + 1
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
            input_w, input_h = int((width / test_rescale + x - 1) // x * x),\
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

    trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        color_aug(_data_rng, inp, _eig_val, _eig_vec)

    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // down_ratio, input_w // down_ratio
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw

@PIPELINES.register_module()
class Augment:
    def __init__(self, mode='test', dataset_type=None):
        dataset_dict = {'CocoDataset': 'coco', 'CityscapesDataset': 'cityscapes'}
        self._cfg = importlib.import_module('args.' + dataset_dict[dataset_type])
        self.mode = mode

    def __call__(self, results):

        img = results['img']

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.mode,
                self._cfg.data.data_rng, self._cfg.data.eig_val, self._cfg.data.eig_vec,
                self._cfg.data.mean, self._cfg.data.std, self._cfg.commen.down_ratio,
                self._cfg.data.input_h, self._cfg.data.input_w, self._cfg.data.scale_range,
                self._cfg.data.scale, self._cfg.test.test_rescale, self._cfg.data.test_scale
            )
        meta = {'inp': inp, 'center': center, 'scale': scale}
        inp = inp.transpose(1, 2, 0)
        results['img'] = inp
        results['meta'] = meta

        return results