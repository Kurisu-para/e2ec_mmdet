# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
import numpy as np
from args import coco as cfg
# import cv2
# import random
# from shapely.geometry import Polygon
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)

class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):   # state_dim=128, feature_dim=64+2, conv_type='dgrid'
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)

_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        if conv_type == 'grid':
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj)
        else:
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
        super(Snake, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim, conv_type)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i])
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256

        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x):
        states = []
        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        x = self.prediction(state)

        return x

def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly

def prepare_training(ret, batch, ro):  # output, batch, self.ro = 4.
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
    init.update({'img_init_polys': ret['poly_coarse'].detach() / ro})
    can_init_polys = img_poly_to_can_poly(ret['poly_coarse'].detach() / ro)
    init.update({'can_init_polys': can_init_polys})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['py_ind']})
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature

def prepare_testing_init(polys, ro):
    polys = polys / ro
    can_init_polys = img_poly_to_can_poly(polys)
    img_init_polys = polys
    ind = torch.zeros((img_init_polys.size(0), ), dtype=torch.int32, device=img_init_polys.device)
    init = {'img_init_polys': img_init_polys, 'can_init_polys': can_init_polys, 'py_ind': ind}
    return init

class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4.):
        super(Evolution, self).__init__()
        assert evole_ietr_num >= 1
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
        self.iter = evole_ietr_num - 1  # 2
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w - 1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h - 1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1.,
                    ignore=False):  # evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], stride=self.evolve_stride
        if ignore:
            return i_it_poly * self.ro
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = snake(init_input).permute(0, 2, 1)
        i_poly = i_it_poly * self.ro + offset * stride
        return i_poly

    def foward_train(self, output, batch, cnn_feature):
        ret = output
        init = self.prepare_training(output, batch)
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['img_init_polys'],
                                   init['can_init_polys'], init['py_ind'], stride=self.evolve_stride)
        py_preds = [py_pred]
        for i in range(self.iter):  # 2
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                       init['py_ind'], stride=self.evolve_stride)
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys'] * self.ro})
        return output

    def foward_test(self, output, cnn_feature, ignore):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py = self.evolve_poly(self.evolve_gcn, cnn_feature, img_init_polys, init['can_init_polys'], init['py_ind'],
                                  ignore=ignore[0], stride=self.evolve_stride)
            pys = [py, ]
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                      ignore=ignore[i + 1], stride=self.evolve_stride)
                pys.append(py)
            ret.update({'py': pys})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:
            self.foward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.foward_test(output, cnn_feature, ignore=ignore)
        return output

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode_ct_hm(ct_hm, wh, reg=None, K=100, stride=10.):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, -1, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride
    detection = torch.cat([ct, scores, clses], dim=2)
    return poly, detection

def clip_to_image(poly, h, w):
    poly[..., :2] = torch.clamp(poly[..., :2], min=0)
    poly[..., 0] = torch.clamp(poly[..., 0], max=w-1)
    poly[..., 1] = torch.clamp(poly[..., 1], max=h-1)
    return poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w): #feature shape [16, 64, 128, 128], points shape [154,128+1,2], ct_img_idx, h, w
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature

class Refine(torch.nn.Module):  # cnn_feature, ct, init_polys, ct_img_idx.clone()
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx,
                ignore=False):  # feature=cnn_feature, ct_polys=ct, init_polys=init_polys, ct_img_idx=ct_img_idx.clone()
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)
        poly_num = ct_polys.size(0)

        feature = self.trans_feature(feature)  # [16,64,128,128] -> [16,64,128,128]

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)  # [154,128+1,2] 中心点和边框点的坐标
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys

class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)

    def train_decode(self, data_input, output, cnn_feature):  # data_input=batch, cnn_feature=cnn_feature, ouput=output
        wh_pred = output['wh']  # torch.Size([16, 256, 128, 128])  coco 128*2
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]

        ct_img_idx = data_input['ct_img_idx'][ct_01]
        _, _, height, width = data_input['ct_hm'].size()  # 128,128
        ct_x, ct_y = ct_ind % width, ct_ind // width

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

        ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
        ct = torch.cat([ct_x, ct_y], dim=1)

        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        coarse_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())  # global deformation

        output.update({'poly_init': init_polys * self.down_sample})
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return

    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride)
        valid = detection[0, :, 2] >= min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})

        img_id = torch.zeros((len(poly_init),), dtype=torch.int64)
        poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'detection': detection})
        return

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            self.train_decode(data_input, output, cnn_feature)
        else:
            self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform)


@HEADS.register_module()
class E2ECHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel, #64
                 feat_channel, #64
                 num_classes, #80 coco
                 points_per_poly = 128,
                 down_sample = 4.,
                 init_stride=10.,
                 coarse_stride=4.,
                 min_ct_score=0.05,
                 evole_ietr_num=3,
                 evolve_stride=1,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_init=dict(type='SmoothL1Loss', loss_weight=0.1),
                 loss_coarse=dict(type='SmoothL1Loss', loss_weight=0.1),
                 loss_iter1=dict(type='SmoothL1Loss', loss_weight=1.0),
                 loss_iter2=dict(type='DMLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(E2ECHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.down_sample = down_sample
        self.points_per_poly = points_per_poly
        self.init_stride = init_stride
        self.coarse_stride = coarse_stride
        self.min_ct_score = min_ct_score
        self.evole_ietr_num = evole_ietr_num
        self.evolve_stride = evolve_stride
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2 * points_per_poly)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_init = build_loss(loss_init)
        self.loss_coarse = build_loss(loss_coarse)
        self.loss_iter1 = build_loss(loss_iter1)
        self.loss_iter2 = build_loss(loss_iter2)

        self.cfg = cfg

        self.train_decoder = Decode(num_point=self.cfg.commen.points_per_poly, init_stride=self.cfg.model.init_stride,
                                    coarse_stride=self.cfg.model.coarse_stride, down_sample=self.cfg.commen.down_ratio,
                                    min_ct_score=self.cfg.test.ct_score)
        self.gcn = Evolution(evole_ietr_num=self.cfg.model.evolve_iters, evolve_stride=self.cfg.model.evolve_stride,
                             ro=self.cfg.commen.down_ratio)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats, data_input):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
            cnn_features (List[Tensor])
        """
        outs = multi_apply(self.forward_single, feats)
        center_heatmap_preds, wh_preds, offset_preds, cnn_features = outs
        center_heatmap_preds = torch.stack(center_heatmap_preds, 0)
        wh_preds = torch.stack(wh_preds, 0)
        offset_preds = torch.stack(offset_preds, 0)
        cnn_features = torch.stack(cnn_features, 0)

        output = {}
        output['ct_hm'] = center_heatmap_preds
        output['wh'] = wh_preds

        if 'test' not in data_input['meta']:
            self.train_decoder(data_input, cnn_features, output, is_training=True)
        else:
            with torch.no_grad(): #测试阶段不需要梯度
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(data_input, cnn_features, output, is_training=False, ignore_gloabal_deform=ignore)
        output = self.gcn(output, cnn_features, data_input, test_stage=self.cfg.test.test_stage) #output已经更新了poly_init和poly_coarse 图网络

        return (center_heatmap_preds, wh_preds, offset_preds, cnn_features, output)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes. [h/4,w/4,cls_nums], but it's depending on your tensor shape.
            wh_pred (Tensor): wh predicts, the channels number is 2. [h/4,w/4,2]
            offset_pred (Tensor): offset predicts, the channels number is 2. [h/4,w/4,2]
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        cnn_feature = feat

        return center_heatmap_pred, wh_pred, offset_pred, cnn_feature

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'cnn_features'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             cnn_features,
             output,
             gt_bboxes,
             gt_masks,
             gt_labels,
             data_input,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2*points_per_poly, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """

        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds)

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_preds.shape,
                                                     img_metas[0]['pad_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        keyPointsMask = data_input['keypoints_mask'][data_input['ct_01']]


        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_preds, center_heatmap_target, avg_factor=avg_factor)
        # loss_wh = self.loss_wh(
        #     wh_preds,
        #     wh_target,
        #     wh_offset_target_weight,
        #     avg_factor=avg_factor * 2)
        # loss_offset = self.loss_offset(
        #     offset_preds,
        #     offset_target,
        #     wh_offset_target_weight,
        #     avg_factor=avg_factor * 2)
        loss_init = self.loss_init(
            output['poly_init'],
            output['img_gt_polys'],
            reduction_override='mean')
        loss_coarse = self.loss_coarse(
            output['poly_coarse'],
            output['img_gt_polys'],
            reduction_override='mean')
        loss_iter10 = self.loss_iter1(
            output['py_pred'][0],
            output['img_gt_polys'],
            reduction_override='mean')
        loss_iter11 = self.loss_iter1(
            output['py_pred'][1],
            output['img_gt_polys'],
            reduction_override='mean')
        loss_iter2 = self.loss_iter2(
            output['py_pred'][-2],
            output['py_pred'][-1],
            output['img_gt_polys'],
            keyPointsMask,
            reduction_override='mean')

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            # loss_wh=loss_wh,
            # loss_offset=loss_offset,
            loss_init=loss_init,
            loss_coarse=loss_coarse,
            loss_iter10=loss_iter10,
            loss_iter11=loss_iter11,
            loss_iter2=loss_iter2
            )
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_masks,
                      gt_labels,
                      data_input,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x, data_input) # x是FPN输出的features，self是E2ECHead，()是self.forward
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_masks, data_input, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, data_input, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels
