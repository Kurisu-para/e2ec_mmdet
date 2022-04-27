import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import InstanceData, mask_matrix_nms, multi_apply
from mmdet.core.utils import center_of_mass, generate_coordinate
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target

from .base_mask_head import BaseMaskHead
from .centernet_head import CenterNetHead
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat)

@HEADS.register_module()
class E2ECHead(CenterNetHead):
    def __init__(self,in_channel,feat_channel,num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)


