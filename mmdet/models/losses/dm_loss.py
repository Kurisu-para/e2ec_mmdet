import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weight_reduce_loss


@mmcv.jit(derivate=True, coderize=True)
def dm_loss(ini_pred_poly, pred_poly, gt_poly, keyPointsMask, crit, weight, reduction, avg_factor):

    def interpolation(poly, time=10):
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, 10)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), -1, 2)
        return poly_interpolation

    def compute_distance(pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance

    def lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly):
        gt_poly_interpolation = interpolation(gt_poly)
        distance_pred_gtInterpolation = compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_predto_nearestgt

    def lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly):
        distance_pred_gt = compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
    lossPred2NearestGt = lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
    lossGt2NearestPred = lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

    loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
    loss = loss_set2set / 2.
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class DMLoss(nn.Module):
    def __init__(self, type='smooth_l1', reduction='mean', loss_weight=1.0):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                ini_pred_poly, 
                pred_poly,
                gt_poly, 
                keyPointsMask, 
                weight=None, 
                avg_factor=None, 
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        crit = self.crit
        loss_bbox = self.loss_weight * dm_loss(ini_pred_poly, 
                                               pred_poly,
                                               gt_poly,
                                               keyPointsMask,
                                               crit,
                                               weight,
                                               reduction=reduction,
                                               avg_factor=avg_factor)
        return loss_bbox