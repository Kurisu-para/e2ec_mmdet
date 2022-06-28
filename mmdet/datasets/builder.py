# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
import warnings
from functools import partial

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
# from mmcv.parallel import collate
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .samplers import (DistributedGroupSampler, DistributedSampler,
                       GroupSampler, InfiniteBatchSampler,
                       InfiniteGroupBatchSampler)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def collate_batch(batch):
    data_input = {}
    inp = {'inp': default_collate([b['inp'] for b in batch])}
    meta = default_collate([b['meta'] for b in batch])
    data_input.update(inp)
    data_input.update({'meta': meta})

    if 'test' in meta:
        return data_input

    #collate detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])
    max_len = torch.max(meta['ct_num'])
    batch_size = len(batch)
    wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
    ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1
        ct_img_idx[i, :meta['ct_num'][i]] = i

    if max_len != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
    detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01, 'ct_img_idx': ct_img_idx}
    data_input.update(detection)

    #collate sementation
    num_points_per_poly = 128
    img_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    can_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    keyPointsMask = torch.zeros([batch_size, max_len, num_points_per_poly], dtype=torch.float)

    if max_len != 0:
        img_gt_polys[ct_01] = torch.Tensor(np.array(sum([b['img_gt_polys'] for b in batch], [])))
        can_gt_polys[ct_01] = torch.Tensor(np.array(sum([b['can_gt_polys'] for b in batch], [])))
        keyPointsMask[ct_01] = torch.Tensor(np.array(sum([b['keypoints_mask'] for b in batch], [])))
    data_input.update({'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                       'keypoints_mask': keyPointsMask})

    return data_input

def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.
    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.
    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def main_collate(batch, samples_per_gpu=1):
    mybatch = [b['data_input'] for b in batch]
    mybatch = collate_batch(mybatch)
    # mybatch = DataContainer(mybatch)
    purebatch = []
    for b in batch:
        b.pop('data_input')
        purebatch.append(b)

    out = collate(purebatch, samples_per_gpu=samples_per_gpu)
    out['data_input'] = mybatch

    return out

def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                                   MultiImageMixDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset,
                batch_size,
                world_size,
                rank,
                seed=seed,
                shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset,
                                   samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(main_collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
