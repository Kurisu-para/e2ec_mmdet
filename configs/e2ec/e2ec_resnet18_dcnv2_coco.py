_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='E2EC',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='E2ECHead',
        num_classes=80,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_init=dict(type='SmoothL1Loss', loss_weight=0.1),
        loss_coarse=dict(type='SmoothL1Loss', loss_weight=0.1),
        loss_iter1=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_iter2=dict(type='DMLoss', loss_weight=1.0))
)

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True, poly2mask=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_masks', 'gt_labels', 'data_input'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True, poly2mask=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_input'])
]

dataset_type = 'CocoDataset'
data_root = '/home/sjtu/scratch/tongzhao/e2ec/data/coco/'

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[18, 24])  # the real step is [18*5, 24*5]
# workflow = [('val', 1), ('train', 1)]
workflow = [('train', 1), ('val', 1)]
# workflow = [('train', 1)]
runner = dict(max_epochs=28)  # the real epoch is 28*5=140
# # running setting
# total_epochs = 140