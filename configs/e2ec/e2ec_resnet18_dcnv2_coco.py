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
        loss_iter1=dict(type='SmoothL1Loss', loss_weight=1.0/3.0),
        loss_iter2=dict(type='DMLoss', loss_weight=1.0/3.0))
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True, with_mask=True, poly2mask=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_masks', 'gt_labels', 'data_input'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')
         )
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Augment', mode='test'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape'), 
         keys=['img', 'meta'])
]

dataset_type = 'CocoDataset'
data_root = '/home/sjtu/dataset/coco/'

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline),
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

optimizer = dict(_delete_=True, type='Adam', lr=1e-4, weight_decay=5e-4)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 1000,
#     step=[18, 24])  # the real step is [18*5, 24*5]

# workflow = [('val', 1), ('train', 1)]
workflow = [('train', 1), ('val', 1)]
# workflow = [('train', 1)]
runner = dict(max_epochs=140)