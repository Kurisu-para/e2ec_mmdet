_base_ = [
    '../_base_/datasets/cityscapes_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='MMDetWandbHook',
        #     init_kwargs={'entity':'iszhaotong', 'project':'e2ec', 'name':'e2ec_is_resnet18_dcnv2_minicoco'},
        #     interval=10,
        #     log_checkpoint=True,
        #     log_checkpoint_metadata=True,
        #     num_eval_images=100,
        #     bbox_score_thr=0.3)
        ])

dataset_type = 'CityscapesDataset'
data_root = '/home/sjtu/scratch/tongzhao/e2ec_mmdetection/dataset/cityscapes/'
ann_root = '/home/sjtu/scratch/tongzhao/data/cityscapes/'

model = dict(
    type='E2EC',
    backbone=dict(
        type='DLASeg',
        base_name='dla34',
        pretrained=True,
        down_ratio=4,
        last_level=5,
        out_channel=0,
        use_dcn=True),
    mask_head=dict(
        type='E2ECHead',
        num_classes=8,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_init=dict(type='SmoothL1Loss', loss_weight=0.1),
        loss_coarse=dict(type='SmoothL1Loss', loss_weight=0.1),
        loss_iter10=dict(type='SmoothL1Loss', loss_weight=1/3),
        loss_iter11=dict(type='SmoothL1Loss', loss_weight=1/3),
        loss_iter2=dict(type='DMLoss', loss_weight=1/3),
        dataset_type=dataset_type),
    dataset_cfg=dict(type=dataset_type)
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=True),
    dict(type='Contour', dataset_type=dataset_type),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'data_input'], meta_keys=())
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Augment', mode='test', dataset_type=dataset_type),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape'),
         keys=['img', 'meta'])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=5,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_root + 'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
    val=dict(
            type=dataset_type,
            ann_file=ann_root + 'annotations/instancesonly_filtered_gtFine_val.json',
            img_prefix=data_root + 'leftImg8bit/val/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=ann_root + 'annotations/instancesonly_filtered_gtFine_test.json',
            img_prefix=data_root + 'leftImg8bit/test/',
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# optimizer = dict(_delete_=True, type='Adam', lr=1e-4, weight_decay=5e-4)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     step=[16, 24],
#     gamma=0.5)
# workflow = [('train', 1)]
# runner = dict(max_epochs=28)
# # fp16 = dict(loss_scale=512.)

# CenterNet Strategy
# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[18, 24, 32])  # the real step is [18*5, 24*5, 32*5]
runner = dict(max_epochs=40)  # the real epoch is 28*5=140

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (16 samples per GPU)
# auto_scale_lr = dict(base_batch_size=128)
find_unused_parameters = True