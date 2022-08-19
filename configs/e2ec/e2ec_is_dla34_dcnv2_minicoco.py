_base_ = [
    '../_base_/datasets/coco_instance.py',
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

dataset_type = 'CocoDataset'
data_root = '/home/sjtu/scratch/tongzhao/minicoco/'

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
        num_classes=80,
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
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'mini_instances_train2017.json',
            img_prefix=data_root + 'mini_train2017/',
            pipeline=train_pipeline)),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'mini_instances_val2017.json',
            img_prefix=data_root + 'mini_val2017/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'mini_instances_val2017.json',
            img_prefix=data_root + 'mini_val2017/',
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])

optimizer = dict(_delete_=True, type='Adam', lr=1e-4, weight_decay=5e-4)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[16, 24],
    gamma=0.5)

workflow = [('train', 1)]
runner = dict(max_epochs=28)
# fp16 = dict(loss_scale=512.)
find_unused_parameters = True

custom_hooks = [
    dict(
        type='E2ECModeSwitchHook',
        start_epochs=2, # the real epoch is 2*5
        priority=48)
]