_base_ = [
    '_base_/models/fcn_vissl.py', '_base_/datasets/coco-stuff164k_coarse.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_10k.py'
]
model = dict(
    decode_head=dict(
        in_channels=64,
        channels=128,
        num_convs=1,
        kernel_size=1,
        num_classes=27,
        act_cfg=None),
    vissl_params=dict(
        config_path='sc_exp09/dense_swav_8node_resnet_test.yaml',
        checkpoint_path='sc_exp09/model_iteration5000.torch',
    ))

optimizer = dict(lr=0.01)  # default0.01
