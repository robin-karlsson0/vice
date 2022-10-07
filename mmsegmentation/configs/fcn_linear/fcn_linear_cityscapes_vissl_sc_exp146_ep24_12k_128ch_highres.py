_base_ = [
    '../_base_/models/fcn_vissl.py', '../_base_/datasets/cityscapes_nocrop.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_12k.py'
]
model = dict(
    decode_head=dict(
        in_channels=128,
        channels=128,  # 256,
        num_convs=1,
        kernel_size=1,
        num_classes=27,
        act_cfg=None),
    vissl_params=dict(
        vissl_dir='/home/z44406a/projects/vissl',
        config_path='sc_exp146/vice_8node_fpn_resnet18_cityscapes_exp146.yaml',
        checkpoint_path='sc_exp146/model_final_checkpoint_phase23.torch',
    ))

optimizer = dict(lr=0.01)  # default0.01
