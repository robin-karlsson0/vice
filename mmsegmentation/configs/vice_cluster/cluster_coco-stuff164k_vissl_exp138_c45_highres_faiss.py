_base_ = [
    '../_base_/models/cluster_vissl.py',
    '../_base_/datasets/cityscapes_nocrop.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_12k.py'
]
model = dict(
    decode_head=dict(
        in_channels=64,
        channels=128,
        num_convs=1,
        kernel_size=128,  # FEATURE DIM
        num_classes=27,
        act_cfg='exp138_c45_highres_faiss'),
    vissl_params=dict(
        config_path=('/home/robin/projects/vissl/experiments/high_res/'
                     'sc_exp138/vice_8node_resnet18_cityscapes_exp138.yaml'),
        checkpoint_path=('/home/robin/projects/vissl/experiments/high_res/'
                         'sc_exp138/model_final_checkpoint_phase47.torch'),
    ))

optimizer = dict(lr=0.01)  # default0.01
