_base_ = [
    '../_base_/models/cluster_vissl.py',
    '../_base_/datasets/coco-stuff164k_coarse.py',  # _nocrop_lowres.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_12k.py'
]
model = dict(
    decode_head=dict(
        in_channels=64,
        channels=128,
        num_convs=1,
        kernel_size=128,  # FEATURE DIM
        num_classes=27,
        act_cfg='exp175_c256_lowres_faiss'),
    vissl_params=dict(
        config_path=('/home/robin/projects/vissl/experiments/low_res/'
                     'sc_exp175/vice_8node_fpn_resnet50_coco_exp175.yaml'),
        checkpoint_path=('/home/robin/projects/vissl/experiments/low_res/'
                         'sc_exp175/model_final_checkpoint_phase11.torch'),
    ))

optimizer = dict(lr=0.01)  # default0.01
