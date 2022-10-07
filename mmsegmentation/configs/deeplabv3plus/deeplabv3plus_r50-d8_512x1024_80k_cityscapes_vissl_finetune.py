_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_vissl_finetune.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# Layer group specific LR multipliers
optimizer = dict(
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# Modifications for VISSL baseline backbones (double channel amount)
# MMSeg: 64
# SimCLR: 128
# SwAV: 256
# model = dict(
#     backbone=dict(
#         stem_channels=128,  #  64, 128, 256
#         base_channels=128,  # 64, 128, 256
#     ),
#     decode_head=dict(
#         in_channels=4096,  # 2048, 4096, 8192
#         c1_in_channels=512,  # 256, 512, 1024
#     ),
#     auxiliary_head=dict(
#         in_channels=2048,  # 1024, 2048, 4096
#         channels=512,  # 256, 512, 1024
#     ),
# )
