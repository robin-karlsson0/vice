# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderVISSLFCN',
    decode_head=dict(
        type='ClusterHead',
        in_channels=64,
        in_index=0,  # For extracting x <-- [x]
        channels=128,
        num_convs='knut',
        kernel_size=3,
        concat_input=False,
        dropout_ratio=0.0,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        act_cfg=dict(type='ReLU'),  # None
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    vissl_params=dict(
        vissl_dir='/home/robin/projects/vissl',
        config_path=None,
        checkpoint_path=None,
        output_type='trunk',
        default_config_path='vissl/config/defaults.yaml'))
