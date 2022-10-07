python tools/pytorch2onnx.py
    pretrained/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth
    --checkpoint
    --input-img pretrained/20180807145028_camera_frontcenter_000000091.png
    --verify
    --dynamic-export


python tools/pytorch2onnx.py configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.py --checkpoint pretrained/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth --input-img pretrained/20180807145028_camera_frontcenter_000000091.png --verify --dynamic-export
