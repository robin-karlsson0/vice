import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vice(ckpt):
    """
    NOTE: Only extracts ResNet 50 backbone (for now?)

    Args:
        ckpt (OrderedDict): State dict w. original notations.
    """

    new_ckpt = OrderedDict()

    for key, value in ckpt.items():

        # Universal modifications
        # Remove prefix
        key = key.replace('_feature_blocks.model.', '')

        if key.find('backbone') == 0:
            # Change notation for the 'stem'
            key = key.replace('layer0.0', 'conv1')
            key = key.replace('layer0.1', 'bn1')

            new_ckpt[key] = value

        elif key.find('ASSP') == 0:
            # (1)
            key = key.replace('ASSP.aspp1.0',
                              'decode_head.aspp_modules.0.conv')
            key = key.replace('ASSP.aspp1.1', 'decode_head.aspp_modules.0.bn')
            # (2)
            key = key.replace(
                'ASSP.aspp2.depthwise_conv',
                'decode_head.aspp_modules.1.depthwise_conv.conv')
            key = key.replace('ASSP.aspp2.depthwise_bn',
                              'decode_head.aspp_modules.1.depthwise_conv.bn')
            key = key.replace(
                'ASSP.aspp2.pointwise_conv',
                'decode_head.aspp_modules.1.pointwise_conv.conv')
            key = key.replace('ASSP.aspp2.pointwise_bn',
                              'decode_head.aspp_modules.1.pointwise_conv.bn')
            # (3)
            key = key.replace(
                'ASSP.aspp3.depthwise_conv',
                'decode_head.aspp_modules.2.depthwise_conv.conv')
            key = key.replace('ASSP.aspp3.depthwise_bn',
                              'decode_head.aspp_modules.2.depthwise_conv.bn')
            key = key.replace(
                'ASSP.aspp3.pointwise_conv',
                'decode_head.aspp_modules.2.pointwise_conv.conv')
            key = key.replace('ASSP.aspp3.pointwise_bn',
                              'decode_head.aspp_modules.2.pointwise_conv.bn')
            # (4)
            key = key.replace(
                'ASSP.aspp4.depthwise_conv',
                'decode_head.aspp_modules.3.depthwise_conv.conv')
            key = key.replace('ASSP.aspp4.depthwise_bn',
                              'decode_head.aspp_modules.3.depthwise_conv.bn')
            key = key.replace(
                'ASSP.aspp4.pointwise_conv',
                'decode_head.aspp_modules.3.pointwise_conv.conv')
            key = key.replace('ASSP.aspp4.pointwise_bn',
                              'decode_head.aspp_modules.3.pointwise_conv.bn')
            # (5)
            key = key.replace('ASSP.avg_pool.1',
                              'decode_head.image_pool.1.conv')
            key = key.replace('ASSP.avg_pool.2', 'decode_head.image_pool.1.bn')

            # Low-level features from ASSP --> Decoder?
            key = key.replace('ASSP.conv1', 'decode_head.bottleneck.conv')
            key = key.replace('ASSP.bn1', 'decode_head.bottleneck.bn')

            new_ckpt[key] = value

        elif key.find('decoder') == 0:
            # ?
            key = key.replace('decoder.conv1',
                              'decode_head.c1_bottleneck.conv')
            key = key.replace('decoder.bn1', 'decode_head.c1_bottleneck.bn')
            # (1)
            key = key.replace(
                'decoder.output.0.depthwise_conv',
                'decode_head.sep_bottleneck.0.depthwise_conv.conv')
            key = key.replace(
                'decoder.output.0.depthwise_bn',
                'decode_head.sep_bottleneck.0.depthwise_conv.bn')
            key = key.replace(
                'decoder.output.0.pointwise_conv',
                'decode_head.sep_bottleneck.0.pointwise_conv.conv')
            key = key.replace(
                'decoder.output.0.pointwise_bn',
                'decode_head.sep_bottleneck.0.pointwise_conv.bn')
            # (2)
            key = key.replace(
                'decoder.output.1.depthwise_conv',
                'decode_head.sep_bottleneck.1.depthwise_conv.conv')
            key = key.replace(
                'decoder.output.1.depthwise_bn',
                'decode_head.sep_bottleneck.1.depthwise_conv.bn')

            key = key.replace(
                'decoder.output.1.pointwise_conv',
                'decode_head.sep_bottleneck.1.pointwise_conv.conv')
            key = key.replace(
                'decoder.output.1.pointwise_bn',
                'decode_head.sep_bottleneck.1.pointwise_conv.bn')
            # (3)
            key = key.replace('decoder.output.2', 'decode_head.conv_seg')

            new_ckpt[key] = value

        else:
            print(f'Unknown state key: {key} | {value.size()}')

    return new_ckpt


def convert_model_zoo(ckpt):
    """Convert baseline backbone models provided by VISSL.

    Ref: https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md

    Args:
        ckpt (OrderedDict): State dict w. original notations.
    """

    new_ckpt = OrderedDict()

    for key, value in ckpt.items():

        # Universal modifications
        # Substitute prefix
        key = key.replace('_feature_blocks', 'backbone')

        new_ckpt[key] = value

    return new_ckpt


def convert_picie(ckpt):
    """Convert PICIE backbone model.

    Ref: https://github.com/janghyuncho/PiCIE

    Args:
        ckpt (OrderedDict): State dict w. original notations.
    """
    new_ckpt = OrderedDict()

    for key, value in ckpt.items():

        # Only read 'backbone' parameters
        if 'backbone' not in key:
            continue

        # Universal modifications
        # Substitute prefix
        key = key.replace('module.', '')

        print(f'{key}\t{value.size()}')
        new_ckpt[key] = value

    return new_ckpt


def convert_pytorch(ckpt):
    """Convert Pytorch pretrained backbone model."""
    new_ckpt = OrderedDict()

    for key, value in ckpt.items():

        # Universal modifications
        # Add prefix
        key = 'backbone.' + key

        print(f'{key}\t{value.size()}')
        new_ckpt[key] = value

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description=('Convert keys in VISSL pretrained Dense SwAV models to'
                     'MMSegmentation style.'))
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    parser.add_argument('type', help='vice, model_zoo, picie, pytorch')
    args = parser.parse_args()

    type = args.type
    if type == 'vice':
        checkpoint = CheckpointLoader.load_checkpoint(
            args.src, map_location='cpu')
        state_dict = checkpoint['classy_state_dict']['base_model']['model'][
            'trunk']
        weight = convert_vice(state_dict)
    elif type == 'model_zoo':
        checkpoint = CheckpointLoader.load_checkpoint(
            args.src, map_location='cpu')
        state_dict = checkpoint['classy_state_dict']['base_model']['model'][
            'trunk']
        weight = convert_model_zoo(state_dict)
    elif type == 'picie':
        # Extract ResNet 50 + decoder parameters
        state_dict = torch.load(args.src)
        state_dict = state_dict['state_dict']
        weight = convert_picie(state_dict)
    elif type == 'pytorch':
        state_dict = torch.load(args.src)
        weight = convert_pytorch(state_dict)
    else:
        raise Exception(f'Undefinied type: {type}')
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
