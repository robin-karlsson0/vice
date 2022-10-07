import logging
from itertools import chain
from typing import List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from vissl.config import AttrDict
from vissl.models.model_helpers import (
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk


@register_model_trunk("fpn")
class FeaturePyramidNetworkTrunk(nn.Module):
    """
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(FeaturePyramidNetworkTrunk, self).__init__()
        self.model_config = model_config

        # get the params trunk takes from config
        self.trunk_config = self.model_config.TRUNK.FPN
        self.backbone = self.trunk_config.BACKBONE
        self.output_stride = self.trunk_config.OUTPUT_STRIDE
        self.pretrained = self.trunk_config.PRETRAINED
        self.in_feat_dim = self.trunk_config.IN_FEAT_DIM
        self.out_feat_dim = self.trunk_config.OUT_FEAT_DIM
        self.frozen_stages = self.trunk_config.FROZEN_STAGES
        self.use_checkpointing = (self.model_config.ACTIVATION_CHECKPOINTING.
                                  USE_ACTIVATION_CHECKPOINTING)
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.
            NUM_ACTIVATION_CHECKPOINTING_SPLITS)

        # implement the model trunk and construct all the layers that the trunk
        # uses
        model = FPN(self.out_feat_dim,
                    self.in_feat_dim,
                    self.backbone,
                    self.pretrained,
                    self.output_stride,
                    frozen_stages=self.frozen_stages,
                    output_backbone=self.trunk_config.OUTPUT_BACKBONE)
        # model_layer2 = ??
        # ...
        # ...

        # give a name to the layers of your trunk so that these features
        # can be used for other purposes: like feature extraction etc.
        # the name is fully upto user descretion. User may chose to
        # only name one layer which is the last layer of the model.
        self._feature_blocks = nn.ModuleDict([
            ("model", model),
            # ("my_layer1_name", model_layer2),
            # ...
        ])

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "out": "model",
        }

    def forward(self,
                x: torch.Tensor,
                out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        # implement the forward pass of the model. See the forward pass of
        # resnext.py for reference.
        # The output would be a list. The list can have one tensor (the trunk
        # output) or mutliple tensors (corresponding to several features of the
        # trunk)

        # TODO: if isinstance(x, MultiDimensionalTensor):

        model_input = transform_model_input_data_type(
            x, self.model_config.INPUT_TYPE)
        out = get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            feature_mapping=self.feat_eval_mapping,
            use_checkpointing=self.use_checkpointing,
            checkpointing_splits=self.num_checkpointing_splits,
        )
        return out


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(
            BaseModel,
            self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


#############
#  Encoder
#############


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class ResNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 output_stride=16,
                 backbone='resnet50',
                 pretrained=True,
                 frozen_stages=-1):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34'
                                     or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                                                                           s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                                                                           s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34'
                                 or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def forward(self, x):
        x = self.layer0(x)
        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        # Intermediate features ordered from shallow --> deep
        out = OrderedDict()
        out['feat0'] = x0
        out['feat1'] = x1
        out['feat2'] = x2
        out['feat3'] = x3
        return out

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        for i in range(0, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


#############################
#  Feature Pyramid Network
#############################


class FPN(BaseModel):
    """
    # Ref: https://pytorch.org/vision/master/_modules/torchvision/ops/feature_pyramid_network.html
    #      https://github.com/janghyuncho/PiCIE/blob/master/modules/fpn.py
    """

    def __init__(
        self,
        out_channels,
        in_channels=3,
        backbone='resnet50',
        pretrained=True,
        output_stride=16,
        freeze_bn=False,
        output_backbone=False,
        frozen_stages=-1,
    ):

        super(FPN, self).__init__()
        self.backbone = ResNet(in_channels=in_channels,
                               output_stride=output_stride,
                               backbone=backbone,
                               pretrained=pretrained,
                               frozen_stages=frozen_stages)

        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            fpn_in_channels = [256, 512, 1024, 2048]
            mfactor = 4
        elif backbone in ['resnet18', 'resnet34']:
            fpn_in_channels = [64, 128, 256, 512]
            mfactor = 1
        else:
            raise ValueError(f"Undefined backbone: {backbone}")

        # FPN
        # self.fpn = FeaturePyramidNetwork(fpn_in_channels,
        #                                  out_channels=out_channels)

        # FPN decoder
        self.decoder_layer4 = nn.Conv2d(512 * mfactor // 8,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.decoder_layer3 = nn.Conv2d(512 * mfactor // 4,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.decoder_layer2 = nn.Conv2d(512 * mfactor // 2,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.decoder_layer1 = nn.Conv2d(512 * mfactor,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.output_backbone = output_backbone

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        # Resnet
        x = self.backbone(x)
        if self.output_backbone:
            raise NotImplementedError

        # FPN
        # x = self.fpn(x)

        # FPN decoder
        out1 = self.decoder_layer1(x['feat3'])
        out2 = self.upsample_add(out1, self.decoder_layer2(x['feat2']))
        out3 = self.upsample_add(out2, self.decoder_layer3(x['feat1']))
        out4 = self.upsample_add(out3, self.decoder_layer4(x['feat0']))

        out = F.interpolate(out4, [H, W], mode='bilinear', align_corners=True)
        return out

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=False) + y

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return self.fpn.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
