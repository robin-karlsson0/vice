import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain

from vissl.config import AttrDict
from vissl.models.model_helpers import (
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from typing import List


@register_model_trunk("deeplabv3plus_legacy")
class DeepLabV3PlusTrunkLegacy(nn.Module):
    """
    documentation on what the trunk does and links to technical reports
    using this trunk (if applicable)
    """
    def __init__(self, model_config: AttrDict, model_name: str):
        super(DeepLabV3PlusTrunkLegacy, self).__init__()
        self.model_config = model_config

        # get the params trunk takes from the config
        self.trunk_config = self.model_config.TRUNK.DEEPLABV3PLUS
        self.backbone = self.trunk_config.BACKBONE
        self.output_stride = self.trunk_config.OUTPUT_STRIDE
        self.pretrained = self.trunk_config.PRETRAINED
        self.in_feat_dim = self.trunk_config.IN_FEAT_DIM
        self.out_feat_dim = self.trunk_config.OUT_FEAT_DIM
        self.decoder_ch = self.trunk_config.DECODER_CH
        self.use_checkpointing = (self.model_config.ACTIVATION_CHECKPOINTING.
                                  USE_ACTIVATION_CHECKPOINTING)
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.
            NUM_ACTIVATION_CHECKPOINTING_SPLITS)

        # implement the model trunk and construct all the layers that the trunk
        # uses
        model = DeepLabV3Plus(
            self.out_feat_dim,
            self.in_feat_dim,
            self.backbone,
            self.pretrained,
            self.output_stride,
            output_backbone=self.trunk_config.OUTPUT_BACKBONE,
            decoder_ch=self.decoder_ch)
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


class ResNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 output_stride=16,
                 backbone='resnet50',
                 pretrained=True):
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

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


########################################
#  The Atrous Spatial Pyramid Pooling
########################################


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channles,
                  kernel_size,
                  padding=padding,
                  dilation=dilation,
                  bias=False), nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride, decoder_ch=256):
        super(ASSP, self).__init__()

        assert output_stride in [
            8, 16
        ], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels,
                                 decoder_ch,
                                 1,
                                 dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels,
                                 decoder_ch,
                                 3,
                                 dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels,
                                 decoder_ch,
                                 3,
                                 dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels,
                                 decoder_ch,
                                 3,
                                 dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, decoder_ch, 1, bias=False),
            nn.BatchNorm2d(decoder_ch), nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(decoder_ch * 5, decoder_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(decoder_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x),
                           size=(x.size(2), x.size(3)),
                           mode='bilinear',
                           align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


#############
#  Decoder
#############


class Decoder(nn.Module):
    def __init__(self, low_level_channels, out_channels, decoder_ch=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + decoder_ch,
                      decoder_ch,
                      3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_ch,
                      decoder_ch,
                      3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_ch, out_channels, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


#####################
#  Deeplab V3 plus
#####################


class DeepLabV3Plus(BaseModel):
    def __init__(self,
                 out_channels,
                 in_channels=3,
                 backbone='resnet50',
                 pretrained=True,
                 output_stride=16,
                 freeze_bn=False,
                 freeze_backbone=False,
                 output_backbone=False,
                 decoder_ch=256,
                 **_):

        super(DeepLabV3Plus, self).__init__()
        self.backbone = ResNet(in_channels=in_channels,
                               output_stride=output_stride,
                               backbone=backbone,
                               pretrained=pretrained)
        low_level_channels = 256

        self.ASSP = ASSP(in_channels=2048,
                         output_stride=output_stride,
                         decoder_ch=decoder_ch)
        self.decoder = Decoder(low_level_channels, out_channels, decoder_ch)

        self.output_backbone = output_backbone

        if freeze_bn:
            self.freeze_bn()
        # if freeze_backbone:
        #     set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        if self.output_backbone:
            return x
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()