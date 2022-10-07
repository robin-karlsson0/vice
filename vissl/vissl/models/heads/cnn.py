from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head

@register_model_head("cnn")
class CNN(nn.Module):
    """
    Module for projecting elements in a dense output feature map using a
    cascade of 1x1 CNNs.

    Follows the MLP implementation.
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
        skip_last_layer_relu_bn: bool = True):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            dims (int): Number of filters in each 1x1 convolution.
            use_bn (bool): whether to attach BatchNorm after Linear layer.
            use_relu (bool): whether to attach ReLU after (Linear (-> BN optional)).
            use_dropout (bool): whether to attach Dropout after
                                (Linear (-> BN -> relu optional)).
            use_bias (bool): whether the Linear layer should have bias or not.
            skip_last_layer_relu_bn (bool): If the MLP has many layers, we check
                if after the last MLP layer, we should add BN / ReLU or not. By
                default, skip it. If user specifies to not skip, then BN will be
                added if use_bn=True, ReLU will be added if use_relu=True.
        """
        super().__init__()
        # implement what the init of head should do. Example, it can construct the layers in the head
        # like FC etc., initialize the parameters or anything else
        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):
            layers.append(nn.Conv2d(last_dim, dim, 1, bias=use_bias))
            if i == len(dims) - 2 and skip_last_layer_relu_bn:
                break
            if use_bn:
                layers.append(
                    nn.BatchNorm2d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout2d())
            last_dim = dim
        self.clf = nn.Sequential(*layers)

    # the input to the model should be a torch Tensor or list of torch tensors.
    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.tensor): Output tensor w. dim (N, feat, h, w)
        
        Returns:
            Tensor w. dim (N, proj, h, w)
        """
        out = self.clf(batch)
        return out
