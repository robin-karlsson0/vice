from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head


@register_model_head("dense_swav_head")
class DenseSwAVPrototypesHead(nn.Module):
    """
    Dense SwAV head.

    NOTE: Substitute the regular head with the identity function if
          USE_IDENTITY = True !
    """
    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool,
        num_clusters: int,
        use_bias: bool = True,
        return_embeddings: bool = True,
        comp_score_output: bool = True,
        skip_last_bn: bool = True,
        normalize_feats: bool = True,
        activation_name: str = "ReLU",
        use_weight_norm_prototypes: bool = False,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            dims (int): dimensions of the linear layer. Must have length at least 2.
                        Example: [2048, 2048, 128] attaches linear layer
                                 Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128)
            use_bn (bool): whether to attach BatchNorm after Linear layer
            num_clusters (List(int)): number of prototypes or clusters. Typically 3000.
                                      Example dims=[3000] will attach 1 prototype head.
                                              dims=[3000, 3000] will attach 2 prototype heads
            use_bias (bool): whether the Linear layer should have bias or not
            return_embeddings (bool): whether return the projected embeddings or not
            comp_score_output (bool): Computes scores for all projection and prototype vectors if True.
            skip_last_bn (bool): whether to attach BN + Relu at the end of projection head.
                        Example:
                            [2048, 2048, 128] with skip_last_bn=True attaches linear layer
                            Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128)

                            [2048, 2048, 128] with skip_last_bn=False attaches linear layer
                            Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128) -> BN -> ReLU

                        This could be particularly useful when performing full finetuning on
                        hidden layers.
            use_weight_norm_prototypes (bool): whether to use weight norm module for the prototypes layers.
        """
        super().__init__()

        self.normalize_feats = normalize_feats

        # Create projection head or skip it by substituting with identity func
        if model_config.HEAD.USE_IDENTITY:
            self.projection_head = nn.Identity()
        else:
            # Build the 1x1 2D convolution projection head
            layers = []
            last_dim = dims[0]
            for i, dim in enumerate(dims[1:]):
                layers.append(nn.Conv2d(last_dim, dim, 1, bias=use_bias))
                if (i == len(dims) - 2) and skip_last_bn:
                    break
                if use_bn:
                    layers.append(
                        nn.BatchNorm2d(
                            dim,
                            eps=model_config.HEAD.BATCHNORM_EPS,
                            momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                        ))
                if activation_name == "ReLU":
                    layers.append(nn.ReLU(inplace=True))
                if activation_name == "GELU":
                    layers.append(nn.GELU())
                last_dim = dim
            self.projection_head = nn.Sequential(*layers)

        # prototypes (i.e. centroids) layers
        if len(num_clusters) > 0:
            self.nmb_heads = len(num_clusters)
            for i, k in enumerate(num_clusters):
                proto = nn.Linear(dims[-1], k, bias=False)
                if use_weight_norm_prototypes:
                    proto = nn.utils.weight_norm(proto)
                    proto.weight_g.data.fill_(1)
                self.add_module("prototypes" + str(i), proto)
        else:
            self.nmb_heads = 0

        # Edge detector layers
        # layers = []
        # layers.append(nn.Conv2d(dims[-1], 128, kernel_size=3, padding=1))
        # layers.append(nn.BatchNorm2d(128))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(128, 1, kernel_size=1, padding=0))
        # layers.append(nn.Sigmoid())
        # self.edge_head = nn.Sequential(*layers)

        self.output_projections = model_config.HEAD.OUTPUT_PROJECTIONS
        self.return_embeddings = return_embeddings
        self.comp_score_output = comp_score_output

    def forward(self, feat: torch.Tensor):
        """
        Args:
            batch (4D torch.tensor):
        Returns:

        """
        proj = self.projection_head(feat)  # (1, H, Hi, Wi) --> (1, D, Hi, Wi)

        if self.normalize_feats:
            proj = nn.functional.normalize(proj, dim=1, p=2)

        if self.output_projections:
            return proj

        D = proj.shape[1]
        proj = proj[0]  # ()

        # in: (1,D,H,W)
        # out: (1,1,H,W)
        # proj_edges = self.edge_head(proj.unsqueeze(0))

        # Consistent reshape
        # (N, D, H, W) --> (N*H*W, 3000)
        # score[0] --> (N,H,W) = (0,0,0)
        # score[1] --> (N,H,W) = (0,0,1)
        proj = torch.reshape(proj, (D, -1)).T

        out = []
        if self.return_embeddings:
            out.append(proj)
        if self.nmb_heads > 0 and self.comp_score_output:
            for i in range(self.nmb_heads):  # 1
                # Scores: C^T Z (16, 3000)
                scores = getattr(self, "prototypes" + str(i))(proj)
                # scores = torch.tensor(0.)
                out.append(scores)

        # TODO: Change into dictionary to avoid messing up order in list?
        # TODO: Check if this "edge head" works
        # TODO: Add loss based on "edge head" output

        # out.append(proj_edges)

        return out
