# Copyright (c) OpenMMLab. All rights reserved.
# import copy
import pickle

import faiss
# import matplotlib.pyplot as plt
import numpy as np
import torch

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ClusterHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        super(ClusterHead, self).__init__(**kwargs)

        # Setup k-means cluster estimator
        exp = self.act_cfg
        with open(f'/home/robin/projects/vissl/kmeans_centroids_{exp}.pkl',
                  'rb') as f:
            kmeans_centroids = pickle.load(f)
        self.index = faiss.IndexFlatL2(kernel_size)
        self.index.add(kmeans_centroids)

        # Load cluster to class mapping
        with open(
                f'/home/robin/projects/vissl/cluster2class_mapping_{exp}.pkl',
                'rb') as f:
            self.cluster2class_mapping = pickle.load(f)

    def forward(self, inputs):
        """Forward function."""
        inputs = inputs[0][0]  # --> dim (D, H, W)

        # Transform 'tensor' --> 'row vectors' dim (H*W, D)
        D, H, W = inputs.shape
        inputs = torch.reshape(inputs, (D, -1)).T
        inputs = inputs.cpu().numpy()
        inputs = np.ascontiguousarray(inputs)

        # Replace 'embedding features' --> 'cluster idx'
        _, cluster_idxs = self.index.search(inputs, 1)

        # Transform 'row vectors' --> '2d map' dim (H, W)
        cluster_idx_map = np.reshape(cluster_idxs, (H, W, -1))
        cluster_idx_map = cluster_idx_map[:, :, 0]  # (h, w, 1) --> (h, w)

        N_CLASSES = 27
        output = np.zeros((N_CLASSES, H, W))

        unique_idxs = np.unique(cluster_idx_map)
        for idx in unique_idxs:
            class_idx = self.cluster2class_mapping[idx]
            mask = (cluster_idx_map == idx)
            output[class_idx] = np.logical_or(output[class_idx], mask)

        # Visualize clusters
        # cluster_idx_map_ = copy.deepcopy(cluster_idx_map)
        # h, w = cluster_idx_map_.shape
        # cluster_idx_map_rgb = np.zeros((h, w, 3))
        # for idx in np.unique(cluster_idx_map_):
        #     rgb = np.random.randint(0, 255, 3)
        #     cluster_idx_map_rgb[cluster_idx_map_ == idx] = rgb
        # plt.imshow(cluster_idx_map_rgb.astype(int))
        # plt.show()

        # Visualize segmentation
        # output_ = copy.deepcopy(output)
        # for idx in range(27):
        #     output_[idx] *= idx
        # output_ = np.sum(output_, axis=0)

        output = torch.tensor(output).unsqueeze(0)

        return output
