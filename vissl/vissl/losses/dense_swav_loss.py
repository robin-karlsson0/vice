import logging
import math
import os
import pprint
from typing import List

import torch
from classy_vision.generic.distributed_util import (
    all_reduce_max,
    all_reduce_sum,
    get_cuda_device_index,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from fvcore.common.file_io import PathManager
from torch import nn
from vissl.config import AttrDict


LOSS_COHERENCE_W = 1.
LOSS_SWAV_W = 1.


@register_loss("dense_swav_loss")
class DenseSwAVLoss(ClassyLoss):
    """
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()

        self.loss_config = loss_config
        self.queue_start_iter = self.loss_config.queue.start_iter
        self.dense_swav_criterion = DenseSwAVCriterion(
            self.loss_config.temperature,
            self.loss_config.crops_for_assign,
            self.loss_config.num_crops,
            self.loss_config.num_iters,
            self.loss_config.epsilon,
            self.loss_config.use_double_precision,
            self.loss_config.num_prototypes,
            self.loss_config.queue.local_queue_length,
            self.loss_config.queue.vecs_per_sample,
            self.loss_config.embedding_dim,
            self.loss_config.temp_hard_assignment_iters,
            self.loss_config.output_dir,
            self.loss_config.comp_coherence_loss,
        )

        # self.edge_criterion = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SwAVLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SwAVLoss instance.
        """
        return cls(loss_config)

    def forward(self, output: dict, label_superpixels: torch.Tensor):
        """
        Args:
            output (dict): Dense SwAV head output.

                projs (list): Tree with projection vectors for all views and
                              images.
                              projs[view_idx][img_idx] --> tensor (#vec, D)

                scores (list): Tree with prototype score vectors for all
                               projection vectors for all views and images.
                               scores[view_idx][img_idx] --> tensor (#vec, K)

                edges (list): Tree with projection edge maps for all views and
                              images.
                              edges[view_idx][img_idx]
                              --> tensor (1, 1, h_crop, w_crop)

            label_superpixels (torch.Tensor): 
                                        dim (view_M, img_N, h_crop, w_crop).
        Returns:
        """
        prototypes = output['projs']
        prototypes_scores = output['scores']
        score_regions = output['regions']

        self.dense_swav_criterion.use_queue = (
            self.dense_swav_criterion.local_queue_length > 0 and
            self.dense_swav_criterion.num_iteration >= self.queue_start_iter)

        # NOTE: Compared to SwAV implementation - assumes single head scores
        loss, rep_projs = self.dense_swav_criterion(prototypes,
                                                    prototypes_scores,
                                                    score_regions)

        # Edge loss
        # edge_loss = 0.
        # view_M, img_N, _, _ = label_edges.shape
        # for img_idx in range(img_N):
        #     # Only compute edge loss for first view of every image
        #     # Output
        #     view_proj_edge = view_proj_edges[0][img_idx]
        #     view_proj_edge = view_proj_edge[0, 0]  # Remove dummy dims
        #     # Target
        #     label_edge = label_edges[0, img_idx].to(torch.float16)
        #     edge_loss += self.edge_criterion(view_proj_edge, label_edge)

        # print(f"{loss_dense_swav:.3f}, {edge_loss:.3f}")
        # loss = loss_dense_swav  # + edge_loss

        self.dense_swav_criterion.num_iteration += 1
        if self.dense_swav_criterion.use_queue:
            # Collect all 'view 0' projection vectors into single tensor
            img_N = len(rep_projs[0])
            view_0_rep_projs = []
            with torch.no_grad():
                for img_idx in range(img_N):
                    rep_proj = rep_projs[0][img_idx]
                    # Store a random subset of proj vecs to increase variation
                    proj_N = rep_proj.shape[0]
                    sampling_N = self.dense_swav_criterion.vecs_per_sample
                    if sampling_N > 0 and proj_N > sampling_N:
                        # Generate vector of row indices to sample
                        uniform_prob = torch.ones(proj_N, dtype=torch.float)
                        idxs = torch.multinomial(uniform_prob, sampling_N)
                        rep_proj = rep_proj[idxs]
                    view_0_rep_projs.append(rep_proj)
                view_0_rep_projs = torch.cat(view_0_rep_projs)
            # Add current batchg 'view 0' prototypes score vector to queue
            self.dense_swav_criterion.update_emb_queue(
                view_0_rep_projs.detach())
        return loss

    def __repr__(self):
        repr_dict = {
            "name":
            self._get_name(),
            "epsilon":
            self.loss_config.epsilon,
            "use_double_precision":
            self.loss_config.use_double_precision,
            "local_queue_length":
            self.loss_config.queue.local_queue_length,
            "temperature":
            self.loss_config.temperature,
            "num_prototypes":
            self.loss_config.num_prototypes,
            "num_crops":
            self.loss_config.num_crops,
            "nmb_sinkhornknopp_iters":
            self.loss_config.num_iters,
            "embedding_dim":
            self.loss_config.embedding_dim,
            "temp_hard_assignment_iters":
            self.loss_config.temp_hard_assignment_iters,
        }
        return pprint.pformat(repr_dict, indent=2)


class DenseSwAVCriterion(nn.Module):
    """
    """

    def __init__(
        self,
        temperature: float,
        crops_for_assign: List[int],
        num_crops: int,
        num_iters: int,
        epsilon: float,
        use_double_prec: bool,
        num_prototypes: List[int],
        local_queue_length: int,
        vecs_per_sample: int,
        embedding_dim: int,
        temp_hard_assignment_iters: int,
        output_dir: str,
        comp_coherence_loss: bool,
    ):
        super(DenseSwAVCriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1

        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.num_crops = num_crops
        self.nmb_sinkhornknopp_iters = num_iters
        self.epsilon = epsilon
        self.use_double_prec = use_double_prec
        self.num_prototypes = num_prototypes
        self.nmb_heads = len(self.num_prototypes)
        self.embedding_dim = embedding_dim
        self.temp_hard_assignment_iters = temp_hard_assignment_iters
        self.local_queue_length = local_queue_length
        self.dist_rank = get_rank()
        self.world_size = get_world_size()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))
        self.use_queue = False
        self.vecs_per_sample = vecs_per_sample
        if local_queue_length > 0:
            self.initialize_queue()
            self.use_queue = True
        self.output_dir = output_dir
        self.comp_coherence_loss = comp_coherence_loss

        self.pdist = nn.PairwiseDistance()

    def distributed_sinkhornknopp(self, Q: torch.Tensor):
        """
        Apply the distributed sinknorn optimization on the scores matrix to
        find the assignments
        """
        eps_num_stab = 1e-12
        with torch.no_grad():
            # remove potential infs in Q
            # replace the inf entries with the max of the finite entries in Q
            mask = torch.isinf(Q)
            ind = torch.nonzero(mask)
            if len(ind) > 0:
                for i in ind:
                    Q[i[0], i[1]] = 0
                m = torch.max(Q)
                for i in ind:
                    Q[i[0], i[1]] = m
            sum_Q = torch.sum(Q, dtype=Q.dtype)
            all_reduce_sum(sum_Q)
            Q /= sum_Q

            k = Q.shape[0]
            n = Q.shape[1]
            N = self.world_size * Q.shape[1]

            # we follow the u, r, c and Q notations from
            # https://arxiv.org/abs/1911.05371
            r = torch.ones(k) / k
            c = torch.ones(n) / N
            if self.use_double_prec:
                r, c = r.double(), c.double()

            if self.use_gpu:
                r = r.cuda(non_blocking=True)
                c = c.cuda(non_blocking=True)

            for _ in range(self.nmb_sinkhornknopp_iters):
                u = torch.sum(Q, dim=1, dtype=Q.dtype)
                all_reduce_sum(u)

                # for numerical stability, add a small epsilon value
                # for non-zero Q values.
                if len(torch.nonzero(u == 0)) > 0:
                    Q += eps_num_stab
                    u = torch.sum(Q, dim=1, dtype=Q.dtype)
                    all_reduce_sum(u)
                u = r / u
                # remove potential infs in "u"
                # replace the inf entries with the max of the finite entries in "u"
                mask = torch.isinf(u)
                ind = torch.nonzero(mask)
                if len(ind) > 0:
                    for i in ind:
                        u[i[0]] = 0
                    m = torch.max(u)
                    for i in ind:
                        u[i[0]] = m

                Q *= u.unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
            Q = (Q /
                 torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

            # hard assignment
            if self.num_iteration < self.temp_hard_assignment_iters:
                index_max = torch.max(Q, dim=1)[1]
                Q.zero_()
                Q.scatter_(1, index_max.unsqueeze(1), 1)
            return Q

    def forward(self, projs: torch.Tensor, scores: torch.Tensor,
                regions_idxs: torch.Tensor):
        """
        NOTE: First view == center view == "high-res" view in SwAV
        """

        img_N = len(scores[0])
        view_M = len(scores)
        D = projs[0][0].shape[1]
        K = scores[0][0].shape[1]

        total_loss = 0
        n_term_loss = 0

        if self.comp_coherence_loss:
            rep_projs = []
            rep_scores = []
            loss_coherence = []
            for view_idx in range(view_M):
                rep_img_projs = []
                rep_img_scores = []
                for img_idx in range(img_N):

                    proj = projs[view_idx][img_idx]
                    score = scores[view_idx][img_idx]
                    region_idxs = regions_idxs[view_idx][img_idx]

                    # Remove all unlabeled vectors
                    mask = region_idxs.ge(0)
                    proj = torch.masked_select(proj, mask)
                    proj = torch.reshape(proj, (-1, D))

                    score = torch.masked_select(score, mask)
                    score = torch.reshape(score, (-1, K))

                    region_idxs = torch.masked_select(region_idxs, mask)
                    region_idxs = torch.reshape(region_idxs, (-1, 1))

                    unique_region_idxs = torch.unique(region_idxs)

                    # Regions appended to list in same order for all views
                    rep_proj_regions = []
                    rep_score_regions = []
                    loss_regions = []
                    for region_idx in unique_region_idxs:

                        # Region-specific (by region idx)
                        mask = region_idxs == region_idx

                        proj_region = torch.masked_select(proj, mask)
                        proj_region = torch.reshape(proj_region, (-1, D))
                        score_region = torch.masked_select(score, mask)
                        score_region = torch.reshape(score_region, (-1, K))

                        # Get idx of most compatible prototype vec.
                        highest_scores, _ = torch.max(score_region, dim=1)
                        highest_idx = torch.argmax(highest_scores)
                        # Representative vectors of region w. dim (N)
                        rep_proj_region = proj_region[highest_idx].detach()
                        rep_score_region = score_region[highest_idx]

                        # NOTE: Used for assignment loss
                        rep_proj_regions.append(rep_proj_region)
                        rep_score_regions.append(rep_score_region)

                        # Coherence loss
                        region_dist = self.pdist(proj_region, rep_proj_region)
                        region_dist = torch.mean(region_dist)
                        loss_regions.append(region_dist)

                    # All rep. score vecs for [view_idx][img_idx]
                    # [score_1, ... , score_N] --> matrix (N, K)
                    rep_proj_regions = torch.stack(rep_proj_regions)
                    rep_img_projs.append(rep_proj_regions)
                    rep_score_regions = torch.stack(rep_score_regions)
                    rep_img_scores.append(rep_score_regions)

                    # Coherence loss for [view_idx][img_idx]
                    # [loss_scalar_1, ... , loss_scalar_N] --> loss_scalar
                    loss_regions = torch.stack(loss_regions)
                    loss_regions = torch.mean(loss_regions)
                    loss_coherence.append(loss_regions)

                rep_projs.append(rep_img_projs)
                rep_scores.append(rep_img_scores)

            loss_coherence = torch.stack(loss_coherence)
            loss_coherence = torch.mean(loss_coherence)

            total_loss += LOSS_COHERENCE_W * loss_coherence
            n_term_loss += 1
        else:
            rep_projs = projs
            rep_scores = scores

        with torch.no_grad():
            # Collect all 'view 0' prototype score vectors into single tensor
            view_0_scores = []
            for img_idx in range(img_N):
                view_0_score = rep_scores[0][img_idx]
                view_0_scores.append(view_0_score)
            view_0_scores = torch.cat(view_0_scores)  # (N, K)
            scores_N = view_0_scores.shape[0]

            if self.use_queue:
                queue = getattr(self, "local_queue" + str(0))[0].clone()
                view_0_scores = torch.cat((view_0_scores, queue))

            view_0_scores = view_0_scores / self.epsilon
            # use the log-sum-exp trick for numerical stability.
            M = torch.max(view_0_scores)
            all_reduce_max(M)
            view_0_scores -= M
            view_0_scores = torch.exp(view_0_scores).t()
            # Target assignments for all projections in 'view 0' for all images
            assignments = self.distributed_sinkhornknopp(view_0_scores)
            # Discard queue samples as they only used for Sinkhorn-Knopp algo.
            assignments = assignments[:scores_N]

        loss = 0
        for view_idx in range(1, view_M):

            view_scores = []
            for img_idx in range(img_N):
                view_score = rep_scores[view_idx][img_idx]
                view_scores.append(view_score)
            view_scores = torch.cat(view_scores)

            loss -= torch.mean(
                torch.sum(assignments *
                          self.log_softmax(view_scores / self.temperature),
                          dim=1,
                          dtype=assignments.dtype))

        loss /= (view_M - 1)
        total_loss += LOSS_SWAV_W * loss
        n_term_loss += 1

        # print(LOSS_COHERENCE_W * loss_coherence.item(),
        #       LOSS_SWAV_W * loss.item())

        # stop training if NaN appears and log the output to help debugging
        # TODO (prigoyal): extract the logic to be common for all losses
        # debug_state() method that all losses can override
        if torch.isnan(loss):
            logging.info(
                f"Infinite Loss or NaN. Loss value: {loss}, rank: {self.dist_rank}"
            )
            scores_output_file = os.path.join(
                self.output_dir,
                "rank" + str(self.dist_rank) + "_scores" + str(i) + ".pth",
            )
            assignments_out_file = os.path.join(
                self.output_dir,
                "rank" + str(self.dist_rank) + "_assignments" + str(i) +
                ".pth",
            )
            with PathManager.open(scores_output_file, "wb") as fwrite:
                torch.save(rep_scores, fwrite)
            with PathManager.open(assignments_out_file, "wb") as fwrite:
                torch.save(assignments, fwrite)
            logging.info(f"Saved the scores matrix to: {scores_output_file}")
            logging.info(
                f"Saved the assignment matrix to: {assignments_out_file}")

        total_loss /= n_term_loss
        return total_loss, rep_projs

    def update_emb_queue(self, emb):
        with torch.no_grad():
            bs = len(emb)
            queue = self.local_emb_queue[0]
            # Replace end of queue with beginning
            # a[2:] = a[:-2] ==> [1,2,3,4,5,6] --> [1,2,|1,2,3,4|]
            # NOTE: If batch is larger than queue, queue remains unchanged
            queue[bs:] = queue[:-bs].clone()
            # Replace begining of queue with new vectors
            queue[:bs] = emb
            self.local_emb_queue[0] = queue

    def compute_queue_scores(self, head):
        with torch.no_grad():
            for crop_id in range(len(self.crops_for_assign)):
                for i in range(head.nmb_heads):
                    scores = getattr(head, "prototypes" + str(i))(  # Z C
                        self.local_emb_queue[crop_id]  # (N, 128)
                    )
                    getattr(self, "local_queue" +
                            str(i))[crop_id] = scores  # (1, N, 3000)

    def initialize_queue(self):
        for i in range(self.nmb_heads):
            init_queue = (torch.rand(
                len(self.crops_for_assign),
                self.local_queue_length,
                self.num_prototypes[i],
            ) * 2 - 1)
            # Scores queue (1, N, 3000)
            self.register_buffer("local_queue" + str(i), init_queue)
        stdv = 1.0 / math.sqrt(self.embedding_dim / 3)
        init_queue = (torch.rand(
            len(self.crops_for_assign), self.local_queue_length,
            self.embedding_dim).mul_(2 * stdv).add_(-stdv))
        # Projections queue (1, N, 128)
        self.register_buffer("local_emb_queue", init_queue)

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "use_queue": self.use_queue,
            "local_queue_length": self.local_queue_length,
            "temperature": self.temperature,
            "num_prototypes": self.num_prototypes,
            "num_crops": self.num_crops,
            "nmb_sinkhornknopp_iters": self.nmb_sinkhornknopp_iters,
            "embedding_dim": self.embedding_dim,
            "temp_hard_assignment_iters": self.temp_hard_assignment_iters,
        }
        return pprint.pformat(repr_dict, indent=2)
