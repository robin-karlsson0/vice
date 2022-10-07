# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from classy_vision.models import ClassyModel, register_model
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models.heads import SwAVPrototypesHead, get_model_head
from vissl.models.model_helpers import (
    get_trunk_output_feature_names,
    is_feature_extractor_model,
)
from vissl.models.trunks import get_model_trunk
from vissl.models.trunks.feature_extractor import FeatureExtractorModel
from vissl.utils.checkpoint import (
    CheckpointLoader,
    init_model_from_consolidated_weights,
)
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.fsdp_utils import fsdp_recursive_reset_lazy_init
from vissl.utils.misc import set_torch_seed

# Temporary imports
# import debug_func
# import numpy as np


@register_model("multi_input_output_model")
class BaseSSLMultiInputOutputModel(ClassyModel):
    """
    Class to implement a Self-Supervised model.
    The model is split into `trunk' that computes features and `head' that
    computes outputs (projections, classifications etc)

    This class supports many use cases:
    1. Model producing single output as in standard supervised ImageNet
       training
    2. Model producing multiple outputs (Multi-task)
    3. Model producing multiple outputs from different features (layers)
       from the trunk (useful in linear evaluation of features from several
       model layers)
    4. Model that accepts multiple inputs (e.g. image and patches as in PIRL
       appraoch).
    5. Model where the trunk is frozen.
    6. Model that supports multiple resolutions inputs as in SwAV

    * How to specify heads?
        For information on heads see the `_get_heads()` function

    * What inputs do `heads' operate on?
        One can specify the `input' to heads mapping in the list
        MULTI_INPUT_HEAD_MAPPING. See the _setup_multi_input_head_mapping()
        function for details.
    """

    def __init__(self, model_config, optimizer_config):
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        super().__init__()
        self.eval_mode = None  # this is just informational
        self.local_rank, _ = get_machine_local_and_dist_rank()
        self.trunk = self._get_trunk()
        self.heads = nn.ModuleList()
        self.head_names = []
        self._output_feature_names = get_trunk_output_feature_names(
            self.model_config)
        self._get_heads()
        self._setup_multi_input_head_mapping()

        self.vice_with_mean_vec = model_config['TRUNK']['DEEPLABV3PLUS'][
            'use_mean_vec']

    def multi_input_with_head_mapping_forward(self, batch):
        """
        Perform forward pass (trunk + heads) separately on each input and
        return the model output on all inputs as a list.
        """
        all_outputs = []
        for input_idx in range(len(
                self.model_config.MULTI_INPUT_HEAD_MAPPING)):
            input_key = self.model_config.MULTI_INPUT_HEAD_MAPPING[input_idx][
                0]
            # heads that are used for this input
            heads = self._input_to_head_map[input_key]
            feature_names = self._input_to_eval_features_map[input_key]
            outputs = self.single_input_forward(batch[input_key],
                                                feature_names, heads)
            if len(outputs) == 1:
                # single head. do not make nested list
                outputs = outputs[0]
            all_outputs.append(outputs)
        return all_outputs

    def dense_multi_res_input_forward(self, sample, feature_names):
        """
        """
        batch = sample["input"]
        target = sample["target"]
        superpixels = sample["superpixels"][0]

        img_N = sample["input"][0].shape[0]
        view_M = len(sample["input"])

        assert isinstance(batch, list)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor(
                [inp.shape[-1] for inp in batch]),
                                     return_counts=True)[1],
            0,
        )
        feats = []
        start_idx = 0
        # in order to optimize memory usage, we can do single pass per
        # crop as well. Set the flag to be true.
        if self.model_config.SINGLE_PASS_EVERY_CROP:
            idx_crops = torch.Tensor(list(range(1, 1 + len(batch)))).int()

        for end_idx in idx_crops:
            feat = self.trunk(torch.cat(batch[start_idx:end_idx]),
                              feature_names)
            start_idx = end_idx
            assert len(feat) == 1
            feats.append(feat[0])
        feats = torch.cat(feats)  # (8, 128, 400, 400)

        # debug_func.plot_model_input_crops(
        #     torch.cat(batch), target, img_N, view_M)
        #
        # for view_idx in range(view_M):
        #    for img_idx in range(img_N):
        #        sample_idx = view_idx*img_N + img_idx
        #        if torch.any(target[sample_idx] < 0):
        #            batch[view_idx][img_idx] = torch.flip(
        #                batch[view_idx][img_idx], (-1,))
        #
        # debug_func.plot_model_input_crops(
        #     torch.cat(batch), target, img_N, view_M)

        # For each image, slice common region for all views
        view_projs = []
        view_scores = []
        view_scores_regions = []
        for view_idx in range(view_M):
            img_projs = []
            img_scores = []
            img_scores_regions = []
            for img_idx in range(img_N):
                sample_idx = view_idx * img_N + img_idx

                # Non-negate flipped labels
                is_flipped = False
                if torch.any(target[sample_idx] < 0):
                    # target_ = torch.neg(target[sample_idx])
                    is_flipped = True
                # else:
                # target_ = target[sample_idx]

                # tl, br = target_
                # x_tl, y_tl = tl
                # x_br, y_br = br
                # feat_crop = feats[sample_idx:sample_idx + 1, :, y_tl:y_br,
                #                  x_tl:x_br]
                feat_crop = feats[sample_idx:sample_idx + 1, :]
                superpixel = superpixels[view_idx][img_idx]
                # superpixel = superpixel[y_tl:y_br, x_tl:x_br]

                # Restore feature maps to same orientation for flipped views
                if is_flipped:
                    feat_crop = torch.flip(feat_crop, (-1, ))
                    superpixel = torch.flip(superpixel, (-1, ))

                # Replace features with mean feature for regions
                if self.vice_with_mean_vec:
                    feat_reduced = []
                    superpixel_reduced = []
                    region_idxs = torch.unique(superpixel, sorted=True)
                    region_idxs = region_idxs[region_idxs >
                                              0]  # Remove ignored
                    for region_idx in region_idxs:
                        region_mask = superpixel == region_idx
                        # Extract matrix of column vectors [v1, ..., vN]
                        vecs = feat_crop[:, :, region_mask]
                        mean_vec = torch.mean(vecs, 2, keepdim=True)
                        feat_reduced.append(mean_vec)
                        superpixel_reduced.append(region_idx)

                    feat_crop = torch.stack(feat_reduced, dim=3)
                    superpixel = torch.stack(superpixel_reduced, dim=0)
                else:
                    # Reshape to same order as 'proj', 'scores' vector array
                    superpixel = torch.reshape(superpixel, (1, -1)).T

                feat_crop = [feat_crop]

                # (1, H_crop, W_crop)
                # edge_crop = edges[img_idx][view_idx, :, y_tl:y_br, x_tl:x_br]

                # (1, 128, 223, 149)
                out = self.heads_forward(feat_crop, self.heads)
                proj, scores = out[0]

                img_projs.append(proj)
                img_scores.append(scores)
                img_scores_regions.append(superpixel)

            view_projs.append(img_projs)
            view_scores.append(img_scores)
            view_scores_regions.append(img_scores_regions)

        # # Remove non-mutual region idxs
        # for img_idx in range(img_N):
        #     # View 0
        #     region_idxs = view_scores_regions[0][img_idx]
        #     mask = region_idxs.ge(0)
        #     region_idxs = torch.masked_select(region_idxs, mask)
        #     common_idxs = torch.unique(region_idxs)
        #     common_idxs = set(common_idxs.cpu().numpy())
        #     # View m
        #     for view_idx in range(1, view_M):
        #         region_idxs = view_scores_regions[view_idx][img_idx]
        #         mask = region_idxs.ge(0)
        #         region_idxs = torch.masked_select(region_idxs, mask)
        #         unique_idxs = torch.unique(region_idxs)
        #         unique_idxs = set(unique_idxs.cpu().numpy())
        #         common_idxs.intersection(unique_idxs)

        #     for view_idx in range(0, view_M):

        # print("\nssl_model")
        # for idx in range(len(view_scores_regions)):
        #     superpixel = view_scores_regions[idx][0]
        #     idx_set = np.unique(superpixel.cpu().numpy())
        #     print(idx, len(idx_set))

        # Indexing: model_output[view_idx][img_idx] --> torch.Tensor
        model_output = {
            'projs': view_projs,
            'scores': view_scores,
            'regions': view_scores_regions
        }

        return [model_output]

    def multi_res_input_forward(self, batch, feature_names):
        """
        Perform forward pass separately on each resolution input.
        The inputs corresponding to a single resolution are clubbed and single
        forward is run on the same resolution inputs. Hence we do several
        forward passes = number of different resolutions used. We then
        concatenate all the output features. Then run the head forward on the
        concatenated features.
        """
        assert isinstance(batch, list)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor(
                [inp.shape[-1] for inp in batch]),
                                     return_counts=True)[1],
            0,
        )

        feats = []
        start_idx = 0
        # in order to optimize memory usage, we can do single pass per
        # crop as well. Set the flag to be true.
        if self.model_config.SINGLE_PASS_EVERY_CROP:
            idx_crops = torch.Tensor(list(range(1, 1 + len(batch)))).int()
        for end_idx in idx_crops:
            # feat (list): [torch.Tensor (4, 2048) ]
            #     feat == backbone features 'h'
            # batch_0: |0 --> |2 ... | (4, 3, 224, 224) <-- High-res samples
            # batch_1: [ ... |2 --> |8 ... ] (12, 3, 96, 96) <-- Low-res
            #          samples
            feat = self.trunk(torch.cat(batch[start_idx:end_idx]),
                              feature_names)
            # Unroll (10, 128, 300, 300) --> (900000, 128) ?
            # if self.model_config['TRUNK']['NAME'] == 'deeplabv3plus':
            #    pass
            start_idx = end_idx
            assert len(feat) == 1
            feats.append(feat[0])
        # feats: [torch.Tensor (4, 2048), torch.Tensor (12, 2048)]
        # ==> feats: [torch.Tensor (16, 2048)]
        feats = [torch.cat(feats)]
        # [ [torch.Tensor (16, 128), torch.Tensor (16, 3000)] ]
        #    Projected Z
        return self.heads_forward(feats, self.heads)

    def single_input_forward(self, batch, feature_names, heads):
        """
        Simply run the trunk and heads forward on the input tensor. We run the
        trunk first and then the heads on the trunk output.
        If the model is trunk feature extraction only, then we simply return
        the output of the trunk.
        """
        assert isinstance(batch,
                          (torch.Tensor, MultiDimensionalTensor)), type(batch)
        feats = self.trunk(batch, feature_names)
        # if we are interested in evaluating the trunk only, we return the
        # output of the trunk and don't forward through the heads
        if (self.model_config["FEATURE_EVAL_SETTINGS"]["EVAL_MODE_ON"]
                and self.model_config["FEATURE_EVAL_SETTINGS"]
            ["EXTRACT_TRUNK_FEATURES_ONLY"]):
            return feats
        return self.heads_forward(feats, heads)

    def heads_forward(self, feats, heads):
        """
        Run the forward of the head on the trunk output features.
        We have 2 cases:
            1. #heads = #feats -> example training linear classifiers on
                        various layers.
               We run one head on the corresponding feature.
            2. #feats = 1 and #heads > 1 -> head consists of many layers to be
                        run sequentially.
               #outputs = 1
        """
        # Example case: training linear classifiers on various layers
        if len(feats) == len(heads):
            output = []
            for feat, head in zip(feats, heads):
                # List [
                #    0: Projected features Z (16, 128) <-- g( H (16, 2024) )
                #    1: Scores C^T Z (16, 3000)
                # ]
                output.append(head(feat))
            # output: [ [Z tensor (16, 128), Scores tensor (16, 3000)] ]
            return output
        # Example case: Head consisting of several layers
        elif (len(heads) > 1) and (len(feats) == 1):
            output = feats[0]
            for head in heads:
                output = head(output)
            # our model is multiple output.
            return [output]
        else:
            logging.error(
                f"Mismatch in #head: {len(heads)} and #features: {len(feats)}")

    def forward(self, batch):
        """
        Main forward of the model. Depending on the model type the calls are
        patched to the suitable function.
        """
        if self.model_config.TRUNK.NAME in ['deeplabv3plus', 'fpn'
                                            ] and isinstance(batch, dict):
            return self.dense_multi_res_input_forward(
                batch, self._output_feature_names)

        if len(self.model_config.MULTI_INPUT_HEAD_MAPPING) > 0:
            # this model accepts multiple types of inputs and a separate
            # head is applied to each model output of a given input.
            return self.multi_input_with_head_mapping_forward(batch)

        if isinstance(batch, list):
            return self.multi_res_input_forward(batch,
                                                self._output_feature_names)

        return self.single_input_forward(batch, self._output_feature_names,
                                         self.heads)

    def freeze_head(self):
        """
        Freeze the model head by setting requires_grad=False for all the
        parameters
        """
        logging.info("Freezing model heads...")
        for head in self.heads:
            for param in head.parameters():
                param.requires_grad = False

    def freeze_trunk(self):
        """
        Freeze the model trunk by setting requires_grad=False for all the
        parameters
        """
        logging.info("Freezing model trunk...")
        for param in self.trunk.parameters():
            param.requires_grad = False

    def freeze_head_and_trunk(self):
        """
        Freeze the full model including the heads and the trunk. In 99% cases,
        we do not use the pretext head as it is specific to the self-supervised
        pretext task. But in case of some models like NPID, SimCLR, SwAV, the
        head is essentially a low dimensional feature projection which we want
        to use. Hence, we provide utility to freeze the full model.
        """
        logging.info("Freezing model...")
        self.freeze_trunk()
        self.freeze_head()

    def is_fully_frozen_model(self):
        """
        Look at all the parameters of the model (trunk + heads) and check if
        there is any trainable parameter. if not, the model is completely
        frozen.
        """
        trunk_params_list = self.trunk.parameters()
        heads_params_list = self.heads.parameters()
        trunk_trainable_params = list(
            filter(lambda x: x.requires_grad, trunk_params_list))
        heads_trainable_params = list(
            filter(lambda x: x.requires_grad, heads_params_list))
        if len(trunk_trainable_params) == 0 and len(
                heads_trainable_params) == 0:
            return True
        return False

    def get_features(self, batch):
        """
        Run the trunk forward on the input batch. This give us the features
        from the trunk at several layers of the model.

        In case of feature extraction, we don't run the heads and only the
        trunk. The trunk will already have the feature extractor Pooling layers
        and flattened features attached. feature extractor heads are part of
        the trunk already.
        """
        feats = self.trunk(batch)
        return feats

    def _get_trunk(self):
        """
        Construct the model trunk given the architecture specified

        The trunks could be convnet (AlexNet, ResNe(X)t, RegNet,...etc),
        transformers etc.
        """
        # if we are going to evaluate trunk only we shift to feature extractor
        # backbone
        if is_feature_extractor_model(self.model_config):
            self.eval_mode = True
            return FeatureExtractorModel(self.model_config)
        else:
            self.eval_mode = False
            trunk_name = self.model_config.TRUNK.NAME
            return get_model_trunk(trunk_name)(self.model_config, trunk_name)

    def _build_head_module(self, head_param):
        """
        Given the head, contruct the head.

        Args:
            head_param: The head param is a list containing:
                        head_param = [
                            head_name : str,
                            head_settings: dict containing head settings
                        ]

                        Example:
                            head_param = [
                                "mlp",
                                {"dims": [2048, 128]}
                            ]
        Returns:
            pytorch module for the head
        """
        head_name, head_kwargs = head_param[0], head_param[1]
        head_module = get_model_head(head_name)(self.model_config,
                                                **head_kwargs)
        return head_module

    def _get_heads(self):
        """
        This function creates the heads needed by the module.
        HEAD.PARAMS is a list containing parameters for (multiple) heads.
        Each head consist of head_modules that can be composed in different
        ways.

        * Head Module
            A head_module is specified as a list ["name", kwargs], for example,
            ["mlp", {"dims": [2048, 128]}]

        * Heads can be applied to different types of inputs.
          See `_setup_multi_input_head_mapping`

        Examples of Heads one can specify:
        * Case1: Simple Head containing single module - Single Input, Single
                 output
            ["mlp", {"dims": [2048, 128]}]

        * Case2: Complex Head containing chain of head modules
          Single Input, Single output
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False,
                         "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}]
            ]

        * Case3: Multiple Heads (example 2 heads) - Single input, multiple
                 output
            Can be used for multi-task learning
            # head 0
            [
                ["mlp", {"dims": [2048, 128]}]
            ],
            # head 1
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False,
                         "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}],
            ]

        * Case4: Multiple Heads (example 5 simple heads) - Single input,
                 multiple output.
          For example, used in linear evaluation of models
            ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
            ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],

        """
        for head_param in self.model_config.HEAD.PARAMS:
            if isinstance(head_param[0], list):
                # head is composed of several modules
                head_type, head_modules = [], []
                for idx in range(len(head_param)):
                    with set_torch_seed(self.model_config._MODEL_INIT_SEED +
                                        idx):
                        head_modules.append(
                            self._build_head_module(head_param[idx]))
                    head_type.append(head_param[idx][0])
                head_name = "->".join(head_type)
                head = nn.Sequential(*head_modules)
            else:
                # head is a single module
                head_name = head_param[0]
                with set_torch_seed(self.model_config._MODEL_INIT_SEED):
                    head = self._build_head_module(head_param)
            self.heads.append(head)
            self.head_names.append(head_name)

    def _setup_multi_input_head_mapping(self):
        """
        Used for PIRL style training where the model operates
        on image and the patches.

        Assumptions:
        - This assumes that the same trunk is used to extract features
          for the different types of inputs.
        - One head only operates on one kind of input, Every individual
          head can contain several layers. See _get_heads() function for
          examples.

        * Specify Input -> Trunk Features mapping
           Like in the single input case, the heads can operate on features
           from different layers. In this case, we specify
           MODEL.MULTI_INPUT_HEAD_MAPPING to be a list like:
           [
               ["input_key", [list of features heads is applied on]]
           ]
           For example: for a model that applies two heads on images
                        and one head on patches
                        [
                            ["images", ["res5", "res4"]],
                            ["patches", ["res3"]],
                        ]
        """
        if len(self.model_config.MULTI_INPUT_HEAD_MAPPING) == 0:
            return

        assert len(self.model_config.MULTI_INPUT_HEAD_MAPPING) == len(
            self.heads
        ), "MULTI_INPUT_HEAD_MAPPING must be a list of length == #heads"

        # create many-to-one mapping from input_key to head
        self._input_to_head_map = {}
        for idx in range(len(self.model_config.MULTI_INPUT_HEAD_MAPPING)):
            key = self.model_config.MULTI_INPUT_HEAD_MAPPING[idx][0]
            if key not in self._input_to_head_map:
                self._input_to_head_map[key] = []
            self._input_to_head_map[key].append(self.heads[idx])

        # create many-to-one mapping from input key to eval features
        self._input_to_eval_features_map = {}
        for input_idx in range(len(
                self.model_config.MULTI_INPUT_HEAD_MAPPING)):
            key = self.model_config.MULTI_INPUT_HEAD_MAPPING[input_idx][0]
            eval_layer_names = self.model_config.MULTI_INPUT_HEAD_MAPPING[
                input_idx][1]
            if key in self._input_to_eval_features_map:
                raise ValueError(f"duplicate key {key} \
                    specified for MODEL.MULTI_INPUT_HEAD_MAPPING.")
            self._input_to_eval_features_map[key] = eval_layer_names

    def get_classy_state(self, deep_copy=False):
        """
        Return the model state (trunk + heads) to checkpoint.

        We call this on the state.base_model which is not wrapped with DDP.
        get the model state_dict to checkpoint
        """

        if isinstance(self.trunk, FSDP):
            trunk_state_dict = self.trunk.local_state_dict()
            trunk_metadata_dict = self.trunk.local_metadata_dict()
        elif isinstance(self.trunk, FeatureExtractorModel) and isinstance(
                self.trunk.base_model, FSDP):
            trunk_state_dict = self.trunk.base_model.local_state_dict()
            trunk_metadata_dict = self.trunk.base_model.local_metadata_dict()
        else:
            trunk_state_dict = self.trunk.state_dict()
            trunk_metadata_dict = {}

        if any(isinstance(head, FSDP) for head in self.heads):
            heads_state_dict = [
                head.local_state_dict()
                if isinstance(head, FSDP) else head.state_dict()
                for head in self.heads
            ]
        else:
            heads_state_dict = self.heads.state_dict()

        model_state_dict = {
            "model": {
                "trunk": trunk_state_dict,
                "heads": heads_state_dict
            },
            "meta": {
                "trunk": trunk_metadata_dict
            },
        }
        if deep_copy:
            model_state_dict = copy.deepcopy(model_state_dict)
        # ................... DEBUG ...............................
        # from vissl.utils.checkpoint import print_state_dict_shapes

        # print_state_dict_shapes(trunk_state_dict)
        # print_state_dict_shapes_shapes(heads_state_dict)
        return model_state_dict

    def set_classy_state(self, state, strict=True):
        """
        Initialize the model trunk and head from the state dictionary.

        We call this on the state.base_model which is not wrapped with DDP.
        load the model from checkpoint.
        """
        from vissl.utils.checkpoint import print_loaded_dict_info

        # Loading the trunk
        logging.info(f"Rank {self.local_rank}: Loading Trunk state dict....")
        if isinstance(self.trunk, FSDP):
            self.trunk.load_local_state_dict(state["model"]["trunk"])
            fsdp_recursive_reset_lazy_init(self.trunk)
        elif isinstance(self.trunk, FeatureExtractorModel) and isinstance(
                self.trunk.base_model, FSDP):
            self.trunk.base_model.load_local_state_dict(
                state["model"]["trunk"])
            fsdp_recursive_reset_lazy_init(self.trunk.base_model)
        else:
            self.trunk.load_state_dict(state["model"]["trunk"])

        # Loading the head
        logging.info(f"Rank {self.local_rank}: Loading Heads state dict....")
        if any(isinstance(head, FSDP) for head in self.heads):
            for i, head in enumerate(self.heads):
                if isinstance(head, FSDP):
                    head.load_local_state_dict(state["model"]["heads"][i])
                    fsdp_recursive_reset_lazy_init(head)
                else:
                    self.head.load_state_dict(state["model"]["heads"][i])
        else:
            # sometimes, we want to load the partial head only, so strict=False
            self.heads.load_state_dict(state["model"]["heads"], strict=False)

        # Print debug information about layers loaded.
        #
        # Get the model state dict original (if FSDP, calling it on all ranks.)
        logging.info(f"Rank {self.local_rank}: Model state dict loaded!")
        if isinstance(self.trunk, FSDP) or any(
                isinstance(head, FSDP) for head in self.heads):
            return  # TODO (Quentin) - log the weights of the loaded shard

        if self.local_rank == 0:
            trunk_state_dict, heads_state_dict = (
                self.trunk.state_dict(),
                self.heads.state_dict(),
            )
            model_state_dict = {}
            model_state_dict.update(trunk_state_dict)
            model_state_dict.update(heads_state_dict)

            # get the checkpoint state dict
            checkpoint_state_dict = {}
            checkpoint_state_dict.update(state["model"]["trunk"])
            checkpoint_state_dict.update(state["model"]["heads"])
            params_from_file = self.model_config["WEIGHTS_INIT"]
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            print_loaded_dict_info(
                model_state_dict,
                checkpoint_state_dict,
                skip_layers=skip_layers,
                model_config=self.model_config,
            )

    def init_model_from_weights_params_file(self, config: AttrDict,
                                            checkpoint: Dict[str, Any]):
        """
        We initialize the weights from this checkpoint. However, we don't care
        about the other metadata like iteration number etc.
        So the method only reads the state_dict
        """

        # TODO (Quentin) - support: different number of nodes + different
        # checkpoint formats + fine tuning
        # Special cases in which we want to evaluate a model trained with FSDP:
        # - we need to benchmark it in FSDP mode as well and with the same
        #   number of workers
        # - we need to have it trained with VISSL (no support for other
        #   checkpoint types for now)
        if isinstance(self.trunk, FeatureExtractorModel) and isinstance(
                self.trunk.base_model, FSDP):
            CheckpointLoader.init_fsdp_model_from_weights(
                self.trunk.base_model,
                checkpoint,
                weights_path=[
                    "classy_state_dict", "base_model", "model", "trunk"
                ],
            )
            fsdp_recursive_reset_lazy_init(self.trunk.base_model)
        elif isinstance(self.trunk, FSDP):
            CheckpointLoader.init_fsdp_model_from_weights(
                self.trunk,
                checkpoint,
                weights_path=[
                    "classy_state_dict", "base_model", "model", "trunk"
                ],
            )
            fsdp_recursive_reset_lazy_init(self.trunk)

        # General case: support for multiple format of checkpoint
        else:
            params_from_file = config["MODEL"]["WEIGHTS_INIT"]
            skip_layers = params_from_file.get("SKIP_LAYERS", [])
            replace_prefix = params_from_file.get("REMOVE_PREFIX", None)
            append_prefix = params_from_file.get("APPEND_PREFIX", None)
            state_dict_key_name = params_from_file.get("STATE_DICT_KEY_NAME",
                                                       None)
            init_model_from_consolidated_weights(
                config,
                self,
                checkpoint,
                state_dict_key_name=state_dict_key_name,
                skip_layers=skip_layers,
                replace_prefix=replace_prefix,
                append_prefix=append_prefix,
            )

    def is_clustering_model(self):
        """
        Whether or not the model is a clustering model
        (only supports SwAV for the moment)
        """
        return len(self.heads) == 1 and any(
            isinstance(module, SwAVPrototypesHead)
            for module in self.heads[0].modules())

    @property
    def num_classes(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        """
        Not implemented and not required
        """
        raise NotImplementedError

    def validate(self, dataset_output_shape):
        """
        Not implemented and not required
        """
        raise NotImplementedError
