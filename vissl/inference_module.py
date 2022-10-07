from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
from PIL import Image
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
import torch
from typing import List


class InferenceInterface():
    """
    """
    def __init__(self, inference_module: InferenceModule) -> None:
        self._inference_module = inference_module

    @property
    def inference_module(self) -> InferenceModule:
        return self._inference_module

    @inference_module.setter
    def inference_module(self, inference_module: InferenceModule) -> None:
        self._inference_module = inference_module

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        """
        output = self._inference_module.forward(input)
        return output

    def get_clusters(self) -> torch.tensor:
        cluster_mat = self._inference_module.get_clusters()
        return cluster_mat


class InferenceModule(ABC):
    """
    """
    def __init__(
        self,
        config_path: str,
        default_config_path: str,
        checkpoint_path: str,
        output_type: str,
        use_gpu: bool = True,
    ):
        self.config_path = config_path
        self.default_config_path = default_config_path
        self.checkpoint_path = checkpoint_path
        self.output_type = output_type
        self.use_gpu = use_gpu

        ##################
        #  Set up model
        ##################
        # Load configuration files
        config = OmegaConf.load(config_path)
        default_config = OmegaConf.load(default_config_path)
        cfg = OmegaConf.merge(default_config, config)
        cfg = AttrDict(cfg)

        # Configure output extraction
        self.cfg = self.output_type_config(cfg, output_type)

        # Initialize model
        self.model = build_model(cfg.config.MODEL, cfg.config.OPTIMIZER)

        # Load pretrained weights
        state_dict = load_checkpoint(checkpoint_path)
        self.model.init_model_from_weights_params_file(cfg.config, state_dict)
        self.model.eval()
        if self.use_gpu:
            self.model.cuda()

    @staticmethod
    def output_type_config(cfg, output_type):
        """
        """
        if output_type == "trunk":
            cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
            cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
        elif output_type == "head":
            cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
            cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD = True
            cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD = True
        # elif output_type == "backbone":
        #     cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
        #     cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
        #     cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
        #     cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False
        else:
            raise Exception(f"Invalid output type ({output_type})")
        return cfg

    @abstractmethod
    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        """
        pass

    @abstractmethod
    def get_clusters(self) -> torch.tensor:
        """
        """
        pass


class DenseSwAVModule(InferenceModule):
    """
    """
    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Args:
            input (torch.tensor): Input tensor w. dim (batch_N, 3, H, W)
        """

        batch_n, _, input_h, input_w = input.shape

        with torch.no_grad():
            output = self.model.forward(input)

        # Postprocess output into (N, D, H, W) tensor
        output = self.process_output(output, self.output_type, batch_n,
                                     input_h, input_w)

        return output

    @staticmethod
    def process_output(output, output_type, batch_n, input_h, input_w):
        """
        """
        if output_type == "trunk":
            # Extract (N,D,H,W) tensor from list
            output = output[0]
            # Normalize (as usually done within head)
            output = torch.nn.functional.normalize(output, dim=1, p=2)

        elif output_type == "head":
            # Output --> [projection_tensor, score_tensor]
            output = output[0]

        elif output_type == "backbone":
            raise NotImplementedError()

        else:
            raise Exception(f"Invalid output type ({output_type})")

        return output

    def get_clusters(self) -> torch.tensor:
        """
        Returns row vector matrix with prototype vectors w. dim (K, D)
        """
        return list(self.model.heads[0].prototypes0.parameters())[0]


"""
Usage example
"""
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("config_path", type=str, help="Path to config file.")
    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Path to checkpoint file.")
    parser.add_argument("output_type",
                        type=str,
                        help="Output type choice (viz_emb, backbone).")
    parser.add_argument("--default-config-path",
                        type=str,
                        default="vissl/config/defaults.yaml",
                        help="Path to default config file")
    args = parser.parse_args()

    img_path = args.img_path
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    output_type = args.output_type
    default_config_path = args.default_config_path

    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 1. Read image
    img = Image.open(img_path).convert("RGB")
    input = pipeline(img)
    input = input.unsqueeze(0)
    # Transform into batch
    input = input.repeat(2, 1, 1, 1)

    # 2. Setup inference module
    vissl_module = InferenceInterface(
        DenseSwAVModule(
            config_path,
            default_config_path,
            checkpoint_path,
            output_type,
            use_gpu=True,
        ))

    # 3. Run inference
    input = input.cuda()
    output = vissl_module.forward(input)

    print(f"input:  {input.shape}")
    print(f"output: {output.shape}")

    # 4. Visualize output
    #output = output.cpu().data
    #pca = PCA(n_components=3)
    #pca.fit(output_vec)
#
#vecs_transf = pca.transform(output_vec)
#
#w, h = img.size
#feat_map_transf = np.reshape(vecs_transf, (h, w, -1))
#
#a = feat_map_transf
#a = (a - np.min(a)) / (np.max(a) - np.min(a))
#a *= 255
#a = a.astype(int)
#
#plt.subplot(1, 2, 1)
#plt.imshow(img)
#plt.subplot(1, 2, 2)
#plt.imshow(img)
#plt.imshow(a, alpha=0.75)
#plt.show()
