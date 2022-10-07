import argparse
from PIL import Image
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from typing import List
import numpy as np
import os
import torch
import torchvision.transforms as transforms

from dataset_parser import (
    DatasetInterface,
    CityscapesDatasetParser,
    A2D2DatasetParser,
)


def standardize_img_size(img, std_img_h=960, std_img_w=1280):
    """
        Returns a resized image with the smallest dimension equal to the
        standard image dimension.

        Args:
            img (PIL.Image): Dataset sample image.
            std_img_h (int): Standard image height.
            std_img_w (int): Standard image width.

        Returns:
            img (PIL.Image)
        """
    w, h = img.size
    aspect_ratio = w / h
    std_aspect_ratio = std_img_w / std_img_h

    if aspect_ratio <= std_aspect_ratio:
        scaling = std_img_w / w
    else:
        scaling = std_img_h / h

    w_ = int(scaling * w)
    h_ = int(scaling * h)
    img = img.resize((w_, h_), Image.BICUBIC)

    return img


def parse_dataset(dataset_name,
                  dataset_path,
                  parse_strings: list = None) -> List:
    """
    Returns a list of absolute image paths for the specified dataset.

    Args:
        datset_name (str):    Dataset specification name.
        dataset_path (str):   Path to dataset root.
        parse_strings (list): List of strings used by glob when searching for
                              sample images.

    Returns:
        img_paths (list): List of image paths as strings.
    """
    if dataset_name == "a2d2":
        dataset = DatasetInterface(A2D2DatasetParser(dataset_path))
    elif dataset_name == "cityscapes":
        dataset = DatasetInterface(CityscapesDatasetParser(dataset_path))
    else:
        raise Exception(f"Invalid dataset name ({dataset_name})")

    img_paths = dataset.parse_imgs(parse_strings)

    return img_paths


def output_type_config(cfg, output_type):
    """
    """
    if output_type == "viz_emb":
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD = True
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD = True
    elif output_type == "backbone":
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
        cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False
    else:
        raise Exception(f"Invalid output type ({output_type})")

    return cfg


def process_output(output, output_type, img_h, img_w):
    """
    """
    # Convert matrix of projection vectors to projection feature map (h)
    if output_type == "viz_emb":
        output = output[0][0].cpu().numpy()
        output = np.reshape(output, (img_h, img_w, -1))
        output = np.transpose(output, (2, 0, 1))

    elif output_type == "backbone":
        raise NotImplementedError()

    else:
        raise Exception(f"Invalid output type ({output_type})")

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pregenerates output feature maps for a dataset.")
    parser.add_argument("config_path", type=str, help="Path to config file.")
    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Path to checkpoint file.")
    parser.add_argument("output_type",
                        type=str,
                        help="Output type choice (viz_emb, backbone).")
    parser.add_argument("dataset_name", type=str, help="Name of dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to dataset root.")
    parser.add_argument("--default-config-path",
                        type=str,
                        default="vissl/config/defaults.yaml",
                        help="Path to default config file")
    parser.add_argument(
        "--std-img-height",
        type=int,
        default=960,
        help="Standard image height for resolution unification.")
    parser.add_argument(
        "--std-img-width",
        type=int,
        default=1280,
        help="Standard image width for resolution unification.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    output_type = args.output_type
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    default_config_path = args.default_config_path
    std_img_h = args.std_img_height
    std_img_w = args.std_img_width

    ###################
    #  Parse dataset
    ###################
    img_paths = parse_dataset(dataset_name, dataset_path)
    print(f"Loaded {len(img_paths)} images")
    print(f"  Ex: [0]: {img_paths[0]}")

    ##################
    #  Set up model
    ##################
    # Load configuration files
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load(default_config_path)
    cfg = OmegaConf.merge(default_config, config)
    cfg = AttrDict(cfg)

    # TODO: Do not know how necessary this specification is?
    # cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = checkpoint_path

    # Configure output extraction
    cfg = output_type_config(cfg, output_type)

    # Initialize model
    model = build_model(cfg.config.MODEL, cfg.config.OPTIMIZER)

    # Load pretrained weights
    state_dict = load_checkpoint(checkpoint_path)
    model.init_model_from_weights_params_file(cfg.config, state_dict)
    model.eval()
    model.cuda()

    #####################
    #  Generate output
    #####################

    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for img_path in img_paths:

        img = Image.open(img_path).convert("RGB")
        img = standardize_img_size(img, std_img_h, std_img_w)
        img_w, img_h = img.size

        x = pipeline(img)
        x = x.unsqueeze(0)
        x = x.cuda()

        with torch.no_grad():
            output = model.forward(x)

        output = process_output(output, output_type, img_h, img_w)

        # Separate 'filename' and 'directory path'
        img_name = img_path.split("/")[-1]
        dir_path = img_path[:-len(img_name)]
        # Remove filetype extension
        img_name = img_name.split(".")[0]

        output_path = os.path.join(dir_path, img_name + ".npy")
        np.save(
            output_path,
            output,
        )
