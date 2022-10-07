import argparse
import numpy as np
import os
import random
import yaml

from extra_scripts.datasets.dataset_parsers.a2d2_parser import a2d2_parser
from extra_scripts.datasets.dataset_parsers.apollo_daoxianglake_parser import apollo_daoxianglake_parser
from extra_scripts.datasets.dataset_parsers.apolloscape_parser import apolloscape_parser
from extra_scripts.datasets.dataset_parsers.argoverse_parser import argoverse_parser
from extra_scripts.datasets.dataset_parsers.bdd100k_parser import bdd100k_parser
from extra_scripts.datasets.dataset_parsers.boxy_parser import boxy_parser
from extra_scripts.datasets.dataset_parsers.cadc_parser import cadc_parser
from extra_scripts.datasets.dataset_parsers.cityscapes_parser import cityscapes_parser
from extra_scripts.datasets.dataset_parsers.coco_parser import coco_parser
from extra_scripts.datasets.dataset_parsers.comma2k19 import comma2k19_parser
from extra_scripts.datasets.dataset_parsers.comma_ai_parser import comma_ai_parser
from extra_scripts.datasets.dataset_parsers.ddad_parser import ddad_parser
from extra_scripts.datasets.dataset_parsers.ecp_parser import ecp_parser
from extra_scripts.datasets.dataset_parsers.global_road_damage_detection_parser import global_road_damage_detection_parser
from extra_scripts.datasets.dataset_parsers.gta5_parser import gta5_parser
from extra_scripts.datasets.dataset_parsers.h3d_parser import h3d_parser
from extra_scripts.datasets.dataset_parsers.hdd_parser import hdd_parser
from extra_scripts.datasets.dataset_parsers.hevi_parser import hevi_parser
from extra_scripts.datasets.dataset_parsers.honda_hsd_parser import honda_hsd_parser
from extra_scripts.datasets.dataset_parsers.honda_had_parser import honda_had_parser
from extra_scripts.datasets.dataset_parsers.honda_titan_parser import honda_titan_parser
from extra_scripts.datasets.dataset_parsers.idd_parser import idd_parser
from extra_scripts.datasets.dataset_parsers.imagenet_parser import imagenet_parser
from extra_scripts.datasets.dataset_parsers.kitti_parser import kitti_parser
from extra_scripts.datasets.dataset_parsers.lyft_perception_parser import lyft_perception_parser
from extra_scripts.datasets.dataset_parsers.mapillary_vistas_parser import mapillary_vistas_parser
from extra_scripts.datasets.dataset_parsers.nuimages_parser import nuimages_parser
from extra_scripts.datasets.dataset_parsers.nuscenes_parser import nuscenes_parser
from extra_scripts.datasets.dataset_parsers.pandaset_parser import pandaset_parser
from extra_scripts.datasets.dataset_parsers.tusimple_parser import tusimple_parser
from extra_scripts.datasets.dataset_parsers.waymo_parser import waymo_parser


def parse_dataset(dataset: str, path: str):
    """
    Returns a list of image paths contined in a given dataset.

    Args:
        dataset (str): Dataset specifier string (dataset name).
        path (str): Absolute path to root directory of dataset.

    Returns:

    Raises:
        NotImplementedError: Specified dataset does not have a parser.
    """
    if dataset == "a2d2":
        filelist = a2d2_parser(path)
    elif dataset == "apollo_daoxianglake":
        filelist = apollo_daoxianglake_parser(path)
    elif dataset == "apolloscape":
        filelist = apolloscape_parser(path)
    elif dataset == "argoverse":
        filelist = argoverse_parser(path)
    elif dataset == "bdd100k":
        filelist = bdd100k_parser(path)
    elif dataset == "boxy":
        filelist = boxy_parser(path)
    elif dataset == "cadc":
        filelist = cadc_parser(path)
    elif dataset == "cityscapes":
        filelist = cityscapes_parser(path)
    elif dataset == "coco":
        filelist = coco_parser(path)
    elif dataset == "comma2k19":
        filelist = comma2k19_parser(path)
    elif dataset == "comma_ai":
        filelist = comma_ai_parser(path)
    elif dataset == "ddad":
        filelist = ddad_parser(path)
    elif dataset == "ecp":
        filelist = ecp_parser(path)
    elif dataset == "global_road_damage_detection":
        filelist = global_road_damage_detection_parser(path)
    elif dataset == "gta5":
        filelist = gta5_parser(path)
    elif dataset == "h3d":
        filelist = h3d_parser(path)
    elif dataset == "hdd":
        filelist = hdd_parser(path)
    elif dataset == "hevi":
        filelist = hevi_parser(path)
    elif dataset == "honda_hsd":
        filelist = honda_hsd_parser(path)
    elif dataset == "honda_had":
        filelist = honda_had_parser(path)
    elif dataset == "honda_titan":
        filelist = honda_titan_parser(path)
    elif dataset == "imagenet":
        filelist = imagenet_parser(path)
    elif dataset == "idd":
        filelist = idd_parser(path)
    elif dataset == "kitti":
        filelist = kitti_parser(path)
    elif dataset == "lyft_perception":
        filelist = lyft_perception_parser(path)
    elif dataset == "mapillary_vistas":
        filelist = mapillary_vistas_parser(path)
    elif dataset == "nuimages":
        filelist = nuimages_parser(path)
    elif dataset == "nuscenes":
        filelist = nuscenes_parser(path)
    elif dataset == "pandaset":
        filelist = pandaset_parser(path)
    elif dataset == "tusimple":
        filelist = tusimple_parser(path)
    elif dataset == "waymo":
        filelist = waymo_parser(path)
    else:
        raise NotImplementedError(f"Dataset not implemented ({dataset})")

    return filelist


def gen_dataset_filelist(datasets: list, paths: list, shuffle: bool = True):
    """
    Returns a list of paths to images contained in given datasets.

    Args:
        datasets (list str): List dataset specifiers strings (dataset names).
        paths (list str): List of absolute paths to dataset root directories.
                          NOTE: Same order and length as in 'datasets'.
        shuffle (bool): Shuffle list of image paths if True.

    Returns:
        List of absolute image paths.
    """
    # Parses datasets one-by-one and concatenates lists of found image paths.
    img_paths = []
    for (dataset, path) in zip(datasets, paths):
        dataset_img_paths = parse_dataset(dataset, path)
        img_paths += dataset_img_paths

        print(f"Parsing {dataset}\n    {len(dataset_img_paths)} images")

    if shuffle:
        random.shuffle(img_paths)

    return img_paths


def parse_config(config: dict):
    """
    Returns a list of dataset names and root directory paths as specified in
    the configuration file.
    """
    datasets = []
    paths = []
    for dataset_name, properties in config["datasets"].items():
        if properties["use"]:
            datasets.append(dataset_name)
            paths.append(properties["root_path"])

    return datasets, paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates a disk filelist object for multiple datasets')
    parser.add_argument('config_path',
                        type=str,
                        help='Configuration file path')
    parser.add_argument('--out-dir', help='Filelist output directory')
    parser.add_argument('--out-filename',
                        default='filelist.npy',
                        help='Name of generated file list file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    '''
    How to use:
        python extra_scripts/datasets/gen_dataset_filelist.py extra_scripts/
               datasets/dataset_config.yaml
    '''
    args = parse_args()

    config_path = args.config_path
    out_dir = args.out_dir if args.out_dir else "."
    out_filename = args.out_filename

    if os.path.isfile(config_path) is False:
        print(f"Config file does not exist ({config_path})")
        exit()

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datasets, paths = parse_config(config)

    img_paths = gen_dataset_filelist(datasets, paths)

    print(f"Combined dataset size: {len(img_paths):.2E} images")

    filelist_path = os.path.join(out_dir, out_filename)
    np.save(filelist_path, np.array(img_paths))
