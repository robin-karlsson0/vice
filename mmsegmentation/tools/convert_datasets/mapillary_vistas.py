#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX = '_trainIds.png'

# Cityscapes-like 'trainId' value
#   key: RGB color, value: trainId (from Cityscapes)
SEG_COLOR_DICT_CITYSCAPES = {
    (0, 0, 0): 255,  # Ignore
    (111, 74, 0): 255,  # Dynamic --> Ignore
    (81, 0, 81): 255,  # Ground --> Ignore
    (128, 64, 128): 0,  # Road
    (244, 35, 232): 1,  # Sidewzalk
    (250, 170, 160): 255,  # Parking --> Ignore
    (230, 150, 140): 255,  # Rail track --> Ignore
    (70, 70, 70): 2,  # Building
    (102, 102, 156): 3,  # Wall
    (190, 153, 153): 4,  # Fence
    (180, 165, 180): 255,  # Guard rail
    (150, 100, 100): 255,  # Bridge
    (150, 120, 90): 255,  # Tunnel
    (153, 153, 153): 5,  # Pole
    (250, 170, 30): 6,  # Traffic light
    (220, 220, 0): 7,  # Traffic sign
    (107, 142, 35): 8,  # Vegetation
    (152, 251, 152): 9,  # Terrain
    (70, 130, 180): 10,  # Sky
    (220, 20, 60): 11,  # Person
    (255, 0, 0): 12,  # Rider
    (0, 0, 142): 13,  # Car
    (0, 0, 70): 14,  # Truck
    (0, 60, 100): 15,  # Bus
    (0, 0, 90): 255,  # Caravan
    (0, 0, 110): 255,  # Trailer
    (0, 80, 100): 16,  # Train
    (0, 0, 230): 17,  # Motorcycle (license plate --> Car)
    (119, 11, 32): 18,  # Bicycle
}


def modify_label_filename(label_filepath, label_choice):
    """Returns a mmsegmentation-compatible label filename."""
    # Ensure that label filenames are modified only once
    if 'TrainIds.png' in label_filepath:
        return label_filepath

    if label_choice == 'cityscapes':
        label_filepath = label_filepath.replace('.png', LABEL_SUFFIX)
    else:
        raise ValueError
    return label_filepath


def convert_cityscapes_trainids(label_filepath, ignore_id=255):
    """Saves a new semantic label following the Cityscapes 'trainids' format.

    The new image is saved into the same directory as the original image having
    an additional suffix.
    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    # Read label file as Numpy array (H, W, 3)
    orig_label = mmcv.imread(label_filepath, channel_order='rgb')

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    mod_label = ignore_id * np.ones((H, W), dtype=int)

    seg_colors = list(SEG_COLOR_DICT_CITYSCAPES.keys())
    for seg_color in seg_colors:
        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # Take the product channel-wise to falsify any partial match and
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        mask = mask.astype(bool)
        # Segment masked elements with 'trainIds' value
        mod_label[mask] = SEG_COLOR_DICT_CITYSCAPES[seg_color]

    # Save new 'trainids' semantic label
    label_filepath = modify_label_filename(label_filepath, 'cityscapes')
    label_img = mod_label.astype(np.uint8)
    mmcv.imwrite(label_img, label_filepath)


def restructure_vistas_directory(vistas_path,
                                 train_on_val_and_test=False,
                                 use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the Mapillary Vistas dataset conform to the mmsegmentation
    frameworks expected dataset structure.

    └── img_dir
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── ann_dir
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        |   ...
        ...
    Args:
        vistas_path: Absolute path to the Mapillary Vistas 'vistas/' directory.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'testing'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'testing'))

    for split in ['training', 'validation', 'testing']:
        img_filepaths = glob.glob(f'{vistas_path}/{split}/images/*.jpg')

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:

            img_filename = img_filepath.split('/')[-1]

            ann_filename = img_filename[:-4] + LABEL_SUFFIX
            ann_filepath = f'{vistas_path}/{split}/v2.0/labels/{ann_filename}'

            img_linkpath = f'{vistas_path}/img_dir/{split}/{img_filename}'
            if split == 'testing':
                ann_linkpath = None
            else:
                ann_linkpath = f'{vistas_path}/ann_dir/{split}/{ann_filename}'

            if use_symlinks:
                # NOTE: Can only create new symlinks if no priors ones exists
                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    if split != 'testing':
                        symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

            else:
                copyfile(img_filepath, img_linkpath)
                if split != 'testing':
                    copyfile(ann_filepath, ann_linkpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary Vistas annotations to trainIds')
    parser.add_argument(
        'vistas_path',
        help='Mapillary vistas segmentation data absolute path\
                           (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-convert',
        dest='convert',
        action='store_false',
        help='Skips converting label images')
    parser.set_defaults(convert=True)
    parser.add_argument(
        '--no-restruct',
        dest='restruct',
        action='store_false',
        help='Skips restructuring directory structure')
    parser.set_defaults(restruct=True)
    parser.add_argument(
        '--choice',
        default='cityscapes',
        help='Label conversion type choice: \'cityscapes\' (18 classes)')
    parser.add_argument(
        '--train-on-val-and-test',
        dest='train_on_val_and_test',
        action='store_true',
        help='Use validation and test samples as training samples')
    parser.set_defaults(train_on_val_and_test=False)
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    parser.add_argument(
        '--no-symlink',
        dest='symlink',
        action='store_false',
        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """A script for making the Mapillary Vistas dataset compatible with
    mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/mapillary_vistas)

    Segmentation label conversion:

        The function 'convert_TYPE_trainids()' converts all RGB segmentation to
        their corresponding categorical segmentation and saves them as new
        label image files.

        Label choice 'cityscapes' (default) results in labels with 18 classes
        with the filename suffix '_trainIds.png'.

    Dataset split:

        Arranges samples into 'train', 'val', and 'test' splits according to
        predetermined directory structure

    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is usefull in the synthetic-to-real domain adaptation
    experiment setting.

    Add `--nproc N` for multiprocessing using N threads.

    Example usage:
        python tools/convert_datasets/mapillary_vistas.py
            abs_path/to/mapillary_vistas
    """
    args = parse_args()
    vistas_path = args.vistas_path
    out_dir = args.out_dir if args.out_dir else vistas_path
    mmcv.mkdir_or_exist(out_dir)

    # Convert segmentation images to the Cityscapes 'TrainIds' values
    if args.convert:

        # Create a list of filepaths to all original labels
        suffix_wo_png = LABEL_SUFFIX[:-4]
        label_filepaths = glob.glob(
            osp.join(vistas_path, f'*/v2.0/labels/*[!{suffix_wo_png}].png'))

        seg_choice = args.choice
        if seg_choice == 'cityscapes':
            if args.nproc > 1:
                mmcv.track_parallel_progress(convert_cityscapes_trainids,
                                             label_filepaths, args.nproc)
            else:
                mmcv.track_progress(convert_cityscapes_trainids,
                                    label_filepaths)
        else:
            raise ValueError

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_vistas_directory(out_dir, args.train_on_val_and_test,
                                     args.symlink)


if __name__ == '__main__':
    main()
