#!/usr/bin/python
import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv

random.seed(14)

# Global variables for specifying label suffix according to class count
LABEL_SUFFIX = '_gtFine_labelcsTrainIds.png'


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


def restructure_idd_directory(idd_path,
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
        idd_path: Absolute path to the Mapillary Vistas 'vistas/' directory.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(idd_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(idd_path, 'ann_dir', 'test'))

    for split in ['train', 'val', 'test']:
        img_filepaths = glob.glob(f'{idd_path}/leftImg8bit/{split}/*/*.png')

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:
            img_subdir_idx = img_filepath.split('/')[-2]
            img_filename = img_filepath.split('/')[-1]
            img_idx = img_filename.split('_')[0]

            ann_filename = img_idx + LABEL_SUFFIX
            ann_filepath = f'{idd_path}/gtFine/{split}/{img_subdir_idx}/\
                             {ann_filename}'

            img_linkpath = f'{idd_path}/img_dir/{split}/{img_filename}'
            if split == 'test':
                ann_linkpath = None
            else:
                ann_linkpath = f'{idd_path}/ann_dir/{split}/{ann_filename}'

            if use_symlinks:
                # NOTE: Can only create new symlinks if no priors ones exists
                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    if split != 'test':
                        symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

            else:
                copyfile(img_filepath, img_linkpath)
                if split != 'test':
                    copyfile(ann_filepath, ann_linkpath)


def parse_args():
    parser = argparse.ArgumentParser(description='Restructure the IDD dataset')
    parser.add_argument(
        'idd_path',
        help='Absolute path to IDD_Segmentation root directory \
            (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
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
    """A script for making the IDD dataset compatible with mmsegmentation.

    Restructures the directory structure after generating segmentation label
    maps using the provided tool.
    Ref: https://github.com/AutoNUE/public-code

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/idd)
    Dataset split:

        Arranges samples into 'train', 'val', and 'test' splits according to
        predetermined directory structure

    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is usefull in the synthetic-to-real domain adaptation
    experiment setting.

    Add `--nproc N` for multiprocessing using N threads.

    Example usage:
        python tools/convert_datasets/idd.py
            abs_path/to/idd_segmentation
    """
    args = parse_args()
    idd_path = args.idd_path
    out_dir = args.out_dir if args.out_dir else idd_path
    mmcv.mkdir_or_exist(out_dir)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    restructure_idd_directory(out_dir, args.train_on_val_and_test,
                              args.symlink)


if __name__ == '__main__':
    main()
