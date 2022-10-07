import argparse
import glob
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv


def create_split_dir(img_filepaths,
                     ann_filepaths,
                     split,
                     root_path,
                     subset_name,
                     use_symlinks=True):
    """Creates dataset split directory from given file lists using symbolic
    links or copying files.

    Args:
        img_filepaths: List of filepaths as strings.
        ann_filepaths:
        split: String denoting split (i.e. 'train', 'val', or 'test'),
        root_path: A2D2 dataset root directory (.../camera_lidar_semantic/)
        subset_name: Name of generated subset (i.e. 'subset01')
        use_symlinks: Symbolically link existing files in the original A2D2
                      dataset directory. If false, files will be copied.

    Raises:
        FileExistError: In case of pre-existing files when trying to create new
                        symbolic links.
    """
    assert split in ['train', 'val', 'test']

    subset_path = osp.join(root_path, subset_name)

    for img_filepath, ann_filepath in zip(img_filepaths, ann_filepaths):
        # Partions string: [generic/path/to/file] [/] [filename]
        img_filename = img_filepath.rpartition('/')[2]
        ann_filename = ann_filepath.rpartition('/')[2]

        img_link_path = osp.join(subset_path, 'leftImg8bit', split,
                                 img_filename)
        ann_link_path = osp.join(subset_path, 'gtFine', split, ann_filename)

        if use_symlinks:
            # NOTE: Can only create new symlinks if no priors ones exists
            try:
                symlink(img_filepath, img_link_path)
            except FileExistsError:
                pass
            try:
                symlink(ann_filepath, ann_link_path)
            except FileExistsError:
                pass

        else:
            copyfile(img_filepath, img_link_path)
            copyfile(ann_filepath, ann_link_path)


def sampled_cityscapes_dataset(cityscapes_path,
                               num_samples,
                               subset_name,
                               use_symlinks=True):
    """Creates a new directory structure and link existing files into it.

    cityscapes
    └── subset
        └── leftImg8bit
        │   ├── train
        │   │   ├── xxx{img_suffix}
        |   |   ...
        │   ├── val
        │   │   ├── yyy{img_suffix}
        │   │   ...
        │   ...
        └── gtFine
            ├── train
            │   ├── xxx{seg_map_suffix}
            |   ...
            ├── val
            |   ├── yyy{seg_map_suffix}
            |   ...
            ...
    """
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'leftImg8bit', 'train'))
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'leftImg8bit', 'val'))
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'leftImg8bit', 'test'))
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'gtFine', 'train'))
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'gtFine', 'val'))
    mmcv.mkdir_or_exist(
        osp.join(cityscapes_path, subset_name, 'gtFine', 'test'))

    for split in ['train', 'val']:

        # Lists containing all images and labels to symlinked
        img_filepaths = sorted(
            glob.glob(
                osp.join(cityscapes_path, f'leftImg8bit/{split}/*/*.png')))

        ann_filepaths = sorted(
            glob.glob(
                osp.join(cityscapes_path,
                         f'gtFine/{split}/*/*_gtFine_labelTrainIds.png')))

        # Randomize order of (image, label) pairs
        pairs = list(zip(img_filepaths, ann_filepaths))
        random.shuffle(pairs)
        img_filepaths, ann_filepaths = zip(*pairs)

        # Extract subset
        if split == 'train':
            img_filepaths = img_filepaths[:num_samples]
            ann_filepaths = ann_filepaths[:num_samples]

        print(f'Split: {split}')
        print(f'    imgs: {len(img_filepaths)}')
        print(f'    anns: {len(ann_filepaths)}')

        create_split_dir(
            img_filepaths,
            ann_filepaths,
            split,
            cityscapes_path,
            subset_name,
            use_symlinks=use_symlinks)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to TrainIds')
    parser.add_argument(
        'cityscapes_path',
        help='Cityscapes absolute path (NOT the symbolically linked one!)')
    parser.add_argument(
        'num_samples', type=int, help='Number of random samples')
    parser.add_argument(
        'subset_name', type=str, help='Directory name for generated subset')
    parser.add_argument('--seed', type=int, default=14, help='Random seed')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '--no-symlink',
        dest='symlink',
        action='store_false',
        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """"""
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    num_samples = args.num_samples
    subset_name = args.subset_name
    use_symlinks = args.symlink
    seed = args.seed

    random.seed(seed)

    sampled_cityscapes_dataset(
        cityscapes_path, num_samples, subset_name, use_symlinks=use_symlinks)


if __name__ == '__main__':
    main()
