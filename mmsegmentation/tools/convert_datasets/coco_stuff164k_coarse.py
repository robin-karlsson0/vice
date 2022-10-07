import argparse
import os.path as osp
import shutil
from functools import partial
from glob import glob
from os import symlink

import mmcv
import numpy as np
from PIL import Image

COCO_LEN = 123287
LABEL_SUFFIX = '_labelTrainIds_27'

clsID_to_trID = {
    0: 9,
    1: 11,
    2: 11,
    3: 11,
    4: 11,
    5: 11,
    6: 11,
    7: 11,
    8: 11,
    9: 8,
    10: 8,
    11: 8,
    12: 8,
    13: 8,
    14: 8,
    15: 7,
    16: 7,
    17: 7,
    18: 7,
    19: 7,
    20: 7,
    21: 7,
    22: 7,
    23: 7,
    24: 7,
    25: 6,
    26: 6,
    27: 6,
    28: 6,
    29: 6,
    30: 6,
    31: 6,
    32: 6,
    33: 10,
    34: 10,
    35: 10,
    36: 10,
    37: 10,
    38: 10,
    39: 10,
    40: 10,
    41: 10,
    42: 10,
    43: 5,
    44: 5,
    45: 5,
    46: 5,
    47: 5,
    48: 5,
    49: 5,
    50: 5,
    51: 2,
    52: 2,
    53: 2,
    54: 2,
    55: 2,
    56: 2,
    57: 2,
    58: 2,
    59: 2,
    60: 2,
    61: 3,
    62: 3,
    63: 3,
    64: 3,
    65: 3,
    66: 3,
    67: 3,
    68: 3,
    69: 3,
    70: 3,
    71: 0,
    72: 0,
    73: 0,
    74: 0,
    75: 0,
    76: 0,
    77: 1,
    78: 1,
    79: 1,
    80: 1,
    81: 1,
    82: 1,
    83: 4,
    84: 4,
    85: 4,
    86: 4,
    87: 4,
    88: 4,
    89: 4,
    90: 4,
    91: 17,
    92: 17,
    93: 22,
    94: 20,
    95: 20,
    96: 22,
    97: 15,
    98: 25,
    99: 16,
    100: 13,
    101: 12,
    102: 12,
    103: 17,
    104: 17,
    105: 23,
    106: 15,
    107: 15,
    108: 17,
    109: 15,
    110: 21,
    111: 15,
    112: 25,
    113: 13,
    114: 13,
    115: 13,
    116: 13,
    117: 13,
    118: 22,
    119: 26,
    120: 14,
    121: 14,
    122: 15,
    123: 22,
    124: 21,
    125: 21,
    126: 24,
    127: 20,
    128: 22,
    129: 15,
    130: 17,
    131: 16,
    132: 15,
    133: 22,
    134: 24,
    135: 21,
    136: 17,
    137: 25,
    138: 16,
    139: 21,
    140: 17,
    141: 22,
    142: 16,
    143: 21,
    144: 21,
    145: 25,
    146: 21,
    147: 26,
    148: 21,
    149: 24,
    150: 20,
    151: 17,
    152: 14,
    153: 21,
    154: 26,
    155: 15,
    156: 23,
    157: 20,
    158: 21,
    159: 24,
    160: 15,
    161: 24,
    162: 22,
    163: 25,
    164: 15,
    165: 20,
    166: 17,
    167: 17,
    168: 22,
    169: 14,
    170: 18,
    171: 18,
    172: 18,
    173: 18,
    174: 18,
    175: 18,
    176: 18,
    177: 26,
    178: 26,
    179: 19,
    180: 19,
    181: 24,
    255: 255
}


def convert_to_trainID(maskpath, out_mask_dir, is_train):
    mask = np.array(Image.open(maskpath))
    mask_copy = mask.copy()
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = osp.join(
        out_mask_dir, 'train2017',
        osp.basename(maskpath).split('.')[0] + LABEL_SUFFIX +
        '.png') if is_train else osp.join(
            out_mask_dir, 'val2017',
            osp.basename(maskpath).split('.')[0] + LABEL_SUFFIX + '.png')
    Image.fromarray(mask_copy).save(seg_filename, 'PNG')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert COCO Stuff 164k annotations to mmsegmentation \
                     format')
    parser.add_argument('coco_path', help='coco stuff path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
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
    args = parser.parse_args()
    return args


def main():
    """
    1. Arrange dataset directory as specified in the MMSegmentation docs.
       Ref: https://mmsegmentation.readthedocs.io/en/latest/
            dataset_prepare.html#coco-stuff-164k

        coco_stuff164k/
            annotations/ <-- Dense segmentation labels
                train2017/
                    .png
                val2017/
                    .png
            images/
                train2017/
                    .jpg
                val2017/
                    .jpg

    2. Download the 'curated image' idx lists and extract in dataset directory.
       Ref: https://www.robots.ox.ac.uk/~xuji/datasets/
            COCOStuff164kCurated.tar.gz

       Used list: Coco164kFull_Stuff_Coarse_7.txt

        coco_stuff164k/
            ...
            curated/
                train2017/
                    .txt
                val2017/
                    .txt
            ...

    3. Run script to 1) convert labels and 2) restructure directories.

        $ python tools/convert_datasets/coco_stuff164k_coarse.py
            path/to/dataset
            --nproc 8
    """
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc

    out_dir = args.out_dir or coco_path
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train2017'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'val2017'))

    if out_dir != coco_path:
        shutil.copytree(osp.join(coco_path, 'images'), out_img_dir)

    train_list = glob(osp.join(coco_path, 'annotations', 'train2017', '*.png'))
    train_list = [file for file in train_list if LABEL_SUFFIX not in file]
    test_list = glob(osp.join(coco_path, 'annotations', 'val2017', '*.png'))
    test_list = [file for file in test_list if LABEL_SUFFIX not in file]

    if args.convert:
        if args.nproc > 1:
            mmcv.track_parallel_progress(
                partial(
                    convert_to_trainID,
                    out_mask_dir=out_mask_dir,
                    is_train=True),
                train_list,
                nproc=nproc)
            mmcv.track_parallel_progress(
                partial(
                    convert_to_trainID,
                    out_mask_dir=out_mask_dir,
                    is_train=False),
                test_list,
                nproc=nproc)
        else:
            mmcv.track_progress(
                partial(
                    convert_to_trainID,
                    out_mask_dir=out_mask_dir,
                    is_train=True), train_list)
            mmcv.track_progress(
                partial(
                    convert_to_trainID,
                    out_mask_dir=out_mask_dir,
                    is_train=False), test_list)

    if args.restruct:

        mmcv.mkdir_or_exist(osp.join(out_dir, 'images_coarse/train2017'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations_coarse/train2017'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'images_coarse/val2017'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations_coarse/val2017'))

        train_sample_idx_file = osp.join(
            coco_path, 'curated/train2017/Coco164kFull_Stuff_Coarse_7.txt')
        val_sample_idx_file = osp.join(
            coco_path, 'curated/val2017/Coco164kFull_Stuff_Coarse_7.txt')

        with open(train_sample_idx_file) as f:
            train_sample_idxs = f.read().splitlines()
        with open(val_sample_idx_file) as f:
            val_sample_idxs = f.read().splitlines()

        for sample_idxs, split in [(train_sample_idxs, 'train2017'),
                                   (val_sample_idxs, 'val2017')]:

            for idx in sample_idxs:

                img_filepath = osp.join(out_dir, f'images/{split}/{idx}.jpg')
                ann_filepath = osp.join(
                    out_dir, f'annotations/{split}/{idx}{LABEL_SUFFIX}.png')

                img_linkpath = osp.join(out_dir,
                                        f'images_coarse/{split}/{idx}.jpg')
                ann_linkpath = osp.join(
                    out_dir,
                    f'annotations_coarse/{split}/{idx}{LABEL_SUFFIX}.png')

                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

    print('Done!')


if __name__ == '__main__':
    main()
