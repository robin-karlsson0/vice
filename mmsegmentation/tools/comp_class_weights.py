import argparse
import glob
import os.path as osp
from itertools import repeat
from multiprocessing import Pool

import mmcv
import numpy as np

EPS = 1e-14


def count_class_pixels(label_filepath, class_N):
    """Returns a list of pixels per class for a label."""
    label = mmcv.imread(label_filepath, flag='grayscale')

    class_idxs, counts = np.unique(label, return_counts=True)

    class_count = np.zeros(class_N, dtype=np.float64)
    for idx, class_idx in enumerate(class_idxs):
        # Skip counting ignored labels
        if class_idx == 255 or class_idx == -1:
            continue
        class_count[class_idx] = counts[idx].astype(np.float64)

    return class_count


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute class weight for balancing')
    parser.add_argument(
        'label_dir', help='Absolute path to training label directory')
    parser.add_argument(
        '--label-suffix',
        default='.png',
        type=str,
        help="Substring that distinguishes labels from other files. \
              Ex: '_trainIds.png'")
    parser.add_argument(
        '--classes', default=19, type=int, help='Number of classes in labels')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of process')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Example usage:

    python tools/comp_class_weights.py path/to/train/label_dir
    """

    args = parse_args()
    label_dir = args.label_dir
    label_suffix = args.label_suffix
    class_N = args.classes
    nproc = args.nproc

    label_filepaths = glob.glob(osp.join(label_dir, f'*{label_suffix}'))

    # Generate a list of list of class counts per label and sum
    p = Pool(nproc)
    res = p.starmap(count_class_pixels, zip(label_filepaths, repeat(class_N)))
    p.close()
    print(f'Processed {len(res)} labels')
    tot_class_count = np.zeros(class_N, dtype=np.float64)
    for class_count in res:
        tot_class_count += class_count

    if tot_class_count.any() == 0:
        raise Exception('Classes without elements found')

    # Compute class balancing weights
    w_classes = 1. / (np.log(tot_class_count))
    w_classes = class_N * w_classes / np.sum(w_classes)

    print('Computed class balancing weights')
    for idx, w_class in enumerate(w_classes):
        print(f'{str(idx).rjust(2)} | {w_class}')
