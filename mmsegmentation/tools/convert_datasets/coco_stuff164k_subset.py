import argparse
import glob
import os

import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Absolute path to dataset root directory.')
    parser.add_argument(
        'subset_ratio',
        type=float,
        help='Number between 0 and 1 specifying size of subset.')
    parser.add_argument(
        'mmseg_data_dir',
        type=str,
        help="Absolute path to MMSegmentation's data directory.")
    parser.add_argument('--seed', type=int, default=14)

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    subset_ratio = args.subset_ratio
    root = args.mmseg_data_dir
    seed = args.seed

    np.random.seed(seed)

    ##############################
    #  Make directory structure
    ##############################
    name = f'coco_stuff164k_coarse_{int(subset_ratio*100)}/'
    new_dataset_dir = os.path.join(root, name)
    os.makedirs(os.path.join(new_dataset_dir, 'images_coarse', 'train2017'))
    os.makedirs(os.path.join(new_dataset_dir, 'images_coarse', 'val2017'))
    os.makedirs(
        os.path.join(new_dataset_dir, 'annotations_coarse', 'train2017'))
    os.makedirs(os.path.join(new_dataset_dir, 'annotations_coarse', 'val2017'))

    #######################
    #  Subsample dataset
    #######################
    img_paths = glob.glob(
        os.path.join(dataset_dir, 'images_coarse', 'train2017', '*.jpg'))
    n_imgs = len(img_paths)
    print(f'Found {n_imgs} samples')

    n_subimgs = int(subset_ratio * n_imgs)
    img_paths = np.random.choice(img_paths, n_subimgs, replace=False)
    img_paths = img_paths.tolist()
    print(f'Subset ratio {subset_ratio} ==> {n_subimgs} samples')

    # Add all validation samples
    val_img_paths = glob.glob(
        os.path.join(dataset_dir, 'images_coarse', 'val2017', '*.jpg'))
    img_paths = img_paths + val_img_paths

    ann_paths = [
        img_path.replace('images_coarse', 'annotations_coarse').replace(
            '.jpg', '_labelTrainIds_27.png') for img_path in img_paths
    ]

    ###########################
    #  Create symbolic files
    ###########################

    for img_path in img_paths:
        new_img_path = img_path.replace(dataset_dir, new_dataset_dir)
        os.symlink(img_path, new_img_path)

    for ann_path in ann_paths:
        new_ann_path = ann_path.replace(dataset_dir, new_dataset_dir)
        os.symlink(ann_path, new_ann_path)
