import glob
import os


def kitti_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the KITTI dataset.

    Ref: http://www.cvlibs.net/datasets/kitti/raw_data.php

    Reads the first of each RGB stereo image pair.

    Dataset structure (top level corrspond to 'root directory'):
    kitti/
        2011_09_26/
            2011_09_26_drive_0002_sync/
                image_00/ <-- Grayscale stereo 1/2
                    data/
                        .png
                image_01/ <-- Grayscale stereo 2/2
                    data/
                        .png
                image_02/ <-- RGB stereo 1/2
                    data/
                        .png
                image_03/ <-- RGB stereo 2/2
                    data/
                        .png
                oxts/
                velodyne_points/
            2011_09_26_drive_0005_sync/
            ...
        2011_09_28/
            ...
        2011_09_29/
            ...
        2011_09_30/
            ...
        2011_10_03/
            ...
        object_2d/
            testing/
                image_2/
                    .png
                prev_2/
                    .png
            training/
                image_2/
                    .png
                prev_2/
                    .png

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'raw data' directories
    if os.path.isdir(os.path.join(root_path, '2011_09_26')):
        img_paths = glob.glob(f"{root_path}/2011_09_26/*/image_02/data/*.png")
        dataset_img_paths += img_paths
    if os.path.isdir(os.path.join(root_path, '2011_09_28')):
        img_paths = glob.glob(f"{root_path}/2011_09_28/*/image_02/data/*.png")
        dataset_img_paths += img_paths
    if os.path.isdir(os.path.join(root_path, '2011_09_29')):
        img_paths = glob.glob(f"{root_path}/2011_09_29/*/image_02/data/*.png")
        dataset_img_paths += img_paths
    if os.path.isdir(os.path.join(root_path, '2011_09_30')):
        img_paths = glob.glob(f"{root_path}/2011_09_30/*/image_02/data/*.png")
        dataset_img_paths += img_paths
    if os.path.isdir(os.path.join(root_path, '2011_10_03')):
        img_paths = glob.glob(f"{root_path}/2011_10_03/*/image_02/data/*.png")
        dataset_img_paths += img_paths

    # Parse '2D object detection' directories
    if os.path.isdir(os.path.join(root_path, 'object_2d')):
        img_paths = glob.glob(f"{root_path}/object_2d/*/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
