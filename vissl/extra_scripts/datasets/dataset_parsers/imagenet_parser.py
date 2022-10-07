import glob
import os


def imagenet_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the ImageNet dataset.

    Ref: https://image-net.org/download-images.php

    Dataset structure (top level corrspond to 'root directory'):
    imagenet/
        train/
            0/
                .JPEG
            ...
        train3/
            0/
                .JPEG
            ...
        val/
            0/
                .JPEG
            ...
        test/
            0/
                .JPEG
            ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'train'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/*/*/*.JPEG")
        dataset_img_paths += img_paths

    return dataset_img_paths
