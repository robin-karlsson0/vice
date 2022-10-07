import glob
import os


def bdd100k_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the BDD100k dataset.

    Dataset structure (top level corrspond to 'root directory'):
    bdd100k/
        images/
            100k/
                test/
                    .jpg
                train/
                    .jpg
                val/
                    .jpg
            10k/
                test/
                    .jpg
                train/
                    .jpg
                val/
                    .jpg
        labels/
            ...
        seg/
            ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images'
    if os.path.isdir(os.path.join(root_path, 'images')):
        img_paths = glob.glob(f"{root_path}/images/*/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
