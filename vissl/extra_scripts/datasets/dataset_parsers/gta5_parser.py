import glob
import os


def gta5_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the GTA 5 dataset.

    Ref: https://download.visinf.tu-darmstadt.de/data/from_games/

    Dataset structure (top level corrspond to 'root directory'):
    gta5/
        images/
            .png
            ...
        labels/
            .png
            ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images' (unannotated raw data)
    if os.path.isdir(os.path.join(root_path, 'images')):
        img_paths = glob.glob(f"{root_path}/images/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
