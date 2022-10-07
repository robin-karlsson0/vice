import glob
import os


def boxy_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Boxy dataset.
    
    Ref: https://boxy-dataset.com/boxy/

    Dataset structure (top level corrspond to 'root directory'):
    boxy/
        bluefox_2016-09-27-14-43-04_bag/
            .jpg
        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'boxy'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
