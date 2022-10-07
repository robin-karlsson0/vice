import glob
import os


def waymo_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Waymo Open dataset.
    
    Ref: https://waymo.com/open/about/

    Dataset structure (top level corrspond to 'root directory'):
    waymo_open/
        training/
            images/
                front/
                    0/
                        .png
                    ...
                ...

        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'Domain adaptation'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/domain_adaptation/*/images/*/*/*.png")
        dataset_img_paths += img_paths

    # Parse 'Domain adaptation unlabeled'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/domain_adaptation/*/unlabeled/images/*/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
