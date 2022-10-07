import glob
import os


def hdd_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Honda Driving
    Dataset (HDD).
    
    Ref: https://usa.honda-ri.com/hdd

    Dataset structure (top level corrspond to 'root directory'):
    honda_hdd/
        camera/
            201702271017/
                .jpg

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'camera'
    if os.path.isdir(os.path.join(root_path, 'camera')):
        img_paths = glob.glob(
            f"{root_path}/camera/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
