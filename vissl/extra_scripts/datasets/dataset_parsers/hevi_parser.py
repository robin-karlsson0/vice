import glob
import os


def hevi_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Honda Egocentric
    View-Intersection (HEV-I) Dataset.
    
    Ref: https://usa.honda-ri.com/hevi

    Dataset structure (top level corrspond to 'root directory'):
    honda_hevi/
        can/
        images/
            201802061131000744/
                .jpg
        odom/
        train/
        val/
            .mp4
        videos/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images'
    if os.path.isdir(os.path.join(root_path, 'images')):
        img_paths = glob.glob(
            f"{root_path}/images/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
