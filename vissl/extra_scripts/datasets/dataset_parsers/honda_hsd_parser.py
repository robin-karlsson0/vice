import glob
import os


def honda_hsd_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the HSD Honda Scenes
    Dataset.
    
    Ref: https://usa.honda-ri.com/hsd

    Dataset structure (top level corrspond to 'root directory'):
    honda_hsd/
        FINALDATA/
            imgs/
                201803211358_2018-03-21/
                    .jpg

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/FINALDATA/imgs/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
