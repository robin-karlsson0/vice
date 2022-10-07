import glob
import os


def cityscapes_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Cityscapes dataset.

    Ref: https://www.cityscapes-dataset.com/login/

    Dataset structure (top level corrspond to 'root directory'):
    cityscapes/
        leftImg8bit/
            train/
                aachen/
                    .png
                ...
            ...
        leftImg8bit_foggy/
            train/
                aachen/
                    .png
                ...
            ...
        leftImg8bit_rain/
            train/
                aachen/
                    .png
                ...
            ...
        
    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'leftImg8bit'
    if os.path.isdir(os.path.join(root_path, 'leftImg8bit')):
        img_paths = glob.glob(f"{root_path}/leftImg8bit/*/*/*.png")
        dataset_img_paths += img_paths

    # Parse 'leftImg8bit_foggy' (optional dataset)
    if os.path.isdir(os.path.join(root_path, 'leftImg8bit_foggy')):
        img_paths = glob.glob(
            f"{root_path}/leftImg8bit_foggy/*/*/*.png")
        dataset_img_paths += img_paths
    
    # Parse 'leftImg8bit_rain' (optional dataset)
    if os.path.isdir(os.path.join(root_path, 'leftImg8bit_rain')):
        img_paths = glob.glob(
            f"{root_path}/leftImg8bit_rain/*/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
