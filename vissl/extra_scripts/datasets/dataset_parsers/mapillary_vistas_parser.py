import glob
import os


def mapillary_vistas_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Mapillary Vistas
    dataset.

    Ref: https://www.mapillary.com/dataset/vistas

    Dataset structure (top level corrspond to 'root directory'):
    mapillary_vistas/
        testing/
            images/
                *.jpg
        training/
            images/
                *.jpg
        validation/
            images/
                *.jpg

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'testing'
    if os.path.isdir(os.path.join(root_path, 'testing')):
        img_paths = glob.glob(f"{root_path}/testing/images/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'training'
    if os.path.isdir(os.path.join(root_path, 'training')):
        img_paths = glob.glob(
            f"{root_path}/training/images/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'validation'
    if os.path.isdir(os.path.join(root_path, 'validation')):
        img_paths = glob.glob(
            f"{root_path}/validation/images/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
