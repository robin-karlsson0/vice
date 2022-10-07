import glob
import os


def pandaset_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the PandaSet dataset.

    Ref: https://scale.com/resources/download/pandaset

    Dataset structure (top level corrspond to 'root directory'):
    pandaset/
        001/
            camera/
                back_camera/
                    .jpg
                    ...
                front_camera/
                    .jpg
                    ...
                front_left_camera/
                    .jpg
                    ...
                front_right_camera/
                    .jpg
                    ...
                left_camera/
                    .jpg
                    ...
                right_camera/
                    .jpg
                    ...
            ...
        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'camera_lidar' (unannotated raw data)
    if os.path.isdir(root_path):
        img_paths = glob.glob(f"{root_path}/*/camera/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
