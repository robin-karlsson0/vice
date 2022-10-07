import glob
import os


def ddad_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Dense Depth for
    Autonomous Driving (DDAD) dataset.

    Ref: https://github.com/TRI-ML/DDAD

    Dataset structure (top level corrspond to 'root directory'):
    ddad/
        ddad_train_val/
            000000/
                rgb/
                    CAMERA_01/
                        .png
                    ...
            ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'ddad_train_val'
    if os.path.isdir(os.path.join(root_path, 'ddad_train_val')):
        img_paths = glob.glob(f"{root_path}/ddad_train_val/*/rgb/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
