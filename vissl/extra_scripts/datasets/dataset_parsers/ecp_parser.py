import glob
import os


def ecp_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the EuroCity Persons
    (ECP) dataset.

    Ref: https://eurocity-dataset.tudelft.nl/

    Dataset structure (top level corrspond to 'root directory'):
    ecp/
        day/
            img/
                train/
                    amsterdam/
                        .png
                    ...
                ...
            label/
                ...
        night/
            ...     

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'ecp'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/*/img/*/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
