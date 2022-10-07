import glob
import os


def cadc_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Canadian Adverse
    Driving Conditions Dataset (CADC).

    Ref: https://scale.com/open-datasets/waterloo

    Dataset structure (top level corrspond to 'root directory'):
    cadc/
        2018_03_06/
            0001/
                labeled/
                    image_00/
                        data/
                            .png
                    image_01/
                        .png
                    ...
                    lidar_points/
                    ...
            0002/
            ...
            calib/
        ...
        
    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'cadc'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/*/0*/labeled/image_*/data/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
