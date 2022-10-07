import glob
import os


def tusimple_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the TuSimple datasets.

    Ref: https://github.com/TuSimple/tusimple-benchmark

    Dataset structure (top level corrspond to 'root directory'):
    tusimple/
        lane_detection/
            clips/
                0313-1/
                    10000/
                        .jpg
                    ...
                ...
        velocity_estimation/
            clips/
                1/
                    imgs/
                        001.jpg
                ...
            supp_img/
               .jpg 

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'lane_detection'
    if os.path.isdir(os.path.join(root_path, 'lane_detection')):
        img_paths = glob.glob(f"{root_path}/lane_detection/clips/*/*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'velocity_estimation'
    if os.path.isdir(os.path.join(root_path, 'velocity_estimation')):
        img_paths = glob.glob(
            f"{root_path}/velocity_estimation/clips/*/imgs/*.jpg")
        dataset_img_paths += img_paths
        img_paths = glob.glob(
            f"{root_path}/velocity_estimation/supp_img/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
