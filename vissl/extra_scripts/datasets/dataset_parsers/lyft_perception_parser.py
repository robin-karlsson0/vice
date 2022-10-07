import glob
import os


def lyft_perception_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Lyft Perception
    dataset.

    Ref: https://self-driving.lyft.com/level5/perception/

    Dataset structure (top level corrspond to 'root directory'):
    lyft_perception/
        test_data/
        test_images/
            .jpeg
        test_lidar/
        test_maps/
        train_data/
        train_images/
            .jpeg
        train_lidar/
        train_maps/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'test_images'
    if os.path.isdir(os.path.join(root_path, 'test_images')):
        img_paths = glob.glob(f"{root_path}/test_images/*.jpeg")
        dataset_img_paths += img_paths

    # Parse 'train_images'
    if os.path.isdir(os.path.join(root_path, 'train_images')):
        img_paths = glob.glob(
            f"{root_path}/train_images/*.jpeg")
        dataset_img_paths += img_paths

    return dataset_img_paths
