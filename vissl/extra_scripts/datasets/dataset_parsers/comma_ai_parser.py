import glob
import os


def comma_ai_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Comma.ai Driving
    Dataset.

    Ref: https://research.comma.ai/

    Dataset structure (top level corrspond to 'root directory'):
    comma_ai_driving_dataset/
        camera/
            <date string>/
                <subdir idx>/
                    .jpg

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'camera'
    if os.path.isdir(os.path.join(root_path, 'camera')):
        img_paths = glob.glob(f"{root_path}/camera/*/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
