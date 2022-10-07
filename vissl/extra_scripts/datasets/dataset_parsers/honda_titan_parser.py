import glob
import os


def honda_titan_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Honda Trajectory
    Inference using Targeted Action priors Network (TITAN) dataset.
    
    Ref: https://usa.honda-ri.com/titan

    Dataset structure (top level corrspond to 'root directory'):
    honda_titan_dataset/
        dataset/
            images_anonymized/
                clip_1/
                    images/
                        .png
                ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/dataset/images_anonymized/*/images/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
