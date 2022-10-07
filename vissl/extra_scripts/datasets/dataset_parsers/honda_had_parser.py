import glob
import os


def honda_had_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the HRI Advice Dataset
    (HAD).
    
    Ref: https://usa.honda-ri.com/had

    Dataset structure (top level corrspond to 'root directory'):
    honda_research_institute_advice_dataset/
        img/
            train0001/
                .jpg

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'images'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/img/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
