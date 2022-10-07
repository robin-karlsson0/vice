import glob
import os


def apollo_daoxianglake_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Apollo DaoxiangLake
    dataset.

    Dataset structure (top level corrspond to 'root directory'):
    apollo_daoxianglake/
        image_data/
            20190924124848/
                front/
                    .jpg
                left_back/
                    .jpg
                right_back/
                    .jpg
        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'image_data'
    if os.path.isdir(os.path.join(root_path, 'image_data')):
        img_paths = glob.glob(f"{root_path}/image_data/20190924124848/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
