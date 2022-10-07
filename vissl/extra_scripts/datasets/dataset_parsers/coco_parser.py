import glob
import os


def coco_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the COCO dataset.

    Ref: https://cocodataset.org/#download

    Dataset structure (top level corrspond to 'root directory'):
    coco/
        train2017/
            .jpg
        val2017/
            .jpg
        test2017/
            .jpg
        unlabeled2017/
            .jpg
        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'train2017'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/train2017/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'unlabeled2017'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/unlabeled2017/*.jpg")
        dataset_img_paths += img_paths
    return dataset_img_paths
