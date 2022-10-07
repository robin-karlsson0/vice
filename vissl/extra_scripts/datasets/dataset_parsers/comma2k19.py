import glob
import os


def comma2k19_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Comma2k19 Dataset.

    Ref: https://github.com/commaai/comma2k19

    Dataset structure (top level corrspond to 'root directory'):
    comma2k19/
        Chunk_1/
            b0c9d2329ad1606b_2018-07-27--06-03-57/
                3/
                    imgs/
                        .jpg
                ...
            ...
        Chunk_2/
        ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/Chunk_*/*/*/imgs/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
