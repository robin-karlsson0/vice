import glob
import os


def global_road_damage_detection_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Global Road Damage
    Detection dataset.
    
    Ref: https://github.com/sekilab/RoadDamageDetector

    Dataset structure (top level corrspond to 'root directory'):
    global_road_damage_detection/
        train/
            Japan/
                images/
                    *.jpg
                annotations/
            ...
        test1/
            Japan/
                images/
                    *.jpg
            ...
        test2/
            ...

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/*/*/images/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
