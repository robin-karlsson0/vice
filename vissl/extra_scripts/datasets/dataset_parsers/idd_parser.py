import glob
import os


def idd_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Waymo Open dataset.
    
    Ref: https://idd.insaan.iiit.ac.in/dataset/details/

    Dataset structure (top level corrspond to 'root directory'):
    idd/
        idd20kII/
            leftImg8bit/
                train/
                    201/
                        .jpg
                    ...
                val/
                test/
            gtFine/
        IDD_Detection/
            JPEGImages/
                frontFar/
                    BLR-2018-03-22_17-39-26_2_frontFar/
                       *.jpg 
                    ...
                ...
            Annotations/
        IDD_Segmentation/
            leftImg8bit/
                train/
                    0/
                        .jpg
                    ...
                val/
                test/
            gtFine/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'idd20kII'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(f"{root_path}/idd20kII/leftImg8bit/*/*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'IDD_Detection'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/IDD_Detection/JPEGImages/*/*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'IDD_Segmentation'
    if os.path.isdir(os.path.join(root_path)):
        img_paths = glob.glob(
            f"{root_path}/IDD_Segmentation/leftImg8bit/*/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
