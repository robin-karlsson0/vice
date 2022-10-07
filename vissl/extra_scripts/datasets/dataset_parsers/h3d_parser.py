import glob
import os


def h3d_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Honda 3D dataset
    (H3D).
    
    Ref: https://usa.honda-ri.com/H3D

    Dataset structure (top level corrspond to 'root directory'):
    honda_h3d/
        camera_data/
            icra_benchmark_20200103_image_only/
                scenario_001/
                    *.png
                ...
            ...
        data/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'camera_data'
    if os.path.isdir(os.path.join(root_path, 'camera_data')):
        img_paths = glob.glob(
            f"{root_path}/camera_data/icra_benchmark_20200103_image_only/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
