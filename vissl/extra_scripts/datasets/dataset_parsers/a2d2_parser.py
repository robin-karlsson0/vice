import glob
import os


def a2d2_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the A2D2 dataset.

    Ref: https://www.a2d2.audi/a2d2/en/download.html

    Dataset structure (top level corrspond to 'root directory'):
    a2d2/
        camera_lidar/
            20180810_150607/
                camera/
                    cam_front_center/
                        .png
                    ...
            ...
        camera_lidar_semantic/
            20180807_145028/
                camera/
                    cam_front_center/
                        .png
                label/
                    cam_front_center/
                        .png
                ...
            ...
        camera_lidar_bboxes/
            <Same data as camera_lidar_semantic/>        

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'camera_lidar' (unannotated raw data)
    if os.path.isdir(os.path.join(root_path, 'camera_lidar')):
        img_paths = glob.glob(f"{root_path}/camera_lidar/*/camera/*/*.png")
        dataset_img_paths += img_paths

    # Parse 'camera_lidar_semantic'
    if os.path.isdir(os.path.join(root_path, 'camera_lidar_semantic')):
        img_paths = glob.glob(
            f"{root_path}/camera_lidar_semantic/*/camera/*/*.png")
        dataset_img_paths += img_paths

    return dataset_img_paths
