import glob
import os


def nuscenes_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the NuScenes dataset.

    Ref: https://www.nuscenes.org/nuscenes

    Dataset structure (top level corrspond to 'root directory'):
    nuimages/
        maps/
        samples/
            CAM_BACK/
                *.jpg
            CAM_BACK_LEFT/
                *.jpg
            CAM_BACK_RIGHT/
                *.jpg
            CAM_FRONT/
                *.jpg
            CAM_FRONT_LEFT/
                *.jpg
            CAM_FRON_RIGHT/
                *.jpg
            LIDAR_TOP/
            ...
            RADAR_FRONT/
            ...
        sweeps/
            CAM_BACK/
                *.jpg
            CAM_BACK_LEFT/
                *.jpg
            CAM_BACK_RIGHT/
                *.jpg
            CAM_FRONT/
                *.jpg
            CAM_FRONT_LEFT/
                *.jpg
            CAM_FRON_RIGHT/
                *.jpg
            LIDAR_TOP/
            ...
            RADAR_FRONT/
            ...
        v1.0-mini/
        v1.0-test/
        v1.0-trainval/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'samples'
    if os.path.isdir(os.path.join(root_path, 'samples')):
        img_paths = glob.glob(f"{root_path}/samples/CAM_*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'sweeps'
    if os.path.isdir(os.path.join(root_path, 'sweeps')):
        img_paths = glob.glob(f"{root_path}/sweeps/CAM_*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
