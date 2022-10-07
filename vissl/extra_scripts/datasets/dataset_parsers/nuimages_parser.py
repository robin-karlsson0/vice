import glob
import os


def nuimages_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the NuImages dataset.

    Ref: https://www.nuscenes.org/nuimages

    Dataset structure (top level corrspond to 'root directory'):
    nuimages/
        samples/
            CAM_BACK/
                .jpg
            CAM_BACK_LEFT/
                .jpg
            CAM_BACK_RIGHT/
                .jpg
            CAM_FRONT/
                .jpg
            CAM_FRONT_LEFT/
                .jpg
            CAM_FRONT_RIGHT/
                .jpg
        sweeps/
            CAM_BACK/
                .jpg
            CAM_BACK_LEFT/
                .jpg
            CAM_BACK_RIGHT/
                .jpg
            CAM_FRONT/
                .jpg
            CAM_FRONT_LEFT/
                .jpg
            CAM_FRONT_RIGHT/
                .jpg
        v1.0-mini/
            .json
        v1.0-test/
            .json
        v1.0-train/
            .json
        v1.0-val/
            .json

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'samples'
    if os.path.isdir(os.path.join(root_path, 'samples')):
        img_paths = glob.glob(f"{root_path}/samples/*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'sweeps'
    if os.path.isdir(os.path.join(root_path, 'sweeps')):
        img_paths = glob.glob(
            f"{root_path}/sweeps/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
