import glob
import os


def argoverse_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the Argoverse dataset.

    Dataset structure (top level corrspond to 'root directory'):
    argoverse/
        argoverse-tracking/
            test/
                <character sequences>/
                    ring_front_center/
                        .jpg
                    ring_front_left/
                        .jpg
                    ring_front_right/
                        .jpg
                    ring_side_left/
                        .jpg
                    ring_side_right/
                        .jpg
                    ring_rear_left/
                        .jpg
                    ring_rear_right/
                        .jpg
                    stereo_front_left/
                        .jpg
                    stereo_front_right/ <-- Practically identical to 'left'
                        .jpg
            train1/
                ...
            train2/
                ...
            train3/
                ...
            train4/
                ...
            val/
                ...
        rectified_stereo_images_v1.1/
            test/
                <character sequences>/
                    stereo_front_left_rect/
                        .jpg
                    stereo_front_right_rect/ <-- Practically identical to 'left'
                        .jpg
            train/
            val/

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'argoverse-tracking'
    if os.path.isdir(os.path.join(root_path, 'argoverse-tracking')):
        img_paths = glob.glob(f"{root_path}/argoverse-tracking/*/*/ring_*/*.jpg")
        dataset_img_paths += img_paths
        img_paths = glob.glob(f"{root_path}/argoverse-tracking/*/*/stereo_front_left/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'rectified_stereo_images_v1.1'
    if os.path.isdir(os.path.join(root_path, 'rectified_stereo_images_v1.1')):
        img_paths = glob.glob(
            f"{root_path}/rectified_stereo_images_v1.1/*/*/stereo_front_left_rect/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
