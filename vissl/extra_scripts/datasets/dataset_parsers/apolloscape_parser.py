import glob
import os


def apolloscape_parser(root_path: str):
    """
    Returns a list of absolute image paths contained in the ApolloScape dataset.

    Dataset structure (top level corrspond to 'root directory'):
    apolloscape/
        car_instance/
            train/
                images/
                    .jpg
                ...
            test/
                images/
                    .jpg
        lane_segmentation/
            ColorImage_road02/
                ColorImage/
                    Record001/
                        Camera_5/
                            .jpg
                        ...
                    ...
            ColorImage_road03/
                ColorImage/
                    Record001/
                        Camera_5/
                            .jpg
                        ...
                    ....
        scene_parsing/
            ColorImage/
                Record006/
                    Camera_5/
                        .jpg
                    ...
                ...
        stereo/
            stereo_train_001/
                camera_5/
                    .jpg
                camera_6/
                    .jpg
            ...
        trajectory/
            asdt_sample_image/
                sample_image_1/
                    .jpg
                ...

    NOTE: One needs to remove whitespace in directory
          names (i.e. 'Camera 5' directories)

    Args:
        root_path (str): Root path of dataset.

    Returns:
        List of strings corresponding to absolute image paths.
    """
    dataset_img_paths = []

    # Parse 'car_instance'
    # Excludes 'ignore_mask' directories
    if os.path.isdir(os.path.join(root_path, 'car_instance')):
        img_paths = glob.glob(f"{root_path}/car_instance/*/images/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'lane_segmentation'
    if os.path.isdir(os.path.join(root_path, 'lane_segmentation')):
        img_paths = glob.glob(
            f"{root_path}/lane_segmentation/*/ColorImage/*/*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'scene_parsing'
    # 'Camera_5' and 'Camera_6' are stereo pairs ==> Use only 'Camera_5'
    if os.path.isdir(os.path.join(root_path, 'scene_parsing')):
        # 'ColorImage' dir
        img_paths = glob.glob(
            f"{root_path}/scene_parsing/ColorImage/*/Camera_5/*.jpg")
        dataset_img_paths += img_paths
        # 'road0*_ins' dir
        dir_name = r"Camera 5"  # For space in dir name
        img_paths = glob.glob(
            f"{root_path}/scene_parsing/*_ins/ColorImage/*/{dir_name}/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'stereo'
    if os.path.isdir(os.path.join(root_path, 'stereo')):
        img_paths = glob.glob(f"{root_path}/stereo/*/camera_*/*.jpg")
        dataset_img_paths += img_paths

    # Parse 'trajecoty'
    if os.path.isdir(os.path.join(root_path, 'trajectory')):
        img_paths = glob.glob(
            f"{root_path}/trajectory/asdt_sample_image/*/*.jpg")
        dataset_img_paths += img_paths

    return dataset_img_paths
