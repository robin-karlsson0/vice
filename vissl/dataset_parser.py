from __future__ import annotations
from abc import ABC, abstractmethod
import glob
import os
from typing import List


class DatasetInterface():
    """
    """
    def __init__(self, dataset_parser: DatasetParser) -> None:
        self._dataset_parser = dataset_parser

    @property
    def dataset_parser(self) -> DatasetParser:
        return self._dataset_parser

    @dataset_parser.setter
    def dataset_parser(self, dataset_parser: DatasetParser) -> None:
        self._dataset_parser = dataset_parser

    def parse_imgs(self, arg_parse_strings: list = None) -> List:
        """
        """
        img_paths = self._dataset_parser.parse_imgs()
        return img_paths


class DatasetParser(ABC):
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path (str): Path to dataset root.
        """
        self.dataset_path = dataset_path
        self.default_parse_strings = None

    # @abstractmethod
    def parse_imgs(self, arg_parse_strings: list = None):
        """
        Args:
            arg_parse_strings (list): List of strings used by glob when
                                      searching for sample images.
        """

        parse_strings = []

        if self.default_parse_strings is not None:
            parse_strings += self.default_parse_strings

        if arg_parse_strings is not None:
            parse_strings += arg_parse_strings

        img_paths = []
        for parse_str in parse_strings:

            # Check that subdir exists
            subdir = parse_str.split('/')[0]
            if not os.path.isdir(os.path.join(self.dataset_path, subdir)):
                continue

            img_paths += glob.glob(os.path.join(self.dataset_path, parse_str))

        return img_paths


class DefaultDatasetParser(DatasetParser):
    """
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)


class A2D2DatasetParser(DatasetParser):
    """
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        self.default_parse_strings = [
            "camera_lidar/*/camera/*/*.png",
            "camera_lidar_semantic/*/camera/*/*.png"
        ]


class CityscapesDatasetParser(DatasetParser):
    """
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        self.default_parse_strings = ["leftImg8bit/*/*/*.png"]
