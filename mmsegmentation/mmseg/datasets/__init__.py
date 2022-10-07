# Copyright (c) OpenMMLab. All rights reserved.
from .a2d2_19cls import A2D2Dataset19Classes
from .a2d2_34cls import A2D2Dataset34Classes
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .coco_stuff_coarse import COCOStuffCoarseDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .gta5 import GTA5Dataset
from .hrf import HRFDataset
from .idd import IDDDataset
from .mapillary_vistas import MapillaryVistasDataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .stare import STAREDataset
from .voc import PascalVOCDataset

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'A2D2Dataset18Classes', 'A2D2Dataset34Classes',
    'GTA5Dataset', 'MapillaryVistasDataset', 'IDDDataset',
    'COCOStuffCoarseDataset'
]
