from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffCoarseDataset(CustomDataset):
    """COCO-Stuff coarse dataset."""
    CLASSES = ('class1', 'class2', 'class3', 'class4', 'class5', 'class6',
               'class7', 'class8', 'class9', 'class10', 'class11', 'class12',
               'class13', 'class14', 'class15', 'class16', 'class17',
               'class18', 'class19', 'class20', 'class21', 'class22',
               'class23', 'class24', 'class25', 'class26', 'class27')

    PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(COCOStuffCoarseDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_labelTrainIds_27.png',
            **kwargs)
