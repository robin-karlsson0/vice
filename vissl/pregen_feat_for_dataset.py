from os import pipe
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint

########################################################################
#  Pretrained model files
#  https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md
########################################################################

# Load configuration files
#config = OmegaConf.load(
#    "configs/config/pretrain/swav/dense_swav_8node_resnet_test.yaml")
config = OmegaConf.load("./dense_swav_8node_resnet_test.yaml")
default_config = OmegaConf.load("vissl/config/defaults.yaml")
cfg = OmegaConf.merge(default_config, config)

# Process configuration file
cfg = AttrDict(cfg)
# Do not know how necessary this specification is?
#cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = "model_iteration460000.torch"

#####################################################################
# Extract features from several layers of the trunk
# MODEL:
#     FEATURE_EVAL_SETTINGS:
#     EVAL_MODE_ON: True
#     FREEZE_TRUNK_ONLY: True
#     EXTRACT_TRUNK_FEATURES_ONLY: True
#     SHOULD_FLATTEN_FEATS: False
#     LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
#         ["conv1", ["AvgPool2d", [[10, 10], 10, 4]]],
#         ["res2", ["AvgPool2d", [[16, 16], 8, 0]]],
#         ["res3", ["AvgPool2d", [[13, 13], 5, 0]]],
#         ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
#         ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
#         ["res5avg", ["Identity", []]],  <-- No pooling operation?
#    ]
# Output: vec = features[0].data.numpy() --> (1,2048,7,7)
######################################################################
'''
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False  # (1,2048,1,1) --> (1,2048)
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP = [
    ["res5", ["Identity", []]]  # (1,2048,7,7)
]
'''

########################################################
#  Extract features of the trunk output --> (1,2048)
#  MODEL:
#      FEATURE_EVAL_SETTINGS:
#          EVAL_MODE_ON: True
#          FREEZE_TRUNK_ONLY: True
#          EXTRACT_TRUNK_FEATURES_ONLY: True
#          SHOULD_FLATTEN_FEATS: False
# Output: vec = features[0].data.numpy() --> (1,2048)
########################################################
#cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
#cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
#cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
#cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False  # (1,2048,1,1) --> (1,2048)

##########################################################
#  Extract features of the model head output
#  MODEL:
#  FEATURE_EVAL_SETTINGS:
#    EVAL_MODE_ON: True
#    FREEZE_TRUNK_AND_HEAD: True
#    EVAL_TRUNK_AND_HEAD: True
# NOTE: Need explict eval() ?
# Output: vec = features[0][0].data.numpy() --> (1,128)
##########################################################
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD = True
cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD = True

# Initialize model
model = build_model(cfg.config.MODEL, cfg.config.OPTIMIZER)

# Load pretrained weights
state_dict = load_checkpoint('model_iteration300000.torch')
model.init_model_from_weights_params_file(cfg.config, state_dict)
model.eval()

#####################
#  Model inference
#####################

from PIL import Image
import torch
import torchvision.transforms as transforms

pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('a2d2_1_cars_back.png').convert("RGB")
w, h = img.size
x = pipeline(img)
x = x.unsqueeze(0)

# Input: tensor (1,3,224,224)
# Output: list [tensor_1, ...]
with torch.no_grad():
    features = model.forward(x)

print(features)
vecs = features[0][0].data
print(vecs.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(vecs)

vecs_transf = pca.transform(vecs)

feat_map_transf = np.reshape(vecs_transf, (h, w, -1))

a = feat_map_transf
a = (a - np.min(a)) / (np.max(a) - np.min(a))
a *= 255
a = a.astype(int)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.imshow(a, alpha=0.75)
plt.show()

# torch.sqrt(torch.sum(a[0]**2))
# tensor(1.0000, grad_fn=<SqrtBackward>)
# torch.sqrt(torch.sum(b[0,0]**2))
# tensor(1.0000, grad_fn=<SqrtBackward>)
#feat_map = torch.reshape(vecs, (h, w, -1)).numpy()

#np.savetxt("feat_map.npy", feat_map)
