from detectron2.config.defaults import _C
from detectron2.config import cfgNode as CN


# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #

# Swin
 # Swin Backbones
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
_C.MODEL.SWIN.USE_CHECKPOINT = False
_C.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

# Convnext config
_C.MODEL.CONVNEXT =CN()
_C.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
_C.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
_C.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
_C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
_C.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]
# solver
_C.SOLVER.WEIGHT_DECAY_RATE= 0.95

# VIT



# ---------------------------------------------------------------------------- #
# Meta_arch
# ---------------------------------------------------------------------------- #
# FCOS

# Deformable DETR


# Sparse-RCNN

"""
Add config for SparseRCNN.
"""
_C.MODEL.SparseRCNN = CN()
_C.MODEL.SparseRCNN.NUM_CLASSES = 80
_C.MODEL.SparseRCNN.NUM_PROPOSALS = 300

# RCNN Head.
_C.MODEL.SparseRCNN.NHEADS = 8
_C.MODEL.SparseRCNN.DROPOUT = 0.0
_C.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
_C.MODEL.SparseRCNN.ACTIVATION = 'relu'
_C.MODEL.SparseRCNN.HIDDEN_DIM = 256
_C.MODEL.SparseRCNN.NUM_CLS = 1
_C.MODEL.SparseRCNN.NUM_REG = 3
_C.MODEL.SparseRCNN.NUM_HEADS = 6

# Dynamic Conv.
_C.MODEL.SparseRCNN.NUM_DYNAMIC = 2
_C.MODEL.SparseRCNN.DIM_DYNAMIC = 64

# Loss.
_C.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
_C.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
_C.MODEL.SparseRCNN.L1_WEIGHT = 5.0
_C.MODEL.SparseRCNN.DEEP_SUPERVISION = True
_C.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

# Focal Loss.
_C.MODEL.SparseRCNN.USE_FOCAL = True
_C.MODEL.SparseRCNN.ALPHA = 0.25
_C.MODEL.SparseRCNN.GAMMA = 2.0
_C.MODEL.SparseRCNN.PRIOR_PROB = 0.01

# config파일에서 수정해야함
# Optimizer. 
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 1.0


# DiffusionDet
"""
Add config for DiffusionDet
"""
_C.MODEL.DiffusionDet = CN()
_C.MODEL.DiffusionDet.NUM_CLASSES = 80
_C.MODEL.DiffusionDet.NUM_PROPOSALS = 300

# RCNN Head.
_C.MODEL.DiffusionDet.NHEADS = 8
_C.MODEL.DiffusionDet.DROPOUT = 0.0
_C.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
_C.MODEL.DiffusionDet.ACTIVATION = 'relu'
_C.MODEL.DiffusionDet.HIDDEN_DIM = 256
_C.MODEL.DiffusionDet.NUM_CLS = 1
_C.MODEL.DiffusionDet.NUM_REG = 3
_C.MODEL.DiffusionDet.NUM_HEADS = 6

# Dynamic Conv.
_C.MODEL.DiffusionDet.NUM_DYNAMIC = 2
_C.MODEL.DiffusionDet.DIM_DYNAMIC = 64

# Loss.
_C.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
_C.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
_C.MODEL.DiffusionDet.L1_WEIGHT = 5.0
_C.MODEL.DiffusionDet.DEEP_SUPERVISION = True
_C.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

# Focal Loss.
_C.MODEL.DiffusionDet.USE_FOCAL = True
_C.MODEL.DiffusionDet.USE_FED_LOSS = False
_C.MODEL.DiffusionDet.ALPHA = 0.25
_C.MODEL.DiffusionDet.GAMMA = 2.0
_C.MODEL.DiffusionDet.PRIOR_PROB = 0.01

# Dynamic K
_C.MODEL.DiffusionDet.OTA_K = 5

# Diffusion
_C.MODEL.DiffusionDet.SNR_SCALE = 2.0
_C.MODEL.DiffusionDet.SAMPLE_STEP = 1

# Inference
_C.MODEL.DiffusionDet.USE_NMS = True



# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
 # TEST on corruptions
_C.TEST.NOISE_ALL = False 