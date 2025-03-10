from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


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
_C.MODEL.CONVNEXT = CN()
_C.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
_C.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
_C.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
_C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
_C.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]

# solver
_C.SOLVER.WEIGHT_DECAY_RATE= 0.95
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE="model"
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.BACKBONE_MULTIPLIER = 1.0

# VIT
_C.MODEL.VIT = CN()
_C.MODEL.VIT.INPUT_SIZE = 1024
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.DROP_PATH_RATE =0.1
_C.MODEL.VIT.SCALE_FACTORS = [4.0, 2.0, 1.0, 0.5]
_C.MODEL.VIT.OUT_FEATURE = "last_feat"

_C.MODEL.BACKBONE.ADAPTER = CN()
_C.MODEL.BACKBONE.ADAPTER.MODE = "ft"
_C.MODEL.BACKBONE.ADAPTER.RATIO = 32
_C.MODEL.BACKBONE.ADAPTER.FFN_NUM=64
_C.MODEL.BACKBONE.ADAPTER.FFN_OPTION="parallel"
_C.MODEL.BACKBONE.ADAPTER.FFN_ADAPTER_LAYERNORM_OPTION="none"
_C.MODEL.BACKBONE.ADAPTER.FFN_ADAPTER_INIT_OPTION="lora"
_C.MODEL.BACKBONE.ADAPTER.FFN_ADAPTER_SCALAR=0.1


# ---------------------------------------------------------------------------- #
# Meta_arch
# ---------------------------------------------------------------------------- #
# FCOS
"""
Add config for FCOS.
"""
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"
_C.MODEL.FCOS.USE_SCALE = True
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"
_C.MODEL.FCOS.THRESH_WITH_CTR = False
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False
_C.MODEL.FCOS.YIELD_BOX_FEATURES = False

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
_C.MODEL.SparseRCNN.NUM_CLS = 1
_C.MODEL.SparseRCNN.NUM_REG = 3
_C.MODEL.SparseRCNN.NUM_HEADS = 6
_C.MODEL.SparseRCNN.HIDDEN_DIM = 256

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
_C.MODEL.DiffusionDet.NUM_CLS = 1
_C.MODEL.DiffusionDet.NUM_REG = 3
_C.MODEL.DiffusionDet.NUM_HEADS = 6
_C.MODEL.DiffusionDet.HIDDEN_DIM = 256

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

# Dynamic Conv.
_C.MODEL.DynamicConv = CN()
_C.MODEL.DynamicConv.NUM_DYNAMIC = 2
_C.MODEL.DynamicConv.DIM_DYNAMIC = 64
_C.MODEL.DynamicConv.HIDDEN_DIM = 256

# TTA.
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
_C.TEST.AUG.CVPODS_TTA = True
_C.TEST.AUG.SCALE_FILTER = True
_C.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                [64, 10000], [64, 10000],
                                [64, 10000], [0, 10000],
                                [0, 10000], [0, 256],
                                [0, 256], [0, 192],
                                [0, 192], [0, 96],
                                [0, 10000])
# Model EMA
_C.MODEL_EMA = type(_C)()
_C.MODEL_EMA.ENABLED = False
_C.MODEL_EMA.DECAY = 0.999
# use the same as MODEL.DEVICE when empty
_C.MODEL_EMA.DEVICE = ""
# When True, loading the ema weight to the model when eval_only=True in build_model()
_C.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = False
# when True, use YOLOX EMA: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/ema.py#L22
_C.MODEL_EMA.YOLOX = False

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
 # TEST on corruptions
_C.TEST.NOISE_ALL = False