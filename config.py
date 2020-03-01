# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 10:18   xin      1.0         None
'''


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# Alias for easy usage
cfg = _C

_C.MODEL = CN()
_C.MODEL.NAME = "baseline"
_C.MODEL.N_CHANNEL = 3
_C.MODEL.N_CLASS = 10
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = '0d'
_C.MODEL.BACKBONE = 'wrn40_4'
_C.MODEL.DROPOUT = 0
_C.MODEL.USE_NONLOCAL = False
_C.MODEL.USE_SCSE = False
_C.MODEL.USE_ATTENTION = False



_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2


_C.DATASETS = CN()
_C.DATASETS.NAMES = ('minist')# Root PATH to the dataset
_C.DATASETS.DATA_PATH = r'/usr/demo/common_data'

_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.IMAGE_FOLDER = r'train_rs_imgs'



_C.DATASETS.VAL = CN()
_C.DATASETS.VAL.IMAGE_FOLDER = r'val_rs_imgs'


_C.INPUT = CN()

_C.INPUT.PIXEL_MEAN = [0.28604059698879547, 0.28604059698879547, 0.28604059698879547]
_C.INPUT.PIXEL_STD = [0.3202489254311618, 0.3202489254311618, 0.3202489254311618]

_C.INPUT.RESIZE_TRAIN = (36, 36)
_C.INPUT.SIZE_TRAIN = (32, 32)
_C.INPUT.RESIZE_TEST = (36, 36)
_C.INPUT.SIZE_TEST = (32, 32)
_C.INPUT.PROB = 0.5 # random horizontal flip



# random erase
_C.INPUT.RANDOM_ERASE = CN()
_C.INPUT.RANDOM_ERASE.RE_PROB = 0.5
_C.INPUT.RANDOM_ERASE.RE_MAX_RATIO = 0.4

_C.INPUT.USE_MIX_UP = False
_C.INPUT.USE_CUT_MIX = True
_C.INPUT.USE_AUGMIX = False
_C.INPUT.USE_AUTOAUG = True
_C.INPUT.USE_RICAP = False

_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER_NAME = "Ranger"


_C.SOLVER.MAX_EPOCHS = 320

_C.SOLVER.BASE_LR = 4e-3
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.USE_WARMUP = False
_C.SOLVER.MIN_LR = 4e-5

_C.SOLVER.MOMENTUM = 0.9


_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [40, 70]

_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCH = 10
_C.SOLVER.WARMUP_BEGAIN_LR = 3e-6
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.START_SAVE_EPOCH = 250

_C.SOLVER.TENSORBOARD = CN()
_C.SOLVER.TENSORBOARD.USE = True
_C.SOLVER.TENSORBOARD.LOG_PERIOD = 20

_C.SOLVER.PER_BATCH = 128
_C.SOLVER.SYNCBN = False
_C.SOLVER.RESUME = False
_C.SOLVER.RESUME_CHECKPOINT = ''

_C.OUTPUT_DIR = CN()
_C.OUTPUT_DIR = r'/usr/demo/common_data/minist_outputs'
