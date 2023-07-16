from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 6
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.RAWDATA = ''
config.DATASET = CN()
config.DATASET.ROOT = 'DATA/zhengdayi_pre_dead'
config.DATASET.SPLIT = 'splits_all.pkl'
config.DATASET.input_channel = 4

config.MODEL = CN()
config.MODEL.NAME = 'OSt'
config.MODEL.EXTRA = CN(new_allowed=True)
config.MODEL.INPUT_SIZE = [192, 192, 192]
config.MODEL.ENCODER = [16, 32, 64]

config.TRAIN = CN()
config.TRAIN.logdir = 'runs/OSPrediction'
config.TRAIN.LR = 1e-3
config.TRAIN.WEIGHT_DECAY = 1e-4
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.EPOCH = 101
config.TRAIN.DEVICES = [0, 1]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 16
