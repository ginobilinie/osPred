import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from core.config import config
from core.scheduler import PolyScheduler
from core.function import train, inference, train_os, inference_os
from core.loss import RMSELoss
from utils.utils import save_checkpoint, create_logger, setup_seed
from models.networks import Classifier, OSNet
from dataset.dataloader import get_dataset, get_dataset_BraTS19


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='experiments/TCClassification.pth', help='path for pretrained weights', type=str)
    args = parser.parse_args()
    return args


def main(args):
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    net_classification = Classifier  # use your network architecture here --> <file_name>.<class_name>
    devices = config.TRAIN.DEVICES
    model1 = net_classification(config.DATASET.input_channel, config.MODEL.ENCODER)
    model1 = nn.DataParallel(model1, devices).cuda()
    # load pretrained weights
    checkpoint = torch.load(args.checkpoint)
    model1.load_state_dict(checkpoint['state_dict'])

    net = OSNet  # use your network architecture here --> <file_name>.<class_name>
    model = net(config.DATASET.input_channel, model1)
    model = nn.DataParallel(model, devices).cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.99, nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)  # epoch
    criterion = RMSELoss()

    train_dataset = get_dataset('train', config.DATASET.SPLIT)
    valid_dataset = get_dataset_BraTS19('val')

    mseloss = 10000000
    best_acc = 0
    logger = create_logger('log', 'train.log')

    with open('core/config.py', 'r') as f:
        config_contents = f.read()
    logger.info('Config contents: \n%s', config_contents)

    writer = SummaryWriter(log_dir=config.TRAIN.logdir)

    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        train_os(model, train_dataset, optimizer, criterion, logger, config, epoch, writer)
        scheduler.step()
        loss, acc = inference_os(model, valid_dataset, criterion, logger, config, mseloss, best_acc, epoch, writer)

        if loss < mseloss:
            mseloss = loss
            best_model_loss = True
        else:
            best_model_loss = False

        if acc > best_acc:
            best_acc = acc
            best_model_acc = True
        else:
            best_model_acc = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'mseloss': loss,
            'optimizer': optimizer.state_dict(),
        }, best_model_loss, best_model_acc, 'experiments', filename='checkpoint.pth')
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
