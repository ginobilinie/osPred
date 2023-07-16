import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from core.config import config
from core.loss import CrossEntropyLoss
from core.scheduler import PolyScheduler
from core.function import train, inference
from utils.utils import save_checkpoint, create_logger, setup_seed
from models.networks import Classifier
from dataset.dataloader import get_dataset


def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    net = Classifier  # use your network architecture here --> <file_name>.<class_name>
    devices = config.TRAIN.DEVICES
    model = net(config.DATASET.input_channel, config.MODEL.ENCODER)
    model = nn.DataParallel(model, devices).cuda()

    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.99, nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)  # epoch
    # deep supervision weights, normalize sum to 1
    criterion = CrossEntropyLoss

    train_dataset = get_dataset('train', config.DATASET.SPLIT)
    valid_dataset = get_dataset('val', config.DATASET.SPLIT)

    best_perf = 0.0
    logger = create_logger('log', 'train.log')
    with open('core/config.py', 'r') as f:
        config_contents = f.read()
    logger.info('Config contents: \n%s', config_contents)

    writer = SummaryWriter(log_dir=config.TRAIN.logdir)

    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        train(model, train_dataset, optimizer, criterion, logger, config, epoch, writer)
        scheduler.step()
        perf = inference(model, valid_dataset, criterion, logger, config, epoch, writer)

        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'Acc': perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, 'experiments', filename='checkpoint.pth')
    writer.close()


if __name__ == '__main__':
    main()
