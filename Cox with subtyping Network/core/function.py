import torch
from torch.utils.data.dataloader import DataLoader

from utils.utils import AverageMeter


def train_os(model, train_dataset, optimizer, criterion, logger, config, epoch, writer):
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    outs, ostimes = [], []
    print("train data size:", len(train_dataset))
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        age = data_dict['age']
        ostime = data_dict['ostime']

        data = data.cuda()
        ostime = ostime.cuda()
        age = age.cuda().reshape(len(age), 1)

        # run training
        with torch.cuda.amp.autocast():
            out = model(data, age)
            out = out.squeeze()
            loss = criterion(out, ostime)
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        outs.append(out)
        ostimes.append(ostime)

        if idx % config.PRINT_FREQ == 0:  # 每5小轮打印一次
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss: {loss.val:.3f} ({loss.avg:.3f})\t'.format(epoch, idx, len(loader), loss=losses)
            logger.info(msg)

    outs = torch.cat(outs)
    ostimes = torch.cat(ostimes)
    a = 0.0
    for (i, j) in zip(ostimes, outs):
        if i <= 365 and j <= 365:
            a += 1
        else:
            if 365 < i <= 730 and 365 < j <= 730:
                a += 1
            else:
                if i > 730 and j > 730:
                    a += 1
    acc = a / float(len(outs))

    logger.info('--------------- L2 LOSS ---------------')
    logger.info(f'Loss mean: {losses.avg}')
    logger.info('----------------- ACC -----------------')
    logger.info(f'acc: {acc}')
    logger.info('--------------- ------- ---------------')

    writer.add_scalar(tag="loss/train", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="acc/train", scalar_value=acc, global_step=epoch)

    return losses.avg


def inference_os(model, valid_dataset, criterion, logger, config, best_perf, best_perf2, epoch, writer):
    model.eval()

    losses = AverageMeter()
    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    outs = []
    ostimes = []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        age = data_dict['age'].float()
        ostime = data_dict['ostime']

        data = data.cuda()
        ostime = ostime.cuda()
        age = age.cuda().reshape(len(age), 1)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(data, age)
                out = out.squeeze()
                loss = criterion(out, ostime)
                losses.update(loss)
        outs.append(out)
        ostimes.append(ostime)

        msg = 'val\t\n' \
              'Epoch: [{0}][{1}/{2}]\t' \
              'Loss: {loss.val:.3f} ({loss.avg:.3f})\t\n'.format(epoch, idx, len(loader), loss=losses)
        logger.info(msg)

    outs = torch.cat(outs)
    ostimes = torch.cat(ostimes)
    a = 0.0
    for (i, j) in zip(ostimes, outs):
        if i <= 365 and j <= 365:
            a += 1
        else:
            if 365 < i <= 730 and 365 < j <= 730:
                a += 1
            else:
                if i > 730 and j > 730:
                    a += 1
    acc = a / float(len(outs))

    writer.add_scalar(tag="loss/val", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="acc/val", scalar_value=acc, global_step=epoch)

    logger.info('--------------- scores ----------------')
    logger.info(f'Loss mean: {losses.avg}')
    logger.info(f'acc: {acc}')
    logger.info('------------- best scores -------------')
    logger.info(f'min_loss: {best_perf}')
    logger.info(f'max_acc: {best_perf2}')
    logger.info('--------------- ------- ---------------')
    perf = losses.avg
    return perf, acc

