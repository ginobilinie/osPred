import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from utils.utils import AverageMeter


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_dataset, optimizer, criterion, logger, config, epoch, writer):
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    corrects, sum_t, sum_p = np.zeros(3), np.zeros(3), np.zeros(3)
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    print("train data size:", len(train_dataset))
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        age = data_dict['age']
        who = data_dict['who']
        data = data.cuda()
        age = age.cuda().reshape(len(age), 1)
        who = who.cuda()
        # run training
        with torch.cuda.amp.autocast():
            lam = np.random.beta(0.5, 0.5)
            out_tri, y1, y2 = model(data, age, who, lam, mixup_hidden=True)
            loss_tri = mixup_criterion(criterion, out_tri, y1, y2, lam)

        losses.update(loss_tri.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss_tri).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        corrects1, sum_t1, sum_p1 = np.zeros(3), np.zeros(3), np.zeros(3)
        index = torch.argmax(out_tri, dim=1).cpu().numpy()
        index2 = torch.argmax(who, dim=1).cpu().numpy()
        for j in range(len(data)):
            sum_p1[index[j]] += 1
            sum_t1[index2[j]] += 1
            if index[j] == index2[j]:
                corrects1[index[j]] += 1

        recall1, precision1 = np.zeros(3), np.zeros(3)
        for k in range(3):
            recall1[k] = corrects1[k] / sum_t1[k] if sum_t1[k] != 0 else 1
            precision1[k] = corrects1[k] / sum_p1[k] if sum_p1[k] != 0 else (1 if sum_t1[k] == 0 else 0)
        acc_val1 = corrects1.sum() / sum_t1.sum()

        for k in range(3):
            corrects[k] += corrects1[k]
            sum_p[k] += sum_p1[k]
            sum_t[k] += sum_t1[k]

        if idx % config.PRINT_FREQ == 0:  # 每5小轮打印一次
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Cross_Entropy Loss: {loss.val:.3f} ({loss.avg:.3f})\t' \
                  'Precision: {precision[0]:.5f}\\{precision[1]:.5f}\\{precision[2]:.5f}\t' \
                  'Recall: {recall[0]:.5f}\\{recall[1]:.5f}\\{recall[2]:.5f}\t' \
                  'Acc: {acc:.5f}'.format(epoch, idx, len(loader), loss=losses, precision=precision1, recall=recall1, acc=acc_val1)
            logger.info(msg)

    recall, precision = np.zeros(3), np.zeros(3)
    for k in range(3):
        recall[k] = corrects[k] / sum_t[k] if sum_t[k] != 0 else 1
        precision[k] = corrects[k] / sum_p[k] if sum_p[k] != 0 else (1 if sum_t[k] == 0 else 0)
    acc_val = corrects.sum() / sum_t.sum()

    logger.info('--------- Cross Entropy LOSS ----------')
    logger.info(f'Loss mean: {losses.avg}')
    logger.info('-------------- Precision --------------')
    logger.info(f'Precision mean: {precision[0]:.5f}\\{precision[1]:.5f}\\{precision[2]:.5f}')
    logger.info('--------------- Recall ----------------')
    logger.info(f'Recall mean: {recall[0]:.5f}\\{recall[1]:.5f}\\{recall[2]:.5f}')
    logger.info('---------------  scores ---------------')
    logger.info(f'Acc: {acc_val}')
    logger.info('--------------- ------- ---------------')

    writer.add_scalar(tag="loss/train", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="acc/train", scalar_value=acc_val, global_step=epoch)

    return acc_val


def inference(model, valid_dataset, criterion, logger, config, epoch, writer):
    model.eval()

    losses = AverageMeter()
    corrects, sum_t, sum_p = np.zeros(3), np.zeros(3), np.zeros(3)  # 预测正确数目；实际正样本数目；预测为正样本总数
    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        age = data_dict['age']
        who = data_dict['who']
        data = data.cuda()
        print(len(data))
        age = age.cuda().reshape(len(age), 1)
        who = who.cuda()
        # run training
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(data, age)
                out = out.squeeze()
                loss = criterion(out, who)
                losses.update(loss)

        index = torch.argmax(out, dim=1).cpu().numpy()
        index2 = torch.argmax(who, dim=1).cpu().numpy()
        for j in range(len(data)):
            sum_p[index[j]] += 1
            sum_t[index2[j]] += 1
            if index[j] == index2[j]:
                corrects[index[j]] += 1

    recall, precision = np.zeros(3), np.zeros(3)
    for k in range(3):
        recall[k] = corrects[k] / sum_t[k] if sum_t[k] != 0 else 1
        precision[k] = corrects[k] / sum_p[k] if sum_p[k] != 0 else (1 if sum_t[k] == 0 else 0)
    acc_val = corrects.sum() / sum_t.sum()


    writer.add_scalar(tag="loss/val", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="acc/val", scalar_value=acc_val, global_step=epoch)

    logger.info('------------- COX LOSS ----------------')
    logger.info(f'Loss mean: {losses.avg}')
    logger.info('---------------  scores ---------------')
    logger.info(f'Acc: {acc_val}')
    logger.info('--------------- ------- ---------------')
    perf = acc_val
    return perf

