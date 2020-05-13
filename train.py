import sys
import numpy as np
import torch
from torchvision.models import resnet50
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from dataset import Waste_Data, Waste_Data_Finetune
from torchvision import transforms
import pandas as pd
import os
import random
import time
import pdb
import tensorboard_logger as tb_logger
from util import AverageMeter, adjust_learning_rate
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"


def main():
    train_jpg_dir = "./data"
    files = list(pd.read_csv('train.csv')['Image'])
    label_csv = './train.csv'

    normalize = transforms.Normalize(mean=[0], std=[1])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(1., 1.)),
        transforms.ToTensor(),
        normalize,
    ])
    random.seed(18)
    random.shuffle(files)
    cutoff = int(len(files) * 0.80)
    trainfiles, valfiles = files[:cutoff], files[cutoff:]
    print(f"Training on {len(trainfiles)} and Validating on {len(valfiles)}")
    train_dataset = Waste_Data_Finetune(trainfiles, label_csv, train_jpg_dir, train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, pin_memory=True)
    val_dataset = Waste_Data_Finetune(valfiles, label_csv, train_jpg_dir, train_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, pin_memory=True)
    random.seed()
    sample = next(iter(train_loader))

    print("Finished Preprocessing")
    model = resnet50(pretrain=True)
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, sample[1].size()[1]),
                             nn.LogSoftmax(dim=1))
    model = nn.DataParallel(model)
    lr = 0.003
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    model = model.cuda()
    criterion = nn.NLLLoss().cuda()
    # train unsupervised
    print("Start supervised training")
    tb_folder = "./tb"
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)
    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)
    for epoch in range(1, 201):
        adjust_learning_rate(epoch=epoch, learning_rate=lr, lr_decay_rate=0.1, lr_decay_epochs=[120, 160], optimizer=optimizer)
        print('train epoch {}'.format(epoch))
        train_auc, train_loss = train(epoch, train_loader, model, criterion, optimizer)
        print("==> testing...")
        val_auc, test_loss, accuracy = validate_multilabel(val_loader, model, criterion)
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"{epoch}_model.pth")
        logger.log_value('train_auc', train_auc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('val_auc', val_auc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('accuracy', accuracy, epoch)


def computeAUC(dataGT, dataPRED):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    classCount = datanpGT.shape[1]
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    mean_auc = float(np.mean(np.array(outAUROC)))
    return outAUROC, round(mean_auc, 4)


def train(epoch, train_loader, model, criterion, optimizer):
    model.train()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    losses = AverageMeter()

    for idx, (samples, target) in enumerate(train_loader):
        samples = samples.float()
        target = target.float()
        target = target.contiguous().cuda(non_blocking=True)
        outGT = torch.cat((outGT, target), 0)

        # ===================forward=====================
        output = model(samples)
        outPRED = torch.cat((outPRED, output.data), 0)
        loss = criterion(output, target.float())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================meters======================
        losses.update(loss.item(), samples.size(0))

        if idx % 10 == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t Loss {losses.val:.4f}({losses.avg:.4f})\t')
            sys.stdout.flush()

    auc_train, mean_auc_train = computeAUC(outGT, outPRED)
    print('All Train AUC is {},  Mean_AUC IS {}'.format(auc_train, mean_auc_train))
    return mean_auc_train, losses.avg


def validate_multilabel(val_loader, model, criterion):
    losses = AverageMeter()
    model.eval()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for idx, (samples, target) in enumerate(val_loader):
            samples = samples.float()
            target = target.contiguous().cuda(non_blocking=True).float()
            outGT = torch.cat((outGT, target), 0)
            # compute output
            output = model(samples)
            output = output.detach()
            outPRED = torch.cat((outPRED, output.data), 0)
            loss = criterion(output, target.float())
            losses.update(loss.item(), samples.size(0))
            if idx % 10 == 0:
                print(f'Test: [{idx}/{len(val_loader)}]\t Loss {losses.val:.4f}({losses.avg:.4f})')

    auc_test, mean_auc_test = computeAUC(outGT, outPRED)
    labels = outGT.cpu().numpy().argmax(axis=1)
    preds = outPRED.cpu().numpy().argmax(axis=1)
    matrix = confusion_matrix(labels, preds)
    overallAccuracy = np.trace(matrix) / np.sum(matrix)
    print(f'All Test AUC is {auc_test}, Mean_AUC IS {mean_auc_test}, the overall Accuracy is {overallAccuracy}')
    return mean_auc_test, losses.avg, overallAccuracy


if __name__ == '__main__':
    main()
