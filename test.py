import sys
import numpy as np
import torch
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix
from sklearn.metrics.ranking import roc_auc_score
import torch.nn as nn
from dataset import Waste_Data, Waste_Data_Finetune
from torchvision import transforms
import pandas as pd
import os
import random
import time
import pdb
import tensorboard_logger as tb_logger
from util import AverageMeter
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"


def computeAUC(dataGT, dataPRED):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    classCount = datanpGT.shape[1]
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    mean_auc = float(np.mean(np.array(outAUROC)))
    return outAUROC, round(mean_auc, 4)


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
    val_dataset = Waste_Data_Finetune(valfiles, label_csv, train_jpg_dir, train_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, pin_memory=True)
    random.seed()
    sample = next(iter(val_loader))

    model = resnet50()
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, sample[1].size()[1]),
                             nn.LogSoftmax(dim=1))
    model = nn.DataParallel(model)

    ckpt = torch.load('200_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model = model.cuda()
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

    labels = outGT.cpu().numpy().argmax(axis=1)
    preds = outPRED.cpu().numpy().argmax(axis=1)
    matrix = confusion_matrix(labels, preds)
    overallAccuracy = np.trace(matrix) / np.sum(matrix)
    print(f'All Test AUC is {auc_test}, Mean_AUC IS {mean_auc_test}, the overall Accuracy is {overallAccuracy}')
    # computeAUC(outGT, outPRED)
    pdb.set_trace()


if __name__ == '__main__':
    main()
