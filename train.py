import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.resnet import resnet50
# A Simple Framework for Contrastive Learning of Visual Representations
from models.simCLR import simCLR
from models.LinearModel import LinearClassifierResNet
from sklearn.metrics.ranking import roc_auc_score
import torch.nn as nn
from dataset import Waste_Data, Waste_Data_Finetune
from torchvision import transforms
import pandas as pd
# from models.vanilla_resnet import *
import os
import random
import time
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

train_jpg_dir = "./imagesDataset"
trainfiles = list(pd.read_csv('train.csv')['Image'])
label_csv = './train.csv'

normalize = transforms.Normalize(mean=[0], std=[1])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(.2, 1.)),
    transforms.ToTensor(),
    normalize,
])
train_dataset = Waste_Data_Finetune(trainfiles, label_csv, train_jpg_dir, train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True)

sample = next(iter(train_loader))

print("Finished Preprocessing")
model = simCLR()
classifier = LinearClassifierResNet(6, sample[1].size()[1], 'avg', 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.99)
model = model.cuda()
classifier = classifier.cuda()
criterion = torch.nn.BCEWithLogitsLoss().cuda()
# train unsupervised
print("Start supervised training")
end = time.time()
for epoch in range(200):
    print('train epoch {}'.format(epoch))
    train_auc = train(epoch, train_loader, model, classifier, criterion, optimizer)
    print("==> testing...")
    auc, mean_auc, test_loss = validate_multilabel(val_loader, model, classifier, criterion)


def computeAUC(dataGT, dataPRED):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    classCount = datanpGT.size(1)
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
    mean_auc = float(np.mean(np.array(outAUROC)))
    return outAUROC, round(mean_auc, 4)


def train(epoch, train_loader, model, classifier, criterion, optimizer):
    model.train()
    classifier.train()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()

    for idx, (samples, target) in enumerate(train_loader):
        samples = samples.float()
        target = target.float()
        target = target.contiguous().cuda(async=True)
        outGT = torch.cat((outGT, target), 0)

        # ===================forward=====================
        feat = model(samples, 6)
        output = classifier(feat)
        outPRED = torch.cat((outPRED, output.data), 0)
        loss = criterion(output, target.float())
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print info
    if (idx + 1) % 1 == 0:
        print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t loss: {loss}")
        sys.stdout.flush()
    sys.stdout.flush()

    auc_train, mean_auc_train = computeAUC(outGT, outPRED)
    print('All Train AUC is {},  Mean_AUC IS {}'.format(auc_train, mean_auc_train))
    return mean_auc_train


def validate_multilabel(val_loader, model, classifier, criterion):
    model.eval()
    classifier.eval()
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for idx, (samples, target) in enumerate(val_loader):
            samples = samples.float()
            target = target.contiguous().cuda(async=True).float()
            outGT = torch.cat((outGT, target), 0)
            # compute output
            feat = model(samples, 6)
            feat = feat.detach()
            output = classifier(feat)
            outPRED = torch.cat((outPRED, output.data), 0)
            loss = criterion(output, target.float())

            if idx % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f}'.format(idx, len(val_loader), oss=loss))
    auc_test, mean_auc_test = computeAUC(outGT, outPRED)
    print('All Test AUC is {}, Mean_AUC IS {}'.format(auc_test, mean_auc_test))
    return auc_test, mean_auc_test
