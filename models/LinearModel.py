from __future__ import print_function

import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class LinearClassifierResNet(nn.Module):
    def __init__(self, n_label=1000):
        super(LinearClassifierResNet, self).__init__()
        self.classifier = nn.Sequential()
        nChannels = 2048
        self.classifier.add_module('Flatten', Flatten())
        print('classifier input: {}'.format(nChannels))
        self.classifier.add_module('LinearClassifier', nn.Linear(nChannels, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)
