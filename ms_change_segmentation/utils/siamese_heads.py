import torch
import torch.nn as nn

import torch.nn.functional as Fn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DifferenceNetwork(nn.Module):
    def __init__(self, in_features=32, bn =False):
        super(DifferenceNetwork, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_features, in_features, 1)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features//2, 1)
        self.bn2 = nn.BatchNorm2d(in_features//2)
        self.conv3 = nn.Conv2d(in_features//2, in_features//4, 1)
        self.bn3 = nn.BatchNorm2d(in_features//4)
        self.conv4 = nn.Conv2d(in_features//4, 1, 1)

    def forward(self, x1, x2):
        output = torch.sub(x2, x1)
        output = self.conv1(output)
        if self.bn:
            output = self.bn1(output)
        output = Fn.relu(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        output = Fn.relu(output)
        output = self.conv3(output)
        if self.bn:
            output = self.bn3(output)
        output = Fn.relu(output)
        output = self.conv4(output)
        return output

class ConcatNetwork(nn.Module):
    def __init__(self, in_features=64, bn=False):
        super(ConcatNetwork, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_features, in_features, 1)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features//2, 1)
        self.bn2 = nn.BatchNorm2d(in_features//2)
        self.conv3 = nn.Conv2d(in_features//2, in_features//4, 1)
        self.bn3 = nn.BatchNorm2d(in_features//4)
        self.conv4 = nn.Conv2d(in_features//4, 1, 1)

    def forward(self, x1, x2):
        output = torch.cat((x2, x1), dim=1)
        output = self.conv1(output)
        if self.bn:
            output = self.bn1(output)
        output = Fn.relu(output)
        output = self.conv2(output)
        if self.bn:
            output = self.bn2(output)
        output = Fn.relu(output)
        output = self.conv3(output)
        if self.bn:
            output = self.bn3(output)
        output = Fn.relu(output)
        output = self.conv4(output)
        return output


