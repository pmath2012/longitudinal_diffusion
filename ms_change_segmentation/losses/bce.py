from torch.nn import functional as Fn
import torch

class BCE(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE, self).__init__()
        self.weight=weight

    def forward(self, pred, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = Fn.sigmoid(pred)

        #flatten label and prediction tensors
        pred = pred.view(-1).float()
        targets = targets.view(-1).float()
        return Fn.binary_cross_entropy(pred, targets, weight=self.weight)
