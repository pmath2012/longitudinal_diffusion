from torch import nn
import torch.nn.functional as Fn

# taken from  https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = Fn.sigmoid(pred)

        #flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        intersection = (pred * targets).sum()
        dice = (2.*intersection + smooth)/(pred.sum() + targets.sum() + smooth)

        return 1 - dice