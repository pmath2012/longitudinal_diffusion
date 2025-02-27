import torch

from torch import nn
from glasses.models.segmentation.unet import UNet
from monai.networks.nets import SwinUNETR, UNETR, AttentionUnet

class UNetC(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetC, self).__init__()
        self.net = UNet(n_classes=num_classes, in_channels=in_channels)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1)
        out = self.net(x)
        return out
    
class SwinUNETRC(nn.Module):
    def __init__(self, num_classes):
        super(SwinUNETRC, self).__init__()
        self.net = SwinUNETR(img_size=(256,256),in_channels=2,out_channels=num_classes,spatial_dims=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1)
        out = self.net(x)
        return out

class UNETRC(nn.Module):
    def __init__(self, num_classes):
        super(UNETRC, self).__init__()
        self.net = UNETR(img_size=(256,256),in_channels=2,out_channels=num_classes,spatial_dims=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1)
        out = self.net(x)
        return out

class AttentionUNetC(nn.Module):
    def __init__(self, num_classes=1):
        super(AttentionUNetC, self).__init__()
        self.net = AttentionUnet(in_channels=2,
                                 spatial_dims=2,
                                 channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2),
                                 out_channels=num_classes)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1 )
        out = self.net(x)
        return out