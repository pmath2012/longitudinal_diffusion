import torch
from torch import nn
from torch.nn import functional as F
from glasses.models.segmentation.unet import ConvBnAct, UNetDecoder, UNetEncoder
from glasses.models.base import VisionModule
from functools import partial 

class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, 1)
    
    def forward(self, x):
        out1 = self.relu(self.bnorm(self.conv1(x)))
        out2 = self.relu(self.bnorm(self.conv2(x)))
        out3 = self.relu(self.bnorm(self.conv3(x)))
        out4 = self.relu(self.bnorm(self.conv4(x)))
        out5 = F.adaptive_avg_pool2d(x, 1)
        out5 = F.interpolate(out5, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        out5 = self.relu(self.bnorm(self.conv5(out5)))
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.relu(self.bnorm(self.final_conv(out)))

        return out

class AtrousBasicBlock(nn.Sequential):
    """Basic Block for UNet. It is composed by a double 3x3 conv."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = partial(nn.ReLU, inplace=True),
        atrous_rates: list = [6, 12, 18],
        aspp: bool = False,
        bottleneck_feats: int = 1024,
        *args,
        **kwargs,
    ):
        layers = [
            ConvBnAct(
                in_features,
                out_features,
                kernel_size=3,
                activation=activation,
                *args,
                **kwargs,
            ),
            ConvBnAct(
                out_features,
                out_features,
                kernel_size=3,
                activation=activation,
                *args,
                **kwargs,
            ),
        ]
        # only apply aspp at the bottleneck
        if aspp and out_features == bottleneck_feats:
            layers.append(_ASPP(out_features, out_features, atrous_rates, *args, **kwargs)),
        
        super().__init__(*layers)

    
class ASPPUNet(VisionModule):
    def __init__(self, in_channels, n_classes, encoder=UNetEncoder, decoder=UNetDecoder,
                  block=AtrousBasicBlock, atrous_rates=[6, 12, 18],
                  encoder_widths=[64, 128, 256, 512, 1024],
                  decoder_widths=[256, 128, 64, 32],
                  aspp=True, *args, **kwargs):
        super().__init__()
        self.encoder = encoder(in_channels, block=block, widths=encoder_widths, atrous_rates=atrous_rates,
                               bottleneck_feats=encoder_widths[-1], aspp=aspp  *args, **kwargs)
        self.decoder = decoder(start_features=self.encoder.widths[-1],
                               lateral_widths=self.encoder.features_widths[::-1],
                               widths=decoder_widths, *args, **kwargs)
        self.head = nn.Conv2d(self.decoder.widths[-1], n_classes, 1)

    def forward(self, x1, x2): 
        x = torch.cat([x1, x2], dim=1)
        self.encoder.features
        x = self.encoder(x)
        features = self.encoder.features
        residuals = features[::-1]
        residuals.extend([None]*(len(self.decoder.layers) - len(residuals)))
        x = self.decoder(x, residuals)
        x = self.head(x)
        return x
    