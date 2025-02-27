import torch
import torch.nn.functional as Fn
from torch import nn
from glasses.models.segmentation.unet import UNet
from monai.networks.nets import SwinUNETR, UNETR
from utils.siamese_heads import Identity, DifferenceNetwork, ConcatNetwork

class SiameseNetwork(nn.Module):
    def __init__(self, base_model,head_nw="difference", weights=None, n_classes=1):
        super(SiameseNetwork, self).__init__()

        if base_model is None:
            raise ValueError('basemodel cannot be null')
        elif base_model.lower() == "unet":
            self.base_model = UNet(n_classes=n_classes)
        elif base_model.lower() == "unetr":
            self.base_model = UNETR(img_size=(256,256),in_channels=1,out_channels=1,spatial_dims=2) 
        elif base_model.lower() == "swin":
            self.base_model = SwinUNETR(img_size=(256,256),in_channels=1,out_channels=1,spatial_dims=2) 
        else:
            raise Exception("Not yet Implemented")

        if weights is None:
            pass
        elif weights.endswith(".pt"):
            print(f"loading weights from : {weights}")
            self.base_model.load_state_dict(torch.load(weights))
        else:
            # code to add other weights to the base model like imagenets
            pass
        num_feats = 64
        if base_model == "unet": 
            self.base_model.head = Identity()
        elif base_model == "unetr":
            # unetr has 16 output channels before final layer, so concat network expects 32
            num_feats = 32 
            self.base_model.out = Identity()
        elif base_model == "swin":
            # unetr has 24 output channels before final layer, so concat network expects 32
            num_feats = 48 
            self.base_model.out = Identity()
        else:
            raise Exception("Not yet Implemented")

        

        if head_nw == "concat":
            self.head_nw = ConcatNetwork(num_feats)
        else:
            self.head_nw = DifferenceNetwork()
            
    def forward_once(self, x):
        output = self.base_model(x)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)

        output = self.head_nw(output1, output2)

        return output