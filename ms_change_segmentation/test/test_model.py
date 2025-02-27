import torch
import argparse
import os
import cv2
import numpy as np
from dataset import get_siamese_test_loaders
from baselines.siamese_unet import SiameseNetwork
from torch.nn import functional as F
from tqdm import tqdm
from atrous_networks.unet import ASPPUNet
from baselines.unet import UNetC, SwinUNETRC, UNETRC, AttentionUNetC
from transformer import vision_transformer

def predict(model_name, model, dataloader, target_dir, dataset, device, mode=None, with_boundary=False):
    model.eval()
    for i_batch, sample in tqdm(enumerate(dataloader)):
        if with_boundary:
            image_1, image_2, mask, boundary = sample['image_1'], sample['image_2'], sample['mask'], sample['boundary']
        else:
            image_1, image_2, mask = sample['image_1'], sample['image_2'], sample['mask']
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        mask = mask.to(device)
        if with_boundary:
            boundary = boundary.to(device)
            output_m, output_b = model(image_1, image_2)
        else:
            output_m = F.sigmoid(model(image_1, image_2))
        
        output = output_m.data.cpu().view(256,256).numpy()
        output = np.uint8(output*255)
        filename = f'test_{i_batch}.png'
        path = os.path.join(target_dir,mode, dataset, model_name)
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, filename), output)

def parse_args():
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--model_name', type=str, default='unet', help='Model name')
    parser.add_argument('--model_path', type=str, default='unet.pth', help='Model path')
    parser.add_argument('--dataset', type=str, default='svuh', help='Dataset')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Test file')
    parser.add_argument('--data_directory', type=str, default='/home/prateek/from_kipchoge/ms_project/change_balcrop256/', help='Data directory')
    parser.add_argument('--with_boundary', action='store_true', help='Use boundary')
    parser.add_argument('--target_dir', type=str, default='/home/prateek/ms_change_detection/predictions/', help='Output directory')
    parser.add_argument('--mode', type=str, default='pretrained', help='Mode of training')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    return parser.parse_args()

def adjust_keys(state_dict, orig_key, replace_key):
    # Rename keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(orig_key, replace_key)  # Replace 'bn' with 'norm' for unet
        new_state_dict[new_key] = value

    return new_state_dict

def adjust_keys_extended(state_dict, old_prefix='bn', new_prefix='norm', max_idx=3):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for i in range(1, max_idx + 1):
            old_str = f"{old_prefix}{i}"
            new_str = f"{new_prefix}{i}"
            new_key = new_key.replace(old_str, new_str)
        new_state_dict[new_key] = value
    return new_state_dict


if __name__ == '__main__':
    print('Begin Testing ------->')
    args = parse_args()
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    dataset = args.dataset
    if args.model_name == 'unet':
        model = UNetC(in_channels=2, num_classes=1)
    elif args.model_name == 'siam_unet_c':
        model = SiameseNetwork(base_model="unet", head_nw="concat")
    elif args.model_name == 'attention_unet':
        model = AttentionUNetC(num_classes=1)
    elif args.model_name == 'siam_unet_d':
        model = SiameseNetwork(base_model="unet")
    elif args.model_name == 'unetrc':
        model = UNETRC(num_classes=1)
    elif args.model_name == 'swinunetrc':
        model = SwinUNETRC(num_classes=1)
    elif args.model_name == 'asppunet':
        model = ASPPUNet(in_channels=2, n_classes=1)
    elif args.model_name == 'vitseg_r18_backbone':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet18')
    elif args.model_name == 'vitseg_r50_backbone':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet50')
    else:
        raise ValueError("Unsupported model name")
   
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if args.model_name == "unet" or args.model_name == "siam_unet_c":
        checkpoint = adjust_keys(checkpoint, "bn", "norm")
    if args.model_name == "siam_unet_c":
        checkpoint = adjust_keys_extended(checkpoint, "norm", "bn")

    model.load_state_dict(checkpoint)
    model.to(device)
    print('\tModel loaded successfully -------> ')
    test_dataloader = get_siamese_test_loaders(args.test_file, args.data_directory, with_boundary=args.with_boundary)
    predict(args.model_name, model, test_dataloader, args.target_dir, dataset, device, mode=args.mode, with_boundary=args.with_boundary)

    print('Done!')
