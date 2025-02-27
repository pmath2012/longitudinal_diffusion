import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from baselines.unet import UNetC, SwinUNETRC, UNETRC, AttentionUNetC
from baselines.siamese_unet import SiameseNetwork
from transformer import vision_transformer
from atrous_networks.unet import ASPPUNet
from changeformer.changeformer import ChangeFormerV4
from utils.utils import get_loss_function
from dataset import get_siamese_train_loaders
from train.training import train_cd_model, validate_cd_model

def check_keys(model, pretrain_path):
    # Check for matching keys
    pretrain = torch.load(pretrain_path)
    pretrained_keys = set(pretrain.keys())
    model_keys = set(model.state_dict().keys())

    missing_keys = model_keys - pretrained_keys
    unexpected_keys = pretrained_keys - model_keys

    if len(missing_keys) > 0:
        print("Missing keys in Siamese network:", missing_keys)
    else: 
        print("No Missing keys")

    if len(unexpected_keys) > 0:
        print("Unexpected keys in pre-trained UNet:", unexpected_keys)
    else:
        print("No unexpected keys")


def parse_args():
    parser = argparse.ArgumentParser(description='Train baselines')
    parser.add_argument('--model_name', type=str, default='unet', help='Model name to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--pretrain', action='store_true', help='Use pre-trained model')
    parser.add_argument('--pretrain_path', type=str, default="", help='Path to pre-trained model')
    parser.add_argument('--loss', type=str, choices=['f0.5', 'f1', 'f2'], default='f0.5', help='Mask loss function')
    parser.add_argument('--data_directory', type=str, default='', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--training_file', type=str, default='train.csv', help='Training file')
    parser.add_argument('--validation_file', type=str, default='valid.csv', help='Validation file')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    print('-'*100)
    print(f"\n\n\nTraining {args.model_name} \n\n\n")
    print('-'*100)

    learning_rate = args.learning_rate
    loss = get_loss_function(args.loss)
    epochs = args.epochs
    pretrain = args.pretrain
    pretrain_path = args.pretrain_path
    data_directory = args.data_directory
    train_file = args.training_file
    valid_file = args.validation_file
    batch_size = args.batch_size
    data_directory = args.data_directory
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    
    # Initialize TensorBoard
    # writer = SummaryWriter()
    if args.model_name == 'unet':
        model = UNetC(in_channels=2, num_classes=1)
    if args.model_name == 'attention_unet':
        model = AttentionUNetC(num_classes=1)
    elif args.model_name == 'siam_unet_c':
        model = SiameseNetwork(base_model="unet", head_nw="concat")
    elif args.model_name == 'siam_unet_d':
        model = SiameseNetwork(base_model="unet")
    elif args.model_name == 'unetrc':
        model = UNETRC(num_classes=1)
    elif args.model_name == 'swinunetrc':
        model = SwinUNETRC(num_classes=1)
    elif args.model_name == 'asppunet':
        model = ASPPUNet(in_channels=2, n_classes=1)
    elif args.model_name == 'vitseg_r18_backbone':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet18', pretrained=False)
    elif args.model_name == 'vitseg_r50_backbone':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet50')
    elif args.model_name == 'vitseg_r50_backbone_enc_dec_2':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet50', enc_depth=2, dec_depth=2)
    elif args.model_name == 'vitseg_r101_backbone':
        model = vision_transformer.ViTSeg(in_channels=1, num_classes=1, with_pos='learned',backbone='resnet101')
    elif args.model_name == 'changeformer':
        model = ChangeFormerV4(input_nc=1, output_nc=1, decoder_softmax=True)
    else:
        raise ValueError("Unsupported model name")

    if pretrain:
        check_keys(model, pretrain_path)
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    
    train_dataloader, valid_dataloader = get_siamese_train_loaders(train_file,valid_file, data_directory,batch_size, with_boundary=False)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    ckpt=f"{args.model_name}_{args.loss}_epochs_{epochs}.pth"
    
    # lists to keep track of losses and accuracies
    best_loss = 0
    best_epoch = 0

    # start the training
    model = model.to(device)

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_losses, train_epoch_acc, train_epoch_dice, train_epoch_f1 = train_cd_model(model, train_dataloader,
                                                optimizer, loss ,device)
        valid_epoch_losses, valid_epoch_acc, valid_epoch_dice, valid_epoch_f1 = validate_cd_model(model, valid_dataloader,
                                                    loss, device)

        valid_loss = valid_epoch_losses

        # writer.add_scalar('Train/Loss', train_epoch_losses, epoch)
        # writer.add_scalar('Train/Accuracy', train_epoch_acc, epoch)
        # writer.add_scalar('Train/Dice', train_epoch_dice, epoch)
        # writer.add_scalar('Train/F1', train_epoch_f1, epoch)
        # writer.add_scalar('Validation/Loss', valid_epoch_losses, epoch)
        # writer.add_scalar('Validation/Accuracy', valid_epoch_acc, epoch)
        # writer.add_scalar('Validation/Dice', valid_epoch_dice, epoch)
        # writer.add_scalar('Validation/F1', valid_epoch_f1, epoch)

        print(f"Training : {train_epoch_losses:.3f}, training acc: {train_epoch_acc:.3f}, dice : {train_epoch_dice:.3f}, f1: {train_epoch_f1:.3f}")
        print(f"Validation : {valid_epoch_losses:.3f}, validation acc: {valid_epoch_acc:.3f}, dice : {valid_epoch_dice:.3f}, f1: {valid_epoch_f1:.3f}")

        if epoch == 0:
            print("saving first model")
            best_loss=valid_epoch_losses
            best_epoch=epoch
            torch.save(model.state_dict(), ckpt)
        elif valid_epoch_losses < best_loss:
            print(f"model loss improved from {best_loss} to {valid_epoch_losses} : saving best")
            best_loss=valid_epoch_losses
            best_epoch=epoch
            torch.save(model.state_dict(), ckpt)
        else:
            print(f"val loss did not improve from {best_loss} @ {best_epoch+1}")

        print('-'*100)
    # writer.close()
