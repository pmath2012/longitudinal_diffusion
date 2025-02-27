import torch
import numpy as np
from torchmetrics.functional import dice
from torchmetrics.functional.classification import accuracy, binary_f1_score
from tqdm import tqdm

def ba_inner_loop(model, data, criterion_m, criterion_b, optimizer, device='cpu', train=True):
    running_metrics = {'loss':0, 'boundary_loss':0, 'mask_loss':0, 'accuracy': 0, 'f1': 0, 'dice': 0}
    image1, image2, labels, boundary = data['image_1'], data['image_2'], data['mask'], data['boundary']
    image1 = image1.to(device)
    image2 = image2.to(device)
    labels = labels.to(device)
    boundary = boundary.to(device)
    if train:
        optimizer.zero_grad()
    # forward pass
    outputs_m, outputs_b = model(image1, image2)
    loss_m = criterion_m(outputs_m, labels)
    loss_b = criterion_b(outputs_b, labels)
    loss = loss_m+loss_b
    running_metrics['loss'] = loss.item()
    running_metrics['boundary_loss'] = loss_b.item()
    running_metrics['mask_loss'] = loss_m.item()
    running_metrics['accuracy'] = accuracy(outputs_m.data, labels, task='binary').cpu().numpy()
    running_metrics['f1'] = binary_f1_score(outputs_m.data, labels).cpu().numpy()
    running_metrics['dice'] = dice(outputs_m.data, labels).cpu().numpy()
    if train:
        loss.backward()
        optimizer.step()
    return running_metrics

def cd_inner_loop(model, data, criterion, optimizer, device='cpu', train=True):
    running_metrics = {'loss':0, 'accuracy': 0, 'f1': 0, 'dice': 0}
    image1, image2, labels = data['image_1'], data['image_2'], data['mask']
    image1 = image1.to(device)
    image2 = image2.to(device)
    labels = labels.to(device)
    if train:
        optimizer.zero_grad()
    # forward pass
    outputs_m = model(image1, image2)
    loss = criterion(outputs_m, labels)
    running_metrics['loss'] = loss.item()
    running_metrics['accuracy'] = accuracy(outputs_m.data, labels, task='binary').cpu().numpy()
    running_metrics['f1'] = binary_f1_score(outputs_m.data, labels).cpu().numpy()
    running_metrics['dice'] = dice(outputs_m.data, labels).cpu().numpy()
    if train:
        loss.backward()
        optimizer.step()
    return running_metrics

def train_cd_ba_model(model, trainloader, optimizer, criterion_m, criterion_b, device='cpu'):
    model.train()
    print('Training')
    epoch_metrics = {'loss':[], 'boundary_loss':[], 'mask_loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        running_metrics = ba_inner_loop(model, data, criterion_m, criterion_b, optimizer, device)
        for key in running_metrics.keys():
            epoch_metrics[key].append(running_metrics[key])
    
    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_loss_m = np.mean(epoch_metrics['mask_loss'])
    epoch_loss_b = np.mean(epoch_metrics['boundary_loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])

    return [epoch_loss, epoch_loss_m, epoch_loss_b], epoch_acc, epoch_dice, epoch_f1

# validation
def validate_cd_ba_model(model, testloader, criterion_m, criterion_b, device='cpu'):
    model.eval()
    print('Validation')
    epoch_metrics = {'loss':[], 'boundary_loss':[], 'mask_loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            running_metrics = ba_inner_loop(model, data, criterion_m, criterion_b, None, device, train=False)
            for key in running_metrics.keys():
                epoch_metrics[key].append(running_metrics[key])

    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_loss_m = np.mean(epoch_metrics['mask_loss'])
    epoch_loss_b = np.mean(epoch_metrics['boundary_loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])
    return [epoch_loss, epoch_loss_m, epoch_loss_b], epoch_acc, epoch_dice, epoch_f1

def train_cd_model(model, trainloader, optimizer, criterion, device='cpu'):
    model.train()
    print('Training')
    epoch_metrics = {'loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        running_metrics = cd_inner_loop(model, data, criterion, optimizer, device)
        for key in running_metrics.keys():
            epoch_metrics[key].append(running_metrics[key])
    
    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])

    return epoch_loss, epoch_acc, epoch_dice, epoch_f1

# validation
def validate_cd_model(model, testloader, criterion, device='cpu'):
    model.eval()
    print('Validation')
    epoch_metrics = {'loss':[], 'boundary_loss':[], 'mask_loss':[], 'accuracy': [], 'f1': [], 'dice': []}
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            running_metrics = cd_inner_loop(model, data, criterion, None, device, train=False)
            for key in running_metrics.keys():
                epoch_metrics[key].append(running_metrics[key])

    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])
    return epoch_loss, epoch_acc, epoch_dice, epoch_f1
