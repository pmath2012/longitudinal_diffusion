import os
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.lesions_dataset import MSLesionDataset
from dataset.siamese_dataset import MSSiameseLesionDataset
from dataset.ba_dataset import MSBADataset
from utils.group_transforms import Normalize, NormalizeSiamese, NormalizeBASiamese


def get_train_loaders(train_file,valid_file, data_directory, batch_size):
    train_csv = data_directory+train_file
    valid_csv = data_directory+valid_file
    train_dataset = MSLesionDataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               Normalize()
                                           ]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


    valid_dataset = MSLesionDataset(csv_file=valid_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               Normalize()
                                           ]))

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    return train_dataloader, valid_dataloader

def get_siamese_test_loaders(test_file, data_directory, with_boundary=False):
    test_csv = data_directory+test_file
    if with_boundary:
        target_dataset = MSBADataset
        target_transform = NormalizeBASiamese
    else:
        target_dataset = MSSiameseLesionDataset
        target_transform = NormalizeSiamese
    test_dataset = target_dataset(csv_file=test_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               target_transform(v_flip=False, h_flip=False, elastic_transform=False)
                                           ]))

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0)


    return test_dataloader

def get_siamese_train_loaders(train_file,valid_file, data_directory, batch_size=4, with_boundary=False):
    train_csv = os.path.join(data_directory,train_file)
    valid_csv = os.path.join(data_directory,valid_file)
    if with_boundary:  
        target_dataset = MSBADataset
        target_transform = NormalizeBASiamese
    else :
        target_dataset = MSSiameseLesionDataset
        target_transform = NormalizeSiamese
    train_dataset = target_dataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               target_transform(v_flip=True, h_flip=True, elastic_transform=True)
                                           ]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


    valid_dataset = target_dataset(csv_file=valid_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               target_transform(v_flip=False, h_flip=False, elastic_transform=False)
                                           ]))

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    return train_dataloader, valid_dataloader
