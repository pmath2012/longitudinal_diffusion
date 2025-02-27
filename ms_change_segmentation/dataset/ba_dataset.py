import torch
from torch.utils.data import Dataset
import os
from utils.utils import load_image
import numpy as np
import pandas as pd

class MSBADataset(Dataset):
    """MS Siamese lesion dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ms_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ms_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.ms_frame.iloc[idx, 1])
        img_name2 =os.path.join(self.root_dir,self.ms_frame.iloc[idx, 2])
        mask_name =os.path.join(self.root_dir,self.ms_frame.iloc[idx, 3])
        boundary_name =os.path.join(self.root_dir,self.ms_frame.iloc[idx, -1])
        image_1 = load_image(img_name)
        image_2 = load_image(img_name2)
        mask = load_image(mask_name)
        boundary = load_image(boundary_name)
        sample = {'image_1': np.expand_dims(image_1, axis=0),
                'image_2': np.expand_dims(image_2, axis=0), 
                'mask': np.expand_dims(mask, axis=0),
                'boundary':np.expand_dims(boundary, axis=0)}

        if self.transform:
            sample = self.transform(sample)

        return sample