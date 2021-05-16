
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# =============================================================
# NOTE DATASET CLASS
# =============================================================
class NIIDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.image_file = images
        self.mask_file = masks
        self.transform = transform

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, idx):
        image = nib.load(self.image_file)
        label = nib.load(self.mask_file)

        image = np.array(image.dataobj)
        image = Image.fromarray(image.astype("uint8"), 'RGB')

        label = np.array(label.dataobj)
        label = Image.fromarray(label.astype("uint8"), 'RGB')

        print("Image: ", type(image))
        print("Label: ", type(image))

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
