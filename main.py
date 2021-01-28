import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import os
import cv2
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import glob
import shutil
import numpy as np

random.seed(0)

log_dir = "~/logs"
writer = SummaryWriter(log_dir)
PATH = './data/covct/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    covid_files_path = PATH + "CT_COVID"
    nocov_files_path = PATH + "CT_NonCOVID"
    covid_files = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
    covid_images = [cv2.imread(x) for x in random.sample(covid_files, 5)]

    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image in enumerate(covid_images):
        plt.subplot(int(len(covid_images) / columns) + 1, columns, i + 1)
        plt.imshow(image)
    plt.show()



