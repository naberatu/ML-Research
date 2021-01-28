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
from covct import CovidCTDataset

random.seed(0)

log_dir = "~/logs"
writer = SummaryWriter(log_dir)
PATH = './data/covct/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(img):
    # img = img / 5 + 1     # unnormalize

    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image in enumerate(img):
        plt.subplot(int(len(img) / columns) + 1, columns, i + 1)
        plt.imshow(image[0])
    # for i in img:
    #     npimg = i.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    covid_files_path = PATH + "CT_COVID"
    nocov_files_path = PATH + "CT_NonCOVID"
    # covid_files = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
    # covid_images = [cv2.imread(x) for x in random.sample(covid_files, 5)]

    # plt.figure(figsize=(20, 10))
    # columns = 5
    # for i, image in enumerate(covid_images):
    #     plt.subplot(int(len(covid_images) / columns) + 1, columns, i + 1)
    #     plt.imshow(image)
    # plt.show()

    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    batchsize = 8
    trainset = CovidCTDataset(root_dir=PATH,
                              classes=['CT_NonCOVID', 'CT_COVID'],
                              covid_files=PATH + 'Data-split/COVID/trainCT_COVID.txt',
                              non_covid_files=PATH + 'Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir=PATH,
                            classes=['CT_NonCOVID', 'CT_COVID'],
                            covid_files=PATH + 'Data-split/COVID/valCT_COVID.txt',
                            non_covid_files=PATH + 'Data-split/NonCOVID/valCT_NonCOVID.txt',
                            transform=val_transformer)
    testset = CovidCTDataset(root_dir=PATH,
                             classes=['CT_NonCOVID', 'CT_COVID'],
                             covid_files=PATH + 'Data-split/COVID/testCT_COVID.txt',
                             non_covid_files=PATH + 'Data-split/NonCOVID/testCT_NonCOVID.txt',
                             transform=val_transformer)

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    diter_train = iter(train_loader)
    diter_val = iter(val_loader)
    diter_test = iter(test_loader)

    data = diter_train.__next__()
    image = data["img"]
    label = data["label"]
    print(image)
    imshow(image)
    print(label)


