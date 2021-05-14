import warnings

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchstat import stat

import ctxdataset
from nabernet import NaberNet

import os
import shutil
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil
import numpy as np
import seaborn as sns

from torchvision.models import vgg16
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet34
from torchvision.models import resnet101
from torchvision.models import resnet152

from dataset import CTDataset
from dataset import compute_metrics
# from ctxdataset import CTXDataset
# from ctxdataset import compute_metrics
from earlystop import EarlyStopping

import itertools
import threading
import time
import sys

random.seed(0)

# PARAMETERS
log_dir = "~/logs"
writer = SummaryWriter(log_dir)
PATH = './data/covct/'
CTX_PATH = './data/CTX/'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CLASSES = ['UCSD_NC', 'UCSD_CO']            # Group 1 Distinctions
CLASSES = ['SARSCT_NC', 'SARSCT_CO']       # Group 2 Distinctions
# CLASSES = ['CTX_NC', 'CTX_CO']       # Group 3 Distinctions

IMGSIZE = 224     # For Groups 1, 2
# IMGSIZE = 448       # For Group 3


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #
    # ==========================================
    # IMAGE-SORTING CODE
    # ==========================================
    #

    # files = ctxdataset.read_txt('./data/CTX/test_COVIDx_CT-2A.txt')
    # images = list()
    # for file in files:
    #     imgname = file.split(" ")[0]
    #     # if "NCP" in imgname or "Normal" in imgname:
    #     if "NCP" not in imgname and "Normal" not in imgname:
    #         images.append('./data/CTX/2A_images_c/' + imgname)
    #
    # print(images[0:5])
    #
    # for img in images:
    #     try:
    #         # shutil.move(img, './data/CTX/nc_test')
    #         shutil.move(img, './data/CTX/co_test')
    #     except: continue
    #
    # print("DONE")

    # #
    # # ==========================================
    # # NAME-READING CODE
    # # ==========================================
    # #
    #
    # a = open("./data/CTX/co_val.txt", "w")
    # for path, subdirs, files in os.walk(r'C:/Users/elite/PycharmProjects/Pytorch/data/CTX/co_val'):
    #     for filename in files:
    #         a.write(str(filename + '\n'))
    #
    # ==========================================
    # DATASET CODE
    # ==========================================
    #

    print("\nConstructing Datasets...", end='\t')

    # NORMALIZATION AND TRANSFORMERS
    # normalize = transforms.Normalize(mean=0.6292, std=0.3024)       # Group 1 Normalization.
    normalize = transforms.Normalize(mean=0.611, std=0.273)         # Group 2, 3 Normalization
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMGSIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transformer = transforms.Compose([
        transforms.Resize((IMGSIZE, IMGSIZE)),
        transforms.ToTensor(),
        normalize
    ])

    # DATASET AND DATALOADER CREATION
    batchsize = 8

    # # GROUP 1 (746 IMAGES)
    # trainset = CTDataset(root_dir=PATH,
    #                           classes=CLASSES,
    #                           covid_files=PATH + 'Data-split/COVID/old_split/trainCT_COVID.txt',
    #                           non_covid_files=PATH + 'Data-split/NonCOVID/old_split/trainCT_NonCOVID.txt',
    #                           transform=train_transformer)
    # valset = CTDataset(root_dir=PATH,
    #                         classes=CLASSES,
    #                         covid_files=PATH + 'Data-split/COVID/old_split/valCT_COVID.txt',
    #                         non_covid_files=PATH + 'Data-split/NonCOVID/old_split/valCT_NonCOVID.txt',
    #                         transform=val_transformer)
    # testset = CTDataset(root_dir=PATH,
    #                          classes=CLASSES,
    #                          covid_files=PATH + 'Data-split/COVID/old_split/testCT_COVID.txt',
    #                          non_covid_files=PATH + 'Data-split/NonCOVID/old_split/testCT_NonCOVID.txt',
    #                          transform=val_transformer)

    # GROUP 2 (2481 IMAGES)
    trainset = CTDataset(root_dir=PATH,
                              classes=CLASSES,
                              covid_files=PATH + 'Data-split/COVID/old_split/new_CT_CO_train.txt',
                              non_covid_files=PATH + 'Data-split/NonCOVID/old_split/new_CT_NC_train.txt',
                              transform=train_transformer)
    valset = CTDataset(root_dir=PATH,
                            classes=CLASSES,
                            covid_files=PATH + 'Data-split/COVID/old_split/new_CT_CO_val.txt',
                            non_covid_files=PATH + 'Data-split/NonCOVID/old_split/new_CT_NC_val.txt',
                            transform=val_transformer)
    testset = CTDataset(root_dir=PATH,
                             classes=CLASSES,
                             covid_files=PATH + 'Data-split/COVID/old_split/new_CT_CO_test.txt',
                             non_covid_files=PATH + 'Data-split/NonCOVID/old_split/new_CT_NC_test.txt',
                             transform=val_transformer)

    # GROUP 3 (115,837 IMAGES)
    # trainset = CTDataset(root_dir=CTX_PATH + '2A_images',
    #                           classes=CLASSES,
    #                           covid_files=CTX_PATH + 'co_train.txt',
    #                           non_covid_files=CTX_PATH + 'nc_train.txt',
    #                           transform=train_transformer
    #                           )
    # valset = CTDataset(root_dir=CTX_PATH + '2A_images',
    #                         classes=CLASSES,
    #                         covid_files=CTX_PATH + 'co_val.txt',
    #                         non_covid_files=CTX_PATH + 'nc_val.txt',
    #                         transform=val_transformer)
    # testset = CTDataset(root_dir=CTX_PATH + '2A_images',
    #                          classes=CLASSES,
    #                          covid_files=CTX_PATH + 'co_test.txt',
    #                          non_covid_files=CTX_PATH + 'nc_test.txt',
    #                          transform=val_transformer)

    print("DONE\nSetting Up DataLoaders...", end='\t')
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    #
    # ==========================================
    # NORMALIZATION CODE
    # ==========================================
    #
    # print("DONE\nSetting Up Norm Loader...", end='\t')
    # batchsize = len(trainset)
    # loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=0)
    #
    # print("DONE\nUsing DataLoader...", end='\t')
    # data = next(iter(loader))
    # print("DONE")
    # print(data["img"].mean(), data["img"].std())

    #
    # ==========================================
    # TRAINING CODE
    # ==========================================
    #
    print("DONE\nBeginning Model Setup...", end='\t')

    data = next(iter(train_loader))
    image = data["img"][0]

    # CONSTRUCT MODELS
    input_tensor = image
    input_batch = input_tensor.unsqueeze(0)

    # model_name = "m_vgg16"
    # model = vgg16(pretrained=False)
    # model_name = "m_resnet18"
    # model = resnet18(pretrained=False)
    # model_name = "m_resnet50"
    # model = resnet50(pretrained=False)
    # model_name = "customnet42"

    # model = torch.load("m_resnet18.tar")
    model_name = "nabernet_b_new"
    model = NaberNet(1)

    if model_name == "m_vgg16":
        model.classifier[6] = nn.Linear(4096, 2)
    elif model_name == "m_resnet18":
        model.fc = nn.Linear(512, len(CLASSES))

    input_batch = input_batch.to(DEVICE)
    model.to(DEVICE)

    # OPTIMIZATION DATA
    learning_rate = 0.01       # For fine tuning.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_score = 0
    criterion = nn.CrossEntropyLoss()

    print("DONE")
    # TRAINING LOOP
    for epoch in range(1, 41):
    # for epoch in range(1, 2):
        print("\nTraining Epoch", str(epoch) + "...", end='\t')

        model.train()
        train_loss = 0
        train_correct = 0

        for iter_num, data in enumerate(train_loader):
            image, target = data['img'].to(DEVICE), data['label'].to(DEVICE)

            # Compute the loss
            output = model(image)
            loss = criterion(output, target.long()) / 8

            # Log loss
            train_loss += loss.item()
            loss.backward()

            # Perform gradient update       DONE AT INTERVALS OF 8 FOR THE RTX 2060 GPU.
            if iter_num % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate the number of correctly classified examples
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Calculate performance metrics based on the Validation Set:
        metrics_dict = compute_metrics(model, val_loader, DEVICE)

        # Print Results
        print("DONE")
        print('\n------------------ Epoch {} -------------------------------------'.format(epoch))
        print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))

        # Save the model with best Validation accuracy (the point of validation)
        if metrics_dict['Accuracy'] > best_val_score:
            torch.save(model, model_name + ".tar")
            torch.save(model.state_dict(), model_name + "_dict.pth")
            best_val_score = metrics_dict['Accuracy']

        # print the metrics for training data for the epoch
        print('Average loss: \t{:.4f}\nTraining: \t\t{}/{} ({:.0f}%)\nValidation: \t{:.0f}%'.format(
            train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset), 100.0 * metrics_dict["Accuracy"]))
        print("-----------------------------------------------------------------")

        # log the accuracy and losses in tensorboard
        writer.add_scalars("Losses", {'Train loss': train_loss / len(train_loader),
                                      'Validation_loss': metrics_dict["Validation Loss"]}, epoch)
        writer.add_scalars("Accuracies", {"Train Accuracy": 100.0 * train_correct / len(train_loader.dataset),
                                          "Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

        early_stopper = EarlyStopping(patience=5)
        # Add data to the EarlyStopper object
        early_stopper.add_data(model, metrics_dict['Validation Loss'], metrics_dict['Accuracy'])

        # If both accuracy and loss are not improving, stop the training
        if early_stopper.stop() == 1:
            break

        # if only loss is not improving, lower the learning rate
        if early_stopper.stop() == 3:
            for param_group in optimizer.param_groups:
                learning_rate *= 0.1
                param_group['lr'] = learning_rate
                print('Updating the learning rate to {}'.format(learning_rate))
                early_stopper.reset()

    #
    # ==========================================
    # TESTING CODE
    # ==========================================
    #

    print("DONE\nTesting...", end='\t')
    model = torch.load(model_name + ".tar")
    model.load_state_dict(torch.load(model_name + "_dict.pth"))
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for data in test_loader:
            images = data["img"]
            labels = data["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("DONE\n")
    summary(model, (3, IMGSIZE, IMGSIZE))
    # stat(model.cpu(), (3, IMGSIZE, IMGSIZE))      # Use when you are on the compression stage.

    print("| Valid. Accuracy:  \t{:.1%}".format(best_val_score))
    print("| Testing Accuracy: \t{} /{} = {:.1%}".format(correct, total, correct/total))
    print("================================================================")


