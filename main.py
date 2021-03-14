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
from nabernet import NaberNet

import os
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

from covct import CovidCTDataset
from covct import compute_metrics
from earlystop import EarlyStopping

random.seed(0)

# PARAMETERS
log_dir = "~/logs"
writer = SummaryWriter(log_dir)
PATH = './data/covct/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CLASSES = ['CT_NonCOVID', 'CT_COVID']       # Group 1 Distinctions
CLASSES = ['new_CT_NC', 'new_CT_CO']       # Group 2 Distinctions


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    nocov_files_path = PATH + CLASSES[0]
    covid_files_path = PATH + CLASSES[1]

    # NORMALIZATION AND TRANSFORMERS
    # normalize = transforms.Normalize(mean=0.6292, std=0.3024)       # Group 1 Normalization.
    normalize = transforms.Normalize(mean=0.611, std=0.273)                 # Group 2 Normalization
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # DATASET AND DATALOADER CREATION
    batchsize = 8

    # GROUP 1 (425 TRAINING IMAGES)
    # trainset = CovidCTDataset(root_dir=PATH,
    #                           classes=CLASSES,
    #                           covid_files=PATH + 'Data-split/COVID/trainCT_COVID.txt',
    #                           non_covid_files=PATH + 'Data-split/NonCOVID/trainCT_NonCOVID.txt',
    #                           transform=train_transformer)
    # valset = CovidCTDataset(root_dir=PATH,
    #                         classes=CLASSES,
    #                         covid_files=PATH + 'Data-split/COVID/valCT_COVID.txt',
    #                         non_covid_files=PATH + 'Data-split/NonCOVID/valCT_NonCOVID.txt',
    #                         transform=val_transformer)
    # testset = CovidCTDataset(root_dir=PATH,
    #                          classes=CLASSES,
    #                          covid_files=PATH + 'Data-split/COVID/testCT_COVID.txt',
    #                          non_covid_files=PATH + 'Data-split/NonCOVID/testCT_NonCOVID.txt',
    #                          transform=val_transformer)

    # GROUP 2 (2400 TOTAL IMAGES)
    trainset = CovidCTDataset(root_dir=PATH,
                              classes=CLASSES,
                              covid_files=PATH + 'Data-split/COVID/new_CT_CO_train.txt',
                              non_covid_files=PATH + 'Data-split/NonCOVID/new_CT_NC_train.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir=PATH,
                            classes=CLASSES,
                            covid_files=PATH + 'Data-split/COVID/new_CT_CO_val.txt',
                            non_covid_files=PATH + 'Data-split/NonCOVID/new_CT_NC_val.txt',
                            transform=val_transformer)
    testset = CovidCTDataset(root_dir=PATH,
                             classes=CLASSES,
                             covid_files=PATH + 'Data-split/COVID/new_CT_CO_test.txt',
                             non_covid_files=PATH + 'Data-split/NonCOVID/new_CT_NC_test.txt',
                             transform=val_transformer)

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    #
    # ==========================================
    # NORMALIZATION CODE
    # ==========================================
    #

    # batchsize = len(trainset)
    # loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=1)
    #
    # data = next(iter(loader))
    # print(data["img"].mean(), data["img"].std())

    #
    # ==========================================
    # TRAINING CODE
    # ==========================================
    #

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

    model_name = "customnet"
    model = NaberNet()

    # model = torch.load(model_name + ".tar")

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

    # TRAINING LOOP
    for epoch in range(60):

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

            # Perform gradient udpate       DONE AT INTERVALS OF 8 FOR THE RTX 2060 GPU.
            if iter_num % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate the number of correctly classified examples
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Calculate performance metrics with the Validation Set:
        # This metrics dict essentially stores the results based on the validation set.
        metrics_dict = compute_metrics(model, val_loader, DEVICE)

        # Print Results
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

    summary(model, (3, 224, 224))
    # stat(model.cpu(), (3, 224, 224))      # TODO Use when you are on the compression stage.

    print("| Valid. Accuracy:  \t{:.1%}".format(best_val_score))
    print("| Testing Accuracy: \t{} /{} = {:.1%}".format(correct, total, correct/total))
    print("================================================================")


