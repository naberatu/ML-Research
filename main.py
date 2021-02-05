import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

from covct import CovidCTDataset
from covct import compute_metrics
from earlystop import EarlyStopping

random.seed(0)

log_dir = "~/logs"
writer = SummaryWriter(log_dir)
PATH = './data/covct/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = ['CT_NonCOVID', 'CT_COVID']



def imshow(img):
    # img = img / 5 + 1     # unnormalize

    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image in enumerate(img):
        plt.subplot(int(len(img) / columns) + 1, columns, i + 1)
        plt.imshow(image[0])
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    batchsize = 8
    trainset = CovidCTDataset(root_dir=PATH,
                              classes=CLASSES,
                              covid_files=PATH + 'Data-split/COVID/trainCT_COVID.txt',
                              non_covid_files=PATH + 'Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir=PATH,
                            classes=CLASSES,
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
    # print(image)
    # imshow(image)
    # print(label)

    input_tensor = image[0]
    input_batch = input_tensor.unsqueeze(0)

    # model = vgg16(pretrained=True)
    # model = resnet18(pretrained=True)
    model = resnet50(pretrained=True)
    # model.eval()

    # model.classifier[6] = nn.Linear(4096, 2)
    input_batch = input_batch.to(DEVICE)
    model.to(DEVICE)

    # with torch.no_grad():
    #     output = model(input_batch)

    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # TRAINING LOOP
    best_model = model
    best_val_score = 0

    criterion = nn.CrossEntropyLoss()

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

            # Perform gradient udpate
            if iter_num % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate the number of correctly classified examples
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Compute and print the performance metrics
        metrics_dict = compute_metrics(model, val_loader, DEVICE)
        print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch, iter_num))
        print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
        print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
        print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
        print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
        print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
        print("------------------------------------------------------------------------------")

        # Save the model with best validation accuracy
        if metrics_dict['Accuracy'] > best_val_score:
            torch.save(model, "best_model.pkl")
            best_val_score = metrics_dict['Accuracy']

        # print the metrics for training data for the epoch
        print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
                   100.0 * train_correct / len(train_loader.dataset)))

        # log the accuracy and losses in tensorboard
        writer.add_scalars("Losses", {'Train loss': train_loss / len(train_loader),
                                      'Validation_loss': metrics_dict["Validation Loss"]},
                           epoch)
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
