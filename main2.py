import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from fit_routine import *

from covct import CovidCTDataset
from torchvision.models import resnet18
from torchvision.models import resnet50
from nabernet import NaberNet


PATH = './data/covct/'
IMGPATH = './data/CTX/'

# For Group 1
CLASSES = ['CT_NonCOVID', 'CT_COVID']       # Group 1 Distinctions
IMGSIZE = 224     # For Groups 1, 2

# For Group 3
# CLASSES = ['CTX_NC', 'CTX_CO']       # Group 3 Distinctions
# IMGSIZE = 448     # For Group 3

batchsize = 8

normalize = transforms.Normalize(mean=0.6292, std=0.3024)       # Group 1 Normalization.
# normalize = transforms.Normalize(mean=0.611, std=0.273)         # Group 2, 3 Normalization
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

# # GROUP 1 DATASETS
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
                         classes=CLASSES,
                         covid_files=PATH + 'Data-split/COVID/testCT_COVID.txt',
                         non_covid_files=PATH + 'Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=val_transformer)

# GROUP 3 DATASETS
# trainset = CovidCTDataset(root_dir=IMGPATH + '2A_images',
#                           classes=CLASSES,
#                           covid_files=IMGPATH + 'co_train.txt',
#                           non_covid_files=IMGPATH + 'nc_train.txt',
#                           transform=train_transformer
#                           )
# valset = CovidCTDataset(root_dir=IMGPATH + '2A_images',
#                         classes=CLASSES,
#                         covid_files=IMGPATH + 'co_val.txt',
#                         non_covid_files=IMGPATH + 'nc_val.txt',
#                         transform=val_transformer)
# testset = CovidCTDataset(root_dir=IMGPATH + '2A_images',
#                          classes=CLASSES,
#                          covid_files=IMGPATH + 'co_test.txt',
#                          non_covid_files=IMGPATH + 'nc_test.txt',
#                          transform=val_transformer)

train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=2)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False, num_workers=2)

model_name = "m_resnet18"
model = resnet18(pretrained=False)
# model_name = "m_resnet50"
# model = resnet50(pretrained=False)
# model_name = "nabernet"
# model = NaberNet()
# model = torch.load(model_name + ".tar")

learning_rate = 0.01       # For fine tuning.
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if __name__ == '__main__':
    fit(model, train_loader, test_loader, optimizer, 5)

