from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
import os
import random
import warnings

from ctx_dataset import CTDataset
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import alexnet
from nabernet import NaberNet

from fit_routine import *
from plot import Plot

ORG_PATH = './data/covct/'
CTX_PATH = './data/CTX/'
NII_PATH = './data/MedSeg/'

warnings.filterwarnings("ignore")
random.seed(12)

# =============================================================
# NOTE SELECT MODEL
# =============================================================
# model_name = "resnet18_b2"
# model = resnet18(pretrained=False)
# model_name = "resnet50"
# model = resnet50(pretrained=False)
# model_name = "alexnet"
# model = alexnet(pretrained=False)

# model_name = "nabernet_c3"
# B2 is the best, with 40 epochs.

model_name = "unet_a"
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=5, init_features=32, pretrained=False)
# TODO: Try it out as pretrained to see if dataset and all else is fine or not.


# Loading a pretrained model
# model = torch.load(model_name + ".tar")
# =============================================================

# =============================================================
# NOTE SELECT DATASET
# =============================================================
# SET_NAME = "UCSD AI4H"              # Contains 746 images.        (Set A)
# SET_NAME = "SARS-COV-2 CT-SCAN"     # Contains 2,481 images.      (Set B)
SET_NAME = "COVIDx CT-1"            # Contains 115,837 images.    (Set C)
# =============================================================

# =============================================================
# SUMMON PARAMETERS
# =============================================================
if "UCSD" in SET_NAME:
    CLASSES = ['UCSD_NC', 'UCSD_CO']
    IMGSIZE = 224
    EPOCHS = 10
    normalize = transforms.Normalize(mean=0.6292, std=0.3024)
    if "naber" in model_name:
        model = NaberNet(0)
if "SARS" in SET_NAME:
    CLASSES = ['SARSCT_NC', 'SARSCT_CO']
    IMGSIZE = 224
    EPOCHS = 40
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name:
        model = NaberNet(1)
elif "COVIDx" in SET_NAME:
    CLASSES = ['CTX_NC', 'CTX_CO']
    IMGSIZE = 424
    EPOCHS = 20
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name:
        model = NaberNet(2)

# =============================================================

# =============================================================
# SETUP TRANSFORMERS (Every dataset uses the same ones)
# =============================================================
batchsize = 8       # Chosen for the GPU: RTX 2060
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
# =============================================================

# =============================================================
# GENERATE DATASETS
# =============================================================
if "UCSD" in SET_NAME:
    trainset = CTDataset(root_dir=ORG_PATH,
                         classes=CLASSES,
                         covid_files=ORG_PATH + 'Data-split/COVID/old_split/trainCT_COVID.txt',
                         non_covid_files=ORG_PATH + 'Data-split/NonCOVID/old_split/trainCT_NonCOVID.txt',
                         transform=train_transformer)
    valset = CTDataset(root_dir=ORG_PATH,
                       classes=CLASSES,
                       covid_files=ORG_PATH + 'Data-split/COVID/old_split/valCT_COVID.txt',
                       non_covid_files=ORG_PATH + 'Data-split/NonCOVID/old_split/valCT_NonCOVID.txt',
                       transform=val_transformer)
    testset = CTDataset(root_dir=ORG_PATH,
                        classes=CLASSES,
                        covid_files=ORG_PATH + 'Data-split/COVID/old_split/testCT_COVID.txt',
                        non_covid_files=ORG_PATH + 'Data-split/NonCOVID/old_split/testCT_NonCOVID.txt',
                        transform=val_transformer)
elif "SARS" in SET_NAME:
    trainset = CTDataset(root_dir=ORG_PATH,
                         classes=CLASSES,
                         covid_files=ORG_PATH + 'Data-split/COVID/sarsct_co_train.txt',
                         non_covid_files=ORG_PATH + 'Data-split/NonCOVID/sarsct_nc_train.txt',
                         transform=train_transformer)
    testset = CTDataset(root_dir=ORG_PATH,
                        classes=CLASSES,
                        covid_files=ORG_PATH + 'Data-split/COVID/sarsct_co_test.txt',
                        non_covid_files=ORG_PATH + 'Data-split/NonCOVID/sarsct_nc_test.txt',
                        transform=val_transformer)
elif "COVIDx" in SET_NAME:
    trainset = CTDataset(root_dir=CTX_PATH + '2A_images',
                         classes=CLASSES,
                         covid_files=CTX_PATH + 'co_train.txt',
                         non_covid_files=CTX_PATH + 'nc_train.txt',
                         transform=train_transformer,
                         is_ctx=True)
    valset = CTDataset(root_dir=CTX_PATH + '2A_images',
                       classes=CLASSES,
                       covid_files=CTX_PATH + 'co_val.txt',
                       non_covid_files=CTX_PATH + 'nc_val.txt',
                       transform=val_transformer)
    testset = CTDataset(root_dir=CTX_PATH + '2A_images',
                        classes=CLASSES,
                        covid_files=CTX_PATH + 'co_test.txt',
                        non_covid_files=CTX_PATH + 'nc_test.txt',
                        transform=val_transformer,
                        is_ctx=True)

# =============================================================

# =============================================================
# CREATE DATA-LOADERS FOR ALL SETS
# =============================================================
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=1)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False, num_workers=1)
# =============================================================

# =============================================================
# SETS UP LEARNING RATE AND OPTIMIZER FOR FINE TUNING
# =============================================================
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# =============================================================

# =============================================================
# NOTE MAIN FUNCTION
# =============================================================
if __name__ == '__main__':

    # STARTING MESSAGE - LETS US KNOW MODEL AND DATASET IN-USE.
    print("\n==========================================")
    print("SELECTED MODEL: \t", model_name)
    print("SELECTED DATASET: \t", SET_NAME)
    print("EXPECTED EPOCHS: \t", EPOCHS)

    # ENSURES THAT THE RIGHT LOG FILES ARE MADE & USED.
    # NOTE to self, if you're gonna retest a model, erase its log first.
    TRAIN_PATH = "./logs/train_logger/__" + model_name + "__run___training.log"
    TEST_PATH = "./logs/test_logger/__" + model_name + "__run___test.log"
    TEST2_PATH = "./logs/test_logger/--" + model_name + "__split___test.log"
    PLOT_PATH = "./figures/" + model_name + "_plot.png"

    flag = False
    if os.path.exists(TRAIN_PATH):
        os.remove(TRAIN_PATH)
        flag = True
    if os.path.exists(TEST_PATH):
        os.remove(TEST_PATH)
        flag = True
    if os.path.exists(TEST2_PATH):
        os.remove(TEST2_PATH)
        flag = True
    if os.path.exists(PLOT_PATH):
        os.remove(PLOT_PATH)
        flag = True

    if flag:
        print("PRE-EXISTING LOGS: \t CLEARED")
    print("==========================================")

    # RUNS ALL TRAINING AND TESTS.
    fit(model, train_loader, test_loader, optimizer, epochs=EPOCHS, model_name=model_name)
    print("\n> All Epochs completed!")

    print("\n> Generating Plots...")
    Plot(model_name).plot()
    print("> Plot Closed.")
