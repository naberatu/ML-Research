from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
import os
import random
import warnings

from imc_dataset import CTDataset
from imc_nabernet import NaberNet

from imc_fit import *
from imc_plot_run import *

dir_models = 'C:/Users/elite/PycharmProjects/Pytorch/models_old/'
dir_orig = 'C:/Users/elite/PycharmProjects/Pytorch/data/covct/'
dir_ctx = 'C:/Users/elite/PycharmProjects/Pytorch/data/ct_ctx/'

warnings.filterwarnings("ignore")
random.seed(12)

# =============================================================
# SELECT: Model, Name, and test_only
# =============================================================
# model_name = "resnet18_b2"
# model = resnet18(pretrained=False)
# model_name = "resnet50"
# model = resnet50(pretrained=False)

model_name = "nabernet_bn"       # B2 is the best, with 40 epochs.

# Whether the main should just run a test, or do a full fit.
# only_test = True
only_test = False
batchsize = 8       # Chosen for the GPU: RTX 2060

# Loading a pretrained model
model_loaded = False
# model_loaded = True
# model = torch.load(dir_models + model_name + ".tar")

# =============================================================
# SELECT: Dataset Name
# =============================================================
# SET_NAME = "UCSD AI4H"              # Contains 746 images.        (Set A)
SET_NAME = "SARS-COV-2 CT-SCAN"     # Contains 2,481 images.      (Set B)
# SET_NAME = "COVIDx CT-1"            # Contains 115,837 images.    (Set C)

# =============================================================
# STEP: Generate Dataset Prameters
# =============================================================
if "UCSD" in SET_NAME:
    CLASSES = ['UCSD_NC', 'UCSD_CO']
    IMGSIZE = 224
    EPOCHS = 10
    normalize = transforms.Normalize(mean=0.6292, std=0.3024)
    if "naber" in model_name and not model_loaded:
        model = NaberNet(0)
if "SARS" in SET_NAME:
    CLASSES = ['SARSCT_NC', 'SARSCT_CO']
    IMGSIZE = 224
    EPOCHS = 40
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name and not model_loaded:
        model = NaberNet(1)
elif "COVIDx" in SET_NAME:
    CLASSES = ['CTX_NC', 'CTX_CO']
    IMGSIZE = 424
    EPOCHS = 20
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name and not model_loaded:
        model = NaberNet(2)

# =============================================================
# STEP: Setup Dataset Transforms
# =============================================================
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
# STEP: GENERATE DATASETS
# =============================================================
if "UCSD" in SET_NAME:
    trainset = CTDataset(root_dir=dir_orig,
                         classes=CLASSES,
                         covid_files=dir_orig + 'Data-split/COVID/old_split/trainCT_COVID.txt',
                         non_covid_files=dir_orig + 'Data-split/NonCOVID/old_split/trainCT_NonCOVID.txt',
                         transform=train_transformer)
    valset = CTDataset(root_dir=dir_orig,
                       classes=CLASSES,
                       covid_files=dir_orig + 'Data-split/COVID/old_split/valCT_COVID.txt',
                       non_covid_files=dir_orig + 'Data-split/NonCOVID/old_split/valCT_NonCOVID.txt',
                       transform=val_transformer)
    testset = CTDataset(root_dir=dir_orig,
                        classes=CLASSES,
                        covid_files=dir_orig + 'Data-split/COVID/old_split/testCT_COVID.txt',
                        non_covid_files=dir_orig + 'Data-split/NonCOVID/old_split/testCT_NonCOVID.txt',
                        transform=val_transformer)
elif "SARS" in SET_NAME:
    trainset = CTDataset(root_dir=dir_orig,
                         classes=CLASSES,
                         covid_files=dir_orig + 'Data-split/COVID/sarsct_co_train.txt',
                         non_covid_files=dir_orig + 'Data-split/NonCOVID/sarsct_nc_train.txt',
                         transform=train_transformer)
    testset = CTDataset(root_dir=dir_orig,
                        classes=CLASSES,
                        covid_files=dir_orig + 'Data-split/COVID/sarsct_co_test.txt',
                        non_covid_files=dir_orig + 'Data-split/NonCOVID/sarsct_nc_test.txt',
                        transform=val_transformer)
elif "COVIDx" in SET_NAME:
    trainset = CTDataset(root_dir=dir_ctx + '2A_images',
                         classes=CLASSES,
                         covid_files=dir_ctx + 'co_train.txt',
                         non_covid_files=dir_ctx + 'nc_train.txt',
                         transform=train_transformer,
                         is_ctx=True)
    valset = CTDataset(root_dir=dir_ctx + '2A_images',
                       classes=CLASSES,
                       covid_files=dir_ctx + 'co_val.txt',
                       non_covid_files=dir_ctx + 'nc_val.txt',
                       transform=val_transformer)
    testset = CTDataset(root_dir=dir_ctx + '2A_images',
                        classes=CLASSES,
                        covid_files=dir_ctx + 'co_test.txt',
                        non_covid_files=dir_ctx + 'nc_test.txt',
                        transform=val_transformer,
                        is_ctx=True)

# =============================================================

# =============================================================
# STEP: CREATE DATA-LOADERS
# =============================================================
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=1)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False, num_workers=1)

# =============================================================
# SETS UP LEARNING RATE AND OPTIMIZER FOR FINE TUNING
# =============================================================
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# =============================================================
# STEP: MAIN FUNCTION
# =============================================================
if __name__ == '__main__':

    # Initialization information
    divider = '===================================================='
    print('\n' + divider)
    print("MODEL:\t\t\t\t", model_name)
    print("DATASET:\t\t\t", SET_NAME)
    print("CLASSES:\t\t\t", CLASSES)
    print("IMAGE BATCHES:\t\t", str(batchsize) + "x" + str(IMGSIZE) + "x" + str(IMGSIZE))
    print("TOTAL IMAGES:\t\t", '{:,}'.format(len(trainset) + len(testset)))
    print("EPOCHS:\t\t\t\t", str(EPOCHS))

    # Record log & figure directories
    TRAIN_PATH = "./logs/train_logger/__" + model_name + "__run___training.log"
    TEST_PATH = "./logs/test_logger/__" + model_name + "__run___test.log"
    TEST2_PATH = "./logs/test_logger/--" + model_name + "__split___test.log"
    PLOT_PATH = "./figures/" + model_name + "_plot.png"

    if not only_test:
        rem_old_file = False
        if os.path.exists(TRAIN_PATH):
            os.remove(TRAIN_PATH)
            rem_old_file = True
        if os.path.exists(TEST_PATH):
            os.remove(TEST_PATH)
            rem_old_file = True
        if os.path.exists(TEST2_PATH):
            os.remove(TEST2_PATH)
            rem_old_file = True
        if os.path.exists(PLOT_PATH):
            os.remove(PLOT_PATH)
            rem_old_file = True

        if rem_old_file:
            print("PRE-EXISTING LOGS: \t CLEARED")
        print(divider)
        fit(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
            epochs=EPOCHS, model_name=model_name, divider=divider)
        print("\n> All Epochs completed!")
    else:
        test(model=model, model_name=model_name, test_data_loader=test_loader, divider=divider, re_test=True)
        print(divider)
        print("\n> All Epochs completed!")

    print("\n> Generating Plots...")
    Plot(model_name).plot(only_test=only_test)
    print("> Plot Closed.")
