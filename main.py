from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import torch.optim as optim
from torchsummary import summary

from covct import CovidCTDataset
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import alexnet
from nabernet import NaberNet

from fit_routine import *
from plotting import *

ORG_PATH = './data/covct/'
CTX_PATH = './data/CTX/'

# =============================================================
# NOTE SELECT MODEL
# =============================================================
# model_name = "m_resnet18"
# model = resnet18(pretrained=False)
# model_name = "m_resnet50"
# model = resnet50(pretrained=False)
# model_name = "m_alexnet"
# model = alexnet(pretrained=False, num_classes=2)

model_name = "nabernet"

# Loading a pretrained model
# model = torch.load(model_name + ".tar")
# =============================================================

# =============================================================
# NOTE SELECT DATASET
# =============================================================
# SET_NAME = "UCSD AI4H"              # Contains 728 images.
# SET_NAME = "SARS-COV-2 CT-SCAN"     # Contains 2,481 images.
SET_NAME = "COVIDx CT-1"            # Contains 115,837 images.
# =============================================================

# =============================================================
# SUMMON PARAMETERS
# =============================================================
if "UCSD" in SET_NAME:
    CLASSES = ['CT_NonCOVID', 'CT_COVID']
    IMGSIZE = 224
    normalize = transforms.Normalize(mean=0.6292, std=0.3024)
    if "naber" in model_name:
        model = NaberNet(group=0)
if "SARS" in SET_NAME:
    CLASSES = ['new_CT_NC', 'new_CT_CO']
    IMGSIZE = 224
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name:
        model = NaberNet(group=1)
elif "COVIDx" in SET_NAME:
    CLASSES = ['CTX_NC', 'CTX_CO']
    IMGSIZE = 448
    normalize = transforms.Normalize(mean=0.611, std=0.273)
    if "naber" in model_name:
        model = NaberNet(group=2)
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
    trainset = CovidCTDataset(root_dir=ORG_PATH,
                              classes=CLASSES,
                              covid_files=ORG_PATH + 'Data-split/COVID/trainCT_COVID.txt',
                              non_covid_files=ORG_PATH + 'Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    testset = CovidCTDataset(root_dir=ORG_PATH,
                             classes=CLASSES,
                             covid_files=ORG_PATH + 'Data-split/COVID/testCT_COVID.txt',
                             non_covid_files=ORG_PATH + 'Data-split/NonCOVID/testCT_NonCOVID.txt',
                             transform=val_transformer)
elif "SARS" in SET_NAME:
    trainset = CovidCTDataset(root_dir=ORG_PATH,
                              classes=CLASSES,
                              covid_files=ORG_PATH + 'Data-split/COVID/new_CT_CO_train.txt',
                              non_covid_files=ORG_PATH + 'Data-split/NonCOVID/new_CT_NC_train.txt',
                              transform=train_transformer)
    testset = CovidCTDataset(root_dir=ORG_PATH,
                             classes=CLASSES,
                             covid_files=ORG_PATH + 'Data-split/COVID/new_CT_CO_test.txt',
                             non_covid_files=ORG_PATH + 'Data-split/NonCOVID/new_CT_NC_test.txt',
                             transform=val_transformer)
elif "COVIDx" in SET_NAME:
    trainset = CovidCTDataset(root_dir=CTX_PATH + '2A_images',
                              classes=CLASSES,
                              covid_files=CTX_PATH + 'co_train.txt',
                              non_covid_files=CTX_PATH + 'nc_train.txt',
                              transform=train_transformer,
                              is_ctx=True)
    testset = CovidCTDataset(root_dir=CTX_PATH + '2A_images',
                             classes=CLASSES,
                             covid_files=CTX_PATH + 'co_test.txt',
                             non_covid_files=CTX_PATH + 'nc_test.txt',
                             transform=val_transformer,
                             is_ctx=True)
# =============================================================

# =============================================================
# CREATE DATA-LOADERS FOR ALL SETS
# =============================================================
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False, num_workers=2)
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

    print("\n==========================================")
    print("SELECTED MODEL: \t", model_name)
    print("SELECTED DATASET: \t", SET_NAME)

    # RUNS ALL TRAINING AND TESTS.
    # =============================================================
    fit(model, train_loader, test_loader, optimizer, epochs=20, model_name=model_name)

    # DRAWS GRAPH PY-PLOT
    # =============================================================
    TRAIN_PATH = "./logs/train_logger/__" + model_name + "__run___training.log"
    TEST_PATH = "./logs/test_logger/__" + model_name + "__run___test.log"
    pretty_plot(TRAIN_PATH, TEST_PATH, model_name + "_plot")
