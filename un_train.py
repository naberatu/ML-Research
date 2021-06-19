"""
AUTHOR:             Nader K. Atout
FACULTY ADVISOR:    Dr. Nader Bagherzadeh
GRADUATE MENTOR:    Mohammed Alnemari
PURPOSE:            UCI CHC Undergaduate Research Thesis (2020-2021)
TOPIC:              Using a Multiclass Segmentation model (U-Net) to determine the presence & severity of COVID-19
                        in human lungs, and compressing the model for low-resource devices.
REFERENCES:
    Medical Segmentation        (For provided Dataset ~ https://medicalsegmentation.com/covid19/)
    Rudolph Peinaar             (For Med2Image ~ Converting .NII files into .JPG)
    GitHub user Milesial        (Training, Testing, and logging routine templates ~ https://github.com/milesial)
    @PtrBlk & Pytorch forums    (For help with color-maps, & dataset setup/processing)
    Dr. Sreenivas Bhattiprolu   (YouTube: 208 Multiclass semantic segmentation with U-Net)
"""

# ===================================================
# Imports
# ===================================================

# OS & Environment setup
import argparse
import glob
import logging
import os
import sys
import random
import warnings

# Torch, tqdm, optim & tensorboard
import cv2
import keras
import matplotlib.pyplot as plt
import nibabel
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms
import tensorflow as tf

# Header files / Helpers
from un_eval import eval_net
from un_eval import fit_tensor
from un_model import UNet
from un_dataset import SegSet
from un_multi_model import multi_unet_model

# Keras, numpy, and LabelEncoder
import numpy as np
import nibabel as nib
from keras import backend as K
from keras.utils import normalize
from keras.utils import to_categorical
from keras.metrics import TruePositives
from keras.metrics import TrueNegatives
from keras.metrics import FalseNegatives
from keras.metrics import FalsePositives
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ===================================================
# Globals
# ===================================================
warnings.filterwarnings("ignore")
random.seed(12)

# Directories
dir_medseg = './data/MedSeg/'
dir_img = './data/MedSeg/tr_ims/'
dir_mask = './data/MedSeg/tr_masks/'
dir_test = './data/MedSeg/val_ims/'
dir_checkpoint = 'checkpoints/'

dir_sandstone = "C:/Users/elite/Desktop/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches"

# Other global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IM_SIZE = 256   # 256 x 265 square images.
# CLASSES = 4     # Background + 3 classes.       NOTE: 0 Should represent background/unlabeled.
CLASSES = 248     # Used for CT Scan masks which go up to 247 for some reason.
TRAIN_IMAGS = []
TRAIN_MASKS = []

tf.device(DEVICE)

# Retrieve Images as RGB files (0 - 255)
for directory_path in glob.glob(dir_img):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        TRAIN_IMAGS.append(cv2.resize(img, (IM_SIZE, IM_SIZE)))
TRAIN_IMAGS = np.array(TRAIN_IMAGS)

# Retrieve Masks as RGB files (0 - 255)
for directory_path in glob.glob(dir_mask):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (IM_SIZE, IM_SIZE))
        img[img <= 55] = 0
        img[img > 210] = 2
        img[img > 140] = 1
        img[img > 55] = 3
        TRAIN_MASKS.append(img)
TRAIN_MASKS = np.array(TRAIN_MASKS)

# Assign labels
labeler = LabelEncoder()
n, h, w = TRAIN_MASKS.shape
reshaped_masks = TRAIN_MASKS.reshape(-1, 1)
encoded_masks = labeler.fit_transform(reshaped_masks)
updated_masks = encoded_masks.reshape(n, h, w)

np.unique(updated_masks)

# Prepare train_set
TRAIN_IMAGS = np.expand_dims(TRAIN_IMAGS, axis=3)
TRAIN_IMAGS = normalize(TRAIN_IMAGS, axis=1)
input_masks = np.expand_dims(updated_masks, axis=3)

# Create Dataset
N_TEST = 0.1
N_TRAIN = 0.2
x1, x_test, y1, y_test = train_test_split(TRAIN_IMAGS, input_masks, test_size=N_TEST, random_state=0)
x_train, _, y_train, _ = train_test_split(x1, y1, test_size=N_TRAIN, random_state=0)    # Extra split for quick testing.

# Sanity check
print("Class values in the dataset are ... ", np.unique(y_train))

# Categorize
masks_cat_train = to_categorical(y_train, num_classes=CLASSES)
y_train_cat = masks_cat_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], CLASSES))
masks_cat_test = to_categorical(y_test, num_classes=CLASSES)
y_test_cat = masks_cat_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], CLASSES))

# Class weights
weights = class_weight.compute_class_weight('balanced', np.unique(encoded_masks), encoded_masks)
print("Class weights are...:", weights)

IM_HT = x_train.shape[1]
IM_WD = x_train.shape[2]
IM_CH = x_train.shape[3]

model = multi_unet_model(CLASSES, IM_HT, IM_WD, IM_CH)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train_cat, batch_size=8, verbose=1, epochs=5, validation_data=(x_test, y_test_cat), shuffle=False)
# model.save("test.pth")

# Plot model history
_, acc = model.evaluate(x_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# acc = history.history['acc']
val_acc = history.history['val_acc']

# plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Model Evaluation
ypred = model.predict(x_test)
ypred_argmax = np.argmax(ypred, axis=3)
TP = TruePositives.update_state(y_test[:, :, :, 0], ypred_argmax)
TN = TrueNegatives.update_state(y_test[:, :, :, 0], ypred_argmax)
FP = FalsePositives.update_state(y_test[:, :, :, 0], ypred_argmax)
FN = FalseNegatives.update_state(y_test[:, :, :, 0], ypred_argmax)

print("TruePositives: ", TP.numpy())
print("TrueNegatives: ", TN.numpy())
print("FalsePositives: ", FP.numpy())
print("FalseNegatives: ", FN.numpy())

print()




# ===================================================
# Global Methods
# ===================================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ===================================================
# Training routine
# ===================================================
def train_net(net, device, epochs=5, batch_size=8, lr=0.001, val_percent=0.1, save_cp=True, img_scale=0.5):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0., 1.)
    ])

    trainset = SegSet(dir_img, dir_mask, img_scale, trs=transform)

    n_val = int(len(trainset) * val_percent)
    n_train = len(trainset) - n_val
    train, val = random_split(trainset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if OUT_CH > 1 else 'max', patience=2)
    if OUT_CH > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        logs = ["Cross Entropy: \t", "Dice Coeff: \t", "Sensitivity: \t", "Specificity: \t", "Precision: \t\t",
                "G-Mean: \t\t", "F2 Score: \t\t"]
        metrics = [0 for i in range(len(logs))]
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == IN_CH, \
                    f'Network has been defined with {IN_CH} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float)
                mask_type = torch.float if OUT_CH == 1 else torch.long        # will use Long for us

                true_masks = true_masks.to(device=device, dtype=mask_type)
                masks_pred = net(imgs)

                # Enforces 0 <= t < OUT_CH
                fit_tensor(masks_pred, true_masks, OUT_CH)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    # val_score = eval_net(net, val_loader, device)
                    # NOTE: Evaluation
                    evals = eval_net(net, val_loader, device)
                    val_score = evals[0]
                    for i in range(len(metrics)):
                        metrics[i] += evals[i]

                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if OUT_CH > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if OUT_CH == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')

            # TODO: Figure out why specificity & precision are low, despite good dice.
            #  Maybe it has something to do with the FN calculation?
            for i, m in enumerate(metrics):
                logs[i] += str(m / batch_size)

            with open('C:\\Users\\elite\\PycharmProjects\\Pytorch\\un_metrics.txt', 'w') as f:
                f.writelines('\n'.join(logs))
            f.close()

            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


# ===================================================
# Argument reader/parser
# ===================================================
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


# ===================================================
# NOTE Main Function
# ===================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    # =================================
    # NOTE Model starts here.
    # =================================
    IN_CH, OUT_CH = 3, 3

    # bilinear = False
    # net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                      in_channels=IN_CH, out_channels=OUT_CH, init_features=32, pretrained=False)
    net = UNet(n_channels=IN_CH, n_classes=OUT_CH)
    logging.info(f'Network:\n'
                 f'\t{IN_CH} input channels\n'
                 f'\t{OUT_CH} output channels (classes)\n'
                 # f'\t{"Bilinear" if bilinear else "Transposed conv"} upscaling')
                 f'\t{"Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device,
                  img_scale=args.scale, val_percent=args.val / 100)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
