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
import random
import warnings

# General imports
import cv2
import matplotlib.pyplot as plt
import torch
from un_multi_model import multi_unet_model

# Keras, numpy, and LabelEncoder
import numpy as np
from keras.utils import normalize
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import sklearn.metrics as metrics

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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Other global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IM_SIZE = 256   # 256 x 265 square images.
CLASSES = 4     # Background + 3 classes.       NOTE: 0 Should represent background/unlabeled.
# CLASSES = 248     # Used for CT Scan masks which go up to 247 for some reason.
TRAIN_IMAGS = []
TRAIN_MASKS = []

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

# Categorize by one-hot encoding.
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
model.to(DEVICE)
model.summary()

history = model.fit(x_train, y_train_cat, batch_size=8, verbose=1, epochs=5, validation_data=(x_test, y_test_cat), shuffle=False)
model.save("test.pth")

# Plot model history
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

# Model Evaluation
ypred = model.predict(x_test)
ypred_argmax = np.argmax(ypred, axis=3)
metrics.confusion_matrix(y_test[:, :, :, 0], ypred_argmax, labels=[0, 1, 2, 3])

print(metrics.classification_report)
print()