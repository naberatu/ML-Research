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
# NOTE: Imports
# ===================================================
# OS & Environment setup
import math
import os
import logging
import tifffile
import random
import warnings
import sys

# random.seed(12)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# General imports
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf_v1
from un_multi_model import multi_unet_model
import numpy as np

# Keras and LabelEncoder
from tensorflow.python.keras import backend as K
from keras.utils import normalize
from keras.utils import to_categorical
import keras.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ===================================================
# NOTE: Globals
# ===================================================
# Directories
dir_medseg = './data/MedSeg/'
dir_sandstone = 'C:/Users/elite/Desktop/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/'

# Other global variables
DEVICE = '/physical_device:GPU:0'
config = tf_v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
K.set_session(tf_v1.Session(config=config))

# =============================================================
# NOTE: Model Parameters
# =============================================================

# Read from TIFF images (MedSeg).
DATASET = "MedSeg"
TRAIN_IMAGS = np.array(tifffile.imread(dir_medseg + "tr_ims.tif")).astype(np.int8)
TRAIN_MASKS = np.array(tifffile.imread(dir_medseg + "masks.tif")).astype(np.int8)
IM_SIZE = 512
# IM_SIZE = 256
CLASSES = ["Backgnd/Misc", 'Ground Glass', 'Consolidation', 'Pleural Eff.']

# Read from TIFF images (Sandstone).
# DATASET = "Sandstone"
# TRAIN_IMAGS = np.array(tifffile.imread(dir_sandstone + "images.tiff")).astype(np.int8)
# TRAIN_MASKS = np.array(tifffile.imread(dir_sandstone + "masks.tiff")).astype(np.int8)
# IM_SIZE = 128       # Due to 128 x 128 patch images.
# CLASSES = ["Backgd", 'Clay', 'Quartz', 'Pyrite']

N_CLASSES = len(CLASSES)
EPOCHS = 100
BATCH_SIZE = 8      # Selected for RTX 2060
# VERBOSITY = 1       # Progress Bar
VERBOSITY = 2       # One Line/Epoch
# SHUFFLE = True
SHUFFLE = False

# =============================================================
# NOTE: Encoding & Pre-processing.
# =============================================================

# Assign labels & encode them.
labeler = LabelEncoder()
n, h, w = TRAIN_MASKS.shape
reshaped_masks = TRAIN_MASKS.reshape(-1, 1)
encoded_masks = labeler.fit_transform(reshaped_masks)
updated_masks = encoded_masks.reshape(n, h, w)
print(np.unique(updated_masks))

# Prepare training datasets.
TRAIN_IMAGS = np.expand_dims(TRAIN_IMAGS, axis=3)
TRAIN_IMAGS = normalize(TRAIN_IMAGS, axis=1)
input_masks = np.expand_dims(updated_masks, axis=3)

# Create training & testing datasets.
N_TEST = 0.1
N_TRAIN = 0.2
# x1, x_test, y1, y_test = train_test_split(TRAIN_IMAGS, input_masks, test_size=N_TEST, random_state=0)
# x_train, _, y_train, _ = train_test_split(x1, y1, test_size=N_TRAIN, random_state=0)    # Extra split for quick testing.
x_train, x_test, y_train, y_test = train_test_split(TRAIN_IMAGS, input_masks, test_size=N_TEST, random_state=0)

# Sanity check
print("Class values in the dataset are ... ", np.unique(y_train))

# Categorize by one-hot encoding.
masks_cat_train = to_categorical(y_train, num_classes=N_CLASSES)
y_train_cat = masks_cat_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], N_CLASSES))
masks_cat_test = to_categorical(y_test, num_classes=N_CLASSES)
y_test_cat = masks_cat_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], N_CLASSES))

# Calculate class weights.
weights = class_weight.compute_class_weight('balanced', np.unique(encoded_masks), encoded_masks)
print("Class weights are...:", weights)

IM_HT = x_train.shape[1]
IM_WD = x_train.shape[2]
IM_CH = x_train.shape[3]

model = multi_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IM_HT, IMG_WIDTH=IM_WD, IMG_CHANNELS=IM_CH)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[keras.metrics.MeanIoU(num_classes=N_CLASSES)])
# model.summary()

# =============================================================
# NOTE: Model-fitting.
# =============================================================
history = model.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, verbose=VERBOSITY, epochs=EPOCHS,
                    validation_data=(x_test, y_test_cat), class_weight=weights, shuffle=SHUFFLE)
model.save("test.hdf5")

# Model Evaluation
model.load_weights("test.hdf5")
ypred = model.predict(x_test)
ypred_argmax = np.argmax(ypred, axis=3)

# =============================================================
# NOTE: Metrics & Evaluation.
# =============================================================
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=N_CLASSES)

# Generates the confusion matrix.
IOU_keras.update_state(y_test[:, :, :, 0], ypred_argmax)

text = ["=========================================",
        "Dataset: " + DATASET, "Num Classes: " + str(N_CLASSES), "Epochs: " + str(EPOCHS),
        "=========================================",
        "Mean IoU = " + str(IOU_keras.result().numpy())]

# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(N_CLASSES, N_CLASSES)

# Store metrics
TP, FP, FN, TN, IoU = [], [], [], [], []
for i in range(N_CLASSES):
    TP.append(values[i, i])
    fp, fn = 0, 0
    for k in range(N_CLASSES):
        if k != i:
            fp += values[i, k]
            fn += values[k, i]
    FP.append(fp)
    FN.append(fn)
    IoU.append(TP[i] / (TP[i] + FP[i] + FN[i]))
    text.append(CLASSES[i] + " IoU:\t\t " + str(IoU[i]))

text.append("=========================================")
num_vals = len(TP)
for i in range(num_vals):
    if i > 0:
        text.append("-----------------------------------------")
    negatives = 0
    for j in range(num_vals):
        if j != i:
            for k in range(num_vals):
                if k != i:
                    negatives += values[j, k]
    TN.append(negatives)

    # Final metrics.
    sensitivity = TP[i] / max((TP[i] + FN[i]), 1)
    specificity = TN[i] / max((TN[i] + FP[i]), 1)
    precision = TP[i] / max((TP[i] + FP[i]), 1)
    gmean = math.sqrt(max(sensitivity * specificity, 0))
    f2_score = (5 * precision * sensitivity) / max(((4 * precision) + sensitivity), 1)
    text.append("For Class: \t\t" + CLASSES[i] + "...")
    text.append("Sensitivity: \t" + str(sensitivity))
    text.append("Specificity: \t" + str(specificity))
    text.append("Precision: \t\t" + str(precision))
    text.append("G-Mean Score: \t" + str(gmean))
    text.append("F2-Score: \t\t" + str(f2_score))

text.append("=========================================")

with open('C:\\Users\\elite\\PycharmProjects\\Pytorch\\un_metrics.txt', 'w') as f:
    for line in text:
        print(line)
    f.writelines('\n'.join(text))
f.close()
# plt.imshow(TRAIN_IMAGS[0, :, :, 0], cmap='gray')
# plt.imshow(TRAIN_MASKS[0], cmap='gray')

# =============================================================
# NOTE: Display Prediction.
# =============================================================
# model = get_model()
# model.load_weights('???.hdf5')

# Predict on a few images
test_img_number = random.randint(0, len(x_test) - 1)
test_img = x_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0][:, :, None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()
