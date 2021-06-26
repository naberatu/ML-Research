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
import glob
import os
import logging
import tifffile
import random
import sys
import warnings

warnings.filterwarnings("ignore")
# random.seed(12)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# General imports
import cv2
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
# Globals
# ===================================================

# Directories
dir_medseg = './data/MedSeg/'
dir_img = './data/MedSeg/tr_ims/'
dir_mask = './data/MedSeg/tr_masks/'
dir_mask2 = './data/MedSeg/tr_masks_orig/'
dir_test = './data/MedSeg/val_ims/'
dir_checkpoint = 'checkpoints/'
dir_sandstone = 'C:/Users/elite/Desktop/sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/'

# Other global variables
DEVICE = '/physical_device:GPU:0'
config = tf_v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
K.set_session(tf_v1.Session(config=config))

IM_SIZE = 256       # 256 x 256 square images.
# IM_SIZE = 128       # 128 x 128 patch images.
CLASSES = 4         # Background (0) + 3 classes (1-3).

# Read images & other files from Tiff.
TRAIN_IMAGS = np.array(tifffile.imread(dir_medseg + "tr_ims.tif")).astype(np.int8)
TRAIN_MASKS = np.array(tifffile.imread(dir_medseg + "masks.tif")).astype(np.int8)

# for directory_path in glob.glob(dir_mask2):     # NOTE: Update this back to dir_mask1
    # for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         img = cv2.imread(img_path, 0)
#         img = cv2.resize(img, (IM_SIZE, IM_SIZE))
#         # Code for saving new masks.
#         img[img <= 55] = 0
#         img[img > 210] = 2
#         img[img > 140] = 1
#         img[img > 55] = 3
#         print("Image", counter, ":", np.unique(img))
#         name = dir_mask + "mask" + str(counter) + ".jpg"
#         cv2.imwrite(name, img)
#         counter += 1
#         # ============================================
#         # TRAIN_MASKS.append(img)
# # TRAIN_MASKS = np.array(TRAIN_MASKS)
# TRAIN_MASKS = np.array(nib)
# sys.exit()

# USED FOR SANDSTONE IMAGES.
# # Retrieve Images as RGB files (0 - 255)
# for directory_path in glob.glob(dir_sandstone + "images/"):
#     for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         img = cv2.imread(img_path, 0)
#         TRAIN_IMAGS.append(img)
# TRAIN_IMAGS = np.array(TRAIN_IMAGS)
#
# # Retrieve Masks as RGB files (0 - 255)
# for directory_path in glob.glob(dir_sandstone + "masks/"):
#     for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         img = cv2.imread(img_path, 0)
#         TRAIN_MASKS.append(img)
# TRAIN_MASKS = np.array(TRAIN_MASKS)

# Assign labels
labeler = LabelEncoder()
n, h, w = TRAIN_MASKS.shape
reshaped_masks = TRAIN_MASKS.reshape(-1, 1)
encoded_masks = labeler.fit_transform(reshaped_masks)
updated_masks = encoded_masks.reshape(n, h, w)

print(np.unique(updated_masks))

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

# model = multi_unet_model(n_classes=CLASSES, IMG_HEIGHT=IM_HT, IMG_WIDTH=IM_WD, IMG_CHANNELS=IM_CH)
model = multi_unet_model(n_classes=CLASSES, IMG_HEIGHT=IM_HT, IMG_WIDTH=IM_WD, IMG_CHANNELS=IM_CH)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=[keras.metrics.MeanIoU(num_classes=CLASSES)])
# model.summary()

# =============================================================
# NOTE: Model-fitting parameters.
# =============================================================
history = model.fit(x_train, y_train_cat, batch_size=8, verbose=1, epochs=20, validation_data=(x_test, y_test_cat),
                    class_weight=weights, shuffle=False)
model.save("test.hdf5")

# Plot model history
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Model Evaluation
model.load_weights("test.hdf5")
ypred = model.predict(x_test)
ypred_argmax = np.argmax(ypred, axis=3)

# Calculate IoU
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=CLASSES)

# Generates the confusion matrix.
IOU_keras.update_state(y_test[:, :, :, 0], ypred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(CLASSES, CLASSES)
print(values)

# Store metrics
TP, FP, FN, IoU = [], [], [], []
for i in range(CLASSES):
    TP.append(values[i, i])
    fp, fn = 0, 0
    for k in range(CLASSES):
        if k != i:
            fp += values[i, k]
            fn += values[k, i]
    FP.append(fp)
    FN.append(fn)
    IoU.append(TP[i] / (TP[i] + FP[i] + FN[i]))
    print("Class", i, "IoU: ", IoU[i])

# plt.imshow(TRAIN_IMAGS[0, :, :, 0], cmap='gray')
# plt.imshow(TRAIN_MASKS[0], cmap='gray')

#######################################################################
# Predict on a few images
# model = get_model()
# model.load_weights('???.hdf5')
test_img_number = random.randint(0, len(x_test))
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
