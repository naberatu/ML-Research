# ===================================================
# STEP: Imports
# ===================================================
# General imports
import os
import warnings
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow logging suppressor
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===================================================
# STEP: Model Parameters
# ===================================================
CLASSES = ["COVID", "NONCOV"]
SET_NAME = "COVIDx CT-1"
N_CLASSES = len(CLASSES)

IM_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20

dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\CTX\\Keras_Split\\"
DATASET = []
text = [
    "Dataset Name:\t " + SET_NAME,
    "Num Classes:\t " + str(N_CLASSES),
    "Image Size:\t\t " + str(IM_SIZE[0]) + "x" + str(IM_SIZE[1]),
    "Num Images:\t\t " + str(len(DATASET)),
    "Batch Size:\t\t " + str(BATCH_SIZE),
    "Num Epochs:\t\t " + str(EPOCHS)
]
print()
print("==========================================")
for line in text:
    print(line)
print("==========================================")

text.clear()

# ===================================================
# STEP: Prepare Dataset
# ===================================================

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_data,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IM_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_data,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IM_SIZE,
    batch_size=BATCH_SIZE,
)

# for data_class in CLASSES:
#     folder = os.path.join(dir_data, data_class)
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             DATASET.append([img, data_class])

# DATASET = np.array(DATASET)
# plt.imshow(DATASET[0][0])
# plt.title(DATASET[0][1])
# plt.show()

