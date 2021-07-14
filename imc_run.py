# ===================================================
# STEP: Imports
# ===================================================
# General imports
import os
import sys
import warnings
import logging
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow logging suppressor
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Tensorflow imports
from imc_dataset import image_dataset_from_directory
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.keras import backend as K

# Models
from imc_nabernet import nabernet

# Metrics
from sklearn.metrics import confusion_matrix

# NOTE: Establish GPU as device-to-use
DEVICE = '/physical_device:GPU:0'
config = tf_v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
K.set_session(tf_v1.Session(config=config))

# ===================================================
# STEP: Model Parameters
# ===================================================
dir_models = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\models\\"
dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\"
CLASSES = []

# SELECT: An Image Dataset
SET_NAME = "UCSD AI4H"      # Contains 746 images.
# SET_NAME = "SARS-COV-2"     # Contains 2,481 images.
# SET_NAME = "COVIDx CT-1"    # Contains 115,837 images.

# SELECT: Training & Testing Parameters
IM_SIZE = (300, 300)
BATCH_SIZE = 8
VAL_SPLIT = 0.2
EPOCHS = 10
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
VERBOSITY = 2
# SHUFFLE = True
SHUFFLE = False
suffix = ""

# Automatic dataset parameter assignment
if "ucsd" in SET_NAME.lower():
    CLASSES = ["UCSD_CO", "UCSD_NC"]
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_ucsd\\"
    suffix = "ucsd"
elif "sars" in SET_NAME.lower():
    CLASSES = ['SARSCT_NC', 'SARSCT_CO']
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_sars2\\"
    suffix = "sars2"
elif "x" in SET_NAME.lower():
    CLASSES = ['COVID', 'NONCOV']
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_ctx\\Keras_Split\\"
    suffix = "ctx"


# SELECT: A Keras Model
# =========================
N_CLASSES = len(CLASSES)
MODEL_NAME = "NaberNet_" + suffix
MODEL = nabernet(n_classes=N_CLASSES, im_size=IM_SIZE)

# ===================================================
# STEP: Load Images & Labels into Dataset
# ===================================================
train_ds = image_dataset_from_directory(
    dir_data,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=1337,
    image_size=IM_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = image_dataset_from_directory(
    dir_data,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=1337,
    image_size=IM_SIZE,
    batch_size=BATCH_SIZE,
)
NUM_IMGS = len(train_ds.__dict__['file_paths']) + len(val_ds.__dict__['file_paths'])

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(BATCH_SIZE):
#         ax = plt.subplot(2, 4, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")
# plt.show()

# ===================================================
# STEP: Begin Fitting
# ===================================================
text = [
    "==========================================",
    "Dataset Name:\t " + SET_NAME,
    "Model Name:\t\t " + MODEL_NAME.split('_')[0],
    "Classes, Batch:\t " + str(N_CLASSES) + ', ' + str(BATCH_SIZE),
    "Image Size:\t\t " + "{:,}".format(IM_SIZE[0]) + "x" + "{:,}".format(IM_SIZE[1]),
    "Num Images:\t\t " + "{:,}".format(NUM_IMGS),
    "Num Epochs:\t\t " + "{:,}".format(EPOCHS),
    "=========================================="
]
print()
for line in text:
    print(line)

MODEL.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
MODEL.fit(train_ds, verbose=VERBOSITY, epochs=EPOCHS, validation_data=val_ds, shuffle=SHUFFLE)
MODEL.save(dir_models + MODEL_NAME + ".hdf5")

# EVAL: Image Classifier
MODEL.load_weights(dir_models + MODEL_NAME + ".hdf5")
# loss, accuracy = MODEL.evaluate(val_ds)

# model_acc = 'Accuracy: %.2f' % (accuracy * 100) + "%"
# print(model_acc)
# print('Loss:', loss)

