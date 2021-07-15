# ===================================================
# STEP: Imports
# ===================================================
# General imports
import os
import random
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
import sklearn.metrics as metrics

# NOTE: Establish GPU as device-to-use
DEVICE = '/physical_device:GPU:0'
config = tf_v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
K.set_session(tf_v1.Session(config=config))

# ===================================================
# STEP: Model Parameters
# ===================================================
dir_models = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\models\\"
dir_metrics = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\metrics\\"
dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\"
divider = "================================================================="
CLASSES = []

# SELECT: An Image Dataset
SET_NAME = "UCSD AI4H"      # Contains 746 images.
# SET_NAME = "SARS-COV-2"     # Contains 2,481 images.
# SET_NAME = "COVIDx CT-1"    # Contains 115,837 images.

# SELECT: Training & Testing Parameters
IM_SIZE = (300, 300)
BATCH_SIZE = 16
VAL_SPLIT = 0.2
EPOCHS = 10
# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.00001)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.0001)
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

# ===================================================
# STEP: Compile, Fit, and Save Model
# ===================================================
text = [
    divider,
    "Dataset Name:\t\t " + SET_NAME,
    "Model Name:\t\t\t " + MODEL_NAME.split('_')[0],
    "Classes, Batch:\t\t " + str(N_CLASSES) + ', ' + str(BATCH_SIZE),
    "Image Size:\t\t\t " + "{:,}".format(IM_SIZE[0]) + "x" + "{:,}".format(IM_SIZE[1]),
    "Num Images:\t\t\t " + "{:,}".format(NUM_IMGS),
    "Num Epochs:\t\t\t " + "{:,}".format(EPOCHS),
    divider
]
print()
for line in text:
    print(line)

MODEL.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
MODEL.fit(train_ds, verbose=VERBOSITY, epochs=EPOCHS, validation_data=val_ds, shuffle=SHUFFLE)
file = dir_models + MODEL_NAME + ".hdf5"
MODEL.save(file)
print()
print("Saved Model to:\t\t", file)

# ===================================================
# EVAL: Image Classifier
# ===================================================
MODEL.load_weights(dir_models + MODEL_NAME + ".hdf5")
MODEL.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

print("Evaluating Model:\t", MODEL_NAME + "...", end='\t')

# x_test = np.concatenate([x for x, y in val_ds], axis=0)
# y_test = np.concatenate([y for x, y in val_ds], axis=0)

_, accuracy = MODEL.evaluate(val_ds)
# _, accuracy = MODEL.evaluate(x_test, y_test)
text.append("Model.Eval() Accuracy:\t " + "%.1f" % (accuracy * 100) + "%")

# y_pred = MODEL.predict(x_test)
# y_pred = MODEL.predict(val_ds)
# y_pred_argmax = np.argmax(y_pred, axis=1)

# crep = metrics.classification_report(y_true, y_pred.rround(), target_names=CLASSES)
# crep = metrics.classification_report(y_test, y_pred_argmax, target_names=CLASSES)
#
# text.append(crep)
text.append(divider)

# NOTE: Write results to file.
filename = "imc_" + MODEL_NAME + ".txt"
with open(dir_metrics + filename, 'w') as f:
    f.write('\n'.join(text))
f.close()

print("COMPLETE")

# _, accuracy = MODEL.evaluate(val_ds)
# print('Accuracy: %.2f' % (accuracy * 100) + "%")

