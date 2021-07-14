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
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from imc_dataset import image_dataset_from_directory

# Models
from imc_nabernet import nabernet

# Metrics
from sklearn.model_selection import train_test_split

# ===================================================
# STEP: Model Parameters
# ===================================================
dir_models = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\models\\"
dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\"
CLASSES = []

# SELECT: An Image Dataset
SET_NAME = "ucsd"     # Contains 746 images.        (UCSD AI4H)
# SET_NAME = "sars"     # Contains 2,481 images.      (SARS-COV-2 CT-SCAN)
# SET_NAME = "ctx"      # Contains 115,837 images.    (COVIDx CT-1)

# SELECT: Training & Testing Parameters
IM_SIZE = (300, 300)
BATCH_SIZE = 8
VAL_SPLIT = 0.2
EPOCHS = 20
OPTIMIZER = 'adam'
VERBOSITY = 2
SHUFFLE = True
# SHUFFLE = False

# Automatic dataset parameter assignment
if "ucsd" in SET_NAME:
    CLASSES = ["UCSD_CO", "UCSD_NC"]
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_ucsd\\"
elif "sars" in SET_NAME:
    CLASSES = ['SARSCT_NC', 'SARSCT_CO']
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_sars2\\"
elif "ctx" in SET_NAME:
    CLASSES = ['COVID', 'NONCOV']
    dir_data = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\data\\ct_ctx\\Keras_Split\\"

# SELECT: A Keras Model
# =========================
N_CLASSES = len(CLASSES)
MODEL_NAME = "NaberNet_" + SET_NAME
MODEL = nabernet(n_classes=N_CLASSES, im_size=IM_SIZE)

# ===================================================
# STEP: Load Images & Labels into Dataset
# ===================================================
# IMAGES, LABELS = [], []
# for cls_index, im_class in enumerate(CLASSES):
#     folder = os.path.join(dir_data, im_class)
#     for filename in os.listdir(folder):
#         img = load_img(os.path.join(folder, filename), target_size=IM_SIZE)
#         img = img_to_array(img).astype('uint8')
#         IMAGES.append(img)
#         LABELS.append(cls_index)
# IMAGES = np.array(IMAGES)
# LABELS = np.array(LABELS).astype('uint8')

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

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

# ===================================================
# STEP: Begin Fitting
# ===================================================
text = [
    "Dataset Name:\t " + SET_NAME,
    "Num Classes:\t " + str(N_CLASSES),
    "Image Size:\t\t " + str(IM_SIZE[0]) + "x" + str(IM_SIZE[1]),
    "Num Images:\t\t " + str(len(train_ds)),
    "Batch Size:\t\t " + str(BATCH_SIZE),
    "Num Epochs:\t\t " + str(EPOCHS)
]
print()
print("==========================================")
for line in text:
    print(line)
print("==========================================")

text.clear()

# NOTE: Remove this line
sys.exit(0)

y_train = to_categorical(y_train, N_CLASSES)
y_test = to_categorical(y_test, N_CLASSES)

MODEL.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
history = MODEL.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSITY, epochs=EPOCHS,
                    validation_data=(x_test, y_test), shuffle=SHUFFLE)
MODEL.save(dir_models + MODEL_NAME + ".hdf5")

# EVAL: Image Classifier
MODEL.load_weights(dir_models + MODEL_NAME + ".hdf5")
_, accuracy = MODEL.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))

