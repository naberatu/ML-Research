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
from tensorflow.python.keras import backend as K
from imc_dataset import image_dataset_from_directory
import tensorflow as tf
from tensorflow import lite
import tensorflow_model_optimization as tfmot
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical


# Models
from imc_nabernet import nabernet
from imc_compress import eval_imc_tfl

# Metrics
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

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
BATCH_SIZE = 8
EPOCHS = 10
# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.00001)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.0005)
# LOSS = 'categorical_crossentropy'
LOSS = 'binary_crossentropy'
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
# MODEL_NAME = "VGG16_" + suffix
# MODEL = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_shape=IM_SIZE + (3,), classes=2, classifier_activation="sigmoid")
# MODEL_NAME = "Resnet50_" + suffix
# MODEL = tf.keras.applications.resnet.ResNet50(include_top=False, weights=None, input_shape=IM_SIZE + (3,), classes=2)

# ===================================================
# STEP: Load Images & Labels into Dataset
# ===================================================
dataset = image_dataset_from_directory(
    dir_data,
    # label_mode='categorical',
    seed=1337,
    image_size=IM_SIZE,
    batch_size=BATCH_SIZE,
)

x = np.concatenate([x for x, _ in dataset], axis=0)
y = np.concatenate([y for _, y in dataset], axis=0)
x = np.array(x).astype('uint8')
y = np.array(y).astype('uint8')
NUM_IMGS = len(x)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, shuffle=SHUFFLE)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=SHUFFLE)

# y_train = to_categorical(y_train, num_classes=N_CLASSES)
# y_val = to_categorical(y_val, num_classes=N_CLASSES)
# y_test = to_categorical(y_test, num_classes=N_CLASSES)

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

for i, line in enumerate(text):
    if i == 0:
        print()
    print(line)

file = dir_models + MODEL_NAME + ".hdf5"

MODEL.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
MODEL.fit(x=x_train, y=y_train, verbose=VERBOSITY, epochs=EPOCHS, validation_data=(x_val, y_val), shuffle=SHUFFLE)
MODEL.save(file)
print("\nSaved Model to:\t\t", file)

# ===================================================
# EVAL: Image Classifier
# ===================================================
MODEL.load_weights(file)
MODEL.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])


def eval_imc(M_NAME='', MODEL=None, x=None, y=None):
    print("Evaluating Model:\t", M_NAME + "...", end='\t')

    _, accuracy = MODEL.evaluate(x, y)
    text.append("Model.Eval() Accuracy:\t " + "%.1f" % (accuracy * 100) + "%")
    predictions = np.argmax(MODEL.predict(x), axis=1)

    crep = metrics.classification_report(y_pred=predictions, y_true=y, target_names=CLASSES)
    print(crep)
    text.append("\n" + str(crep))
    text.append(divider)

    # Write results to file.
    filename = "imc_" + M_NAME + ".txt"
    with open(dir_metrics + filename, 'w') as f:
        f.write('\n'.join(text))
    f.close()

    text.clear()
    print("COMPLETE")


eval_imc(M_NAME=MODEL_NAME, MODEL=MODEL, x=x_test, y=y_test)

# =============================================================
# STEP: Begin Pruning UNet
# =============================================================
MODEL.load_weights(file)
MODEL.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

with open('metrics/imc_summary_' + MODEL_NAME + '.txt', 'w') as f:
    MODEL.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
prune_epochs = 2
validation_split = 0.1      # 10% of training set will be used for validation set.

num_images = len(x_train)
end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) * prune_epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                               final_sparsity=0.50,
                                                               begin_step=0,
                                                               end_step=end_step)
}
model_for_pruning = prune_low_magnitude(MODEL, **pruning_params)
model_for_pruning.compile(optimizer=OPTIMIZER,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=[tf.keras.metrics.MeanIoU(num_classes=N_CLASSES)])

# =============================================================
# STEP: Export Pruned Model to hdf5
# =============================================================
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save(dir_models + MODEL_NAME + "_pruned.hdf5")
with open('metrics/un_summary_pruned.txt', 'w') as f:
    model_for_export.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()
print('SAVED:\t\t Pruned Keras Model')

# EVAL: Re-Loaded Pruned Model (hdf5)
pruned_model = tf.keras.models.load_model(dir_models + MODEL_NAME + "_pruned.hdf5")
pruned_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
eval_imc(MODEL_NAME + "_pruned", MODEL=pruned_model, x=x_test, y=y_test)
print('EVALUATED:\t Pruned Keras Model')

# # =============================================================
# # STEP: Convert Pruned Model to TFlite
# # =============================================================
converter = lite.TFLiteConverter.from_keras_model(pruned_model)
pruned_tflite_model = converter.convert()

filename = dir_models + MODEL_NAME + '_pruned.tflite'
with open(filename, 'wb') as f:
    f.write(pruned_tflite_model)
f.close()
print('SAVED:\t\t Pruned TFLite Model')

# EVAL: Pruned TFLite File
# eval_imc_tfl()
# print("EVALUATED:\t Pruned TFLite Model")
#
# # =============================================================
# # STEP: Quantize Pruned File
# # =============================================================
converter = lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
prune_quant_tfl_model = converter.convert()                # Weights are quantized now.

filename = dir_models + MODEL_NAME + '_pq.tflite'
with open(filename, 'wb') as f:
    f.write(prune_quant_tfl_model)
f.close()
print("SAVED:\t\t Pruned & Quantized TFLite Model")

# EVAL: Pruned and Quantized File
interpreter = tf.lite.Interpreter(model_content=prune_quant_tfl_model)
interpreter.allocate_tensors()



test_accuracy = eval_imc_tfl(interpreter, )

print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
# print('Pruned TF test accuracy:', model_for_pruning_accuracy)
print("EVALUATED:\t Pruned & Quantized TFLite Model")


