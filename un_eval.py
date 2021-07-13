from tensorflow.keras.metrics import MeanIoU
from un_multi_model import multi_unet_model
import math
import numpy as np
import tensorflow as tf


def eval_unet(FNAME="", DATASET="", MODEL=None, BATCH=0, EPOCHS=0, CLASSES=None, NUM_IMS=0, IM_DIM=32, IM_CH=1,
         TEST_IMS=None, TEST_MASKS=None, PRINT=False):

    NUM_CLS = len(CLASSES)
    if MODEL is None:
        MODEL = multi_unet_model(n_classes=NUM_CLS, IMG_HEIGHT=IM_DIM, IMG_WIDTH=IM_DIM, IMG_CHANNELS=IM_CH)
    IOU_keras = MeanIoU(num_classes=NUM_CLS)

    ypred = MODEL.predict(TEST_IMS)
    ypred_argmax = np.argmax(ypred, axis=3)

    # Generates the confusion matrix.
    IOU_keras.update_state(TEST_MASKS[:, :, :, 0], ypred_argmax)

    text = ["=========================================",
            "Dataset: " + DATASET,
            "Num Images: " + str(NUM_IMS),
            "Image Size: " + str(IM_DIM) + "x" + str(IM_DIM),
            "Num Classes: " + str(NUM_CLS),
            "Batch Size: " + str(BATCH) if BATCH > 0 else None,
            "Epochs: " + str(EPOCHS) if EPOCHS > 0 else None,
            "=========================================",
            "Mean IoU = " + str(IOU_keras.result().numpy())
            ]
    while None in text:
        text.remove(None)

    # To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(NUM_CLS, NUM_CLS)

    # Store metrics
    TP, FP, FN, TN, IoU, Dice = [], [], [], [], [], []
    meanDice = 0
    for i in range(NUM_CLS):
        TP.append(values[i, i])
        fp, fn = 0, 0
        for k in range(NUM_CLS):
            if k != i:
                fp += values[i, k]
                fn += values[k, i]
        FP.append(fp)
        FN.append(fn)
        IoU.append(TP[i] / (TP[i] + FP[i] + FN[i]))
        text.append(CLASSES[i] + " IoU:\t\t " + str(IoU[i]))

        Dice.append((2 * TP[i]) / ((2 * TP[i]) + FP[i] + FN[i]))
        meanDice += Dice[i]

    text.append("-----------------------------------------")
    text.append("Mean Dice = " + str(meanDice / NUM_CLS))
    for i in range(NUM_CLS):
        text.append(CLASSES[i] + " Dice:\t\t " + str(Dice[i]))

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

    PATH = 'C:\\Users\\elite\\PycharmProjects\\Pytorch\\' + FNAME + '.txt'
    with open(PATH, 'w') as f:
        if PRINT:
            for line in text:
                print(line)
        f.writelines('\n'.join(text))
    f.close()


def eval_tfl(TFLModel, FNAME="", DATASET="", CLASSES=None, IM_SIZE=0, X_TEST=None, Y_TEST=None):
    N_CLASSES = len(CLASSES)

    interpreter = tf.lite.Interpreter(model_content=TFLModel)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    ypred = []
    for img in X_TEST:
        img = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, img)

        interpreter.invoke()

        output = interpreter.get_tensor(output_index)
        ypred.append(output[0])

    ypred_argmax = np.argmax(np.array(ypred), axis=3)
    IOU_keras = MeanIoU(num_classes=N_CLASSES)
    IOU_keras.update_state(Y_TEST[:, :, :, 0], ypred_argmax)

    text = ["=========================================",
            "Dataset: " + DATASET,
            "Image Size: " + str(IM_SIZE) + "x" + str(IM_SIZE),
            "Num Classes: " + str(N_CLASSES),
            "=========================================",
            "Mean IoU = " + str(IOU_keras.result().numpy())
            ]
    values = np.array(IOU_keras.get_weights()).reshape(N_CLASSES, N_CLASSES)

    # Store metrics
    TP, FP, FN, TN, IoU, Dice = [], [], [], [], [], []
    meanDice = 0
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

        Dice.append((2 * TP[i]) / ((2 * TP[i]) + FP[i] + FN[i]))
        meanDice += Dice[i]

    text.append("-----------------------------------------")
    text.append("Mean Dice = " + str(meanDice / N_CLASSES))
    for i in range(N_CLASSES):
        text.append(CLASSES[i] + " Dice:\t\t " + str(Dice[i]))

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

    PATH = 'C:\\Users\\elite\\PycharmProjects\\Pytorch\\' + FNAME + '.txt'
    with open(PATH, 'w') as f:
        f.writelines('\n'.join(text))
    f.close()
