
import numpy as np


def eval_imc_tfl(MNAME='', suffix='', mode='w', dir='', divider='', interpreter=None, test_images=None, test_labels=None):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the test dataset.
    prediction_digits = []
    for img in test_images:
        # NOTE: Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        img = np.expand_dims(img, axis=0).astype(np.float32)
        # img = np.array(img).astype('float32')
        interpreter.set_tensor(input_index, img)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()

    text = [
      MNAME + suffix + " (TFL) Accuracy:\t " + "%.1f" % (accuracy * 100) + "%",
      divider + "\n",
    ]

    with open(dir + MNAME + ".txt", mode) as f:
        f.write('\n'.join(text))
    f.close()
