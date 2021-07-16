
import numpy as np
import tensorflow as tf


def eval_imc(name='', suffix='', eval_file='', divider='\n', model=None, x=None, y=None):
    print("Evaluating:\t", name + suffix + "...", end='\t')

    _, accuracy = model.evaluate(x, y, verbose=0)
    text = ['\n' + name + suffix + " Accuracy:\t\t " + "%.1f" % (accuracy * 100) + "%", divider]

    with open(eval_file, 'a') as f:
        f.write('\n'.join(text))
    f.close()

    print("COMPLETE")


def eval_imc_tfl(name='', suffix='', eval_file='', divider='', model=None, test_images=None, test_labels=None):
    print("Evaluating:\t", name + suffix + "...", end="\t")

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the test dataset.
    prediction_digits = []
    for img in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        img = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, img)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()

    text = ['\n' + name + suffix + " TFL Accuracy:\t " + "%.1f" % (accuracy * 100) + "%", divider]

    with open(eval_file, 'a') as f:
        f.write('\n'.join(text))
    f.close()

    print("COMPLETE")
