
import torch
import torchvision
import pandas as pd
import numpy as np
from fastai.vision import *
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
import click
import numpy as np
import cv2
import imutils
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


PATH = Path('./data/celeba/faces/training')
IMG = ''


# def detect_facial_attributes(input_path, output_path, save_video):
if __name__ == '__main__':

    # ==========================================================================
    # Training
    # ==========================================================================

    # Function to filter validation samples
    def validation_func(x):
        return 'validation' in x


    tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=30, max_lighting=0.3)

    src = (ImageList.from_csv(PATH, csv_name='labels.csv')
           .split_by_valid_func(validation_func)
           .label_from_df(cols='tags', label_delim=' '))

    data = (src.transform(tfms, size=128).databunch(bs=256, num_workers=0).normalize(imagenet_stats))

    print('Number of Classes: ', data.c, '\n', data.classes, '\n')

    # Training Algorithm
    data.show_batch(rows=2, figsize=(12, 12))

    print('Initializing Resnet...', end=''),
    arch = models.resnet50
    acc_02 = partial(accuracy_thresh, thresh=0.2)
    acc_03 = partial(accuracy_thresh, thresh=0.3)
    acc_04 = partial(accuracy_thresh, thresh=0.4)
    acc_05 = partial(accuracy_thresh, thresh=0.5)
    f_score = partial(fbeta, thresh=0.2)
    learn = cnn_learner(data, arch, metrics=[acc_02, acc_03, acc_04, acc_05, f_score])
    print('\tDONE')

    # learn.lr_find()
    # learn.recorder.plot()

    print('Learning Rate...', end='')
    lr = 1e-2
    # learn.fit_one_cycle(1, slice(lr))
    learn.fit(4, slice(lr))
    print('DONE')

    print('Saving Lessons Learned')
    learn.save('ff_stage-1-rn50')

    print('Training Full Model Now...')
    # learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot()

    print('Adjusting Decay...')
    learn.fit(5, slice(1e-5, lr / 5))
    print('Saving Lessons...', end='')
    learn.save('ff_stage-2-rn50')
    print('\tDONE')
    # Training Part 2
    print('Starting up again...')
    data = (src.transform(tfms, size=256).databunch(bs=64).normalize(imagenet_stats))

    acc_05 = partial(accuracy_thresh, thresh=0.5)
    f_score = partial(fbeta, thresh=0.5)
    learn = cnn_learner(data, models.resnet50, pretrained=False, metrics=[acc_05, f_score])
    print("Remembering lessons learned...")
    learn.load("ff_stage-2-rn50")

    # print('Plotting...')
    # learn.freeze()
    # learn.lr_find()
    # learn.recorder.plot()

    print('Fitting...')
    lr = 0.01
    learn.fit(1, slice(lr))
    print('Saving...')
    learn.save('ff_stage-1-256-rn50')
    print('Training Complete!')

    # ==========================================================================
    # Testing
    # ==========================================================================

    # # Creating a databunch
    # imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # data = (
    #     ImageList.from_csv(PATH, csv_name="labels.csv")
    #         .no_split()
    #         .label_from_df(label_delim=" ")
    #         .transform(None, size=128)
    #         .databunch(no_check=True)
    #         .normalize(imagenet_stats)
    # )
    #
    # # Loading our model
    # learn = create_cnn(data, models.resnet50, pretrained=False)
    # learn.load("ff_stage-2-rn50")
    #
    # # Loading HAAR cascade
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #
    # while True:
    #     # Capture input-by-input
    #     input = open(IMG)
    #
    #     # Our operations on the input come here
    #     gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    #
    #     # Find faces using Haar cascade
    #     face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    #
    #     # Looping through each face
    #     for coords in face_coord:
    #
    #         # Finding co-ordinates of face
    #         X, Y, w, h = coords
    #
    #         # Finding input size
    #         H, W, _ = input.shape
    #
    #         # Computing larger face co-ordinates
    #         X_1, X_2 = (max(0, X - int(w * 0.35)), min(X + int(1.35 * w), W))
    #         Y_1, Y_2 = (max(0, Y - int(0.35 * h)), min(Y + int(1.35 * h), H))
    #
    #         # Cropping face and changing BGR To RGB
    #         img_cp = input[Y_1:Y_2, X_1:X_2].copy()
    #         img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
    #
    #         # Prediction of facial featues
    #         prediction = str(learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]).split(";")
    #         label = (
    #             " ".join(prediction)
    #             if "Male" in prediction
    #             else "Female " + " ".join(prediction)
    #         )
    #         label = (
    #             " ".join(prediction)
    #             if "No_Beard" in prediction
    #             else "Beard " + " ".join(prediction)
    #         )
    #
    #         # Drawing facial boundaries
    #         cv2.rectangle(
    #             img=input,
    #             pt1=(X, Y),
    #             pt2=(X + w, Y + h),
    #             color=(128, 128, 0),
    #             thickness=2,
    #         )
    #
    #         # Drawing facial attributes identified
    #         label_list = label.split(" ")
    #         for idx in range(1, len(label_list) + 1):
    #             cv2.putText(
    #                 input,
    #                 label_list[idx - 1],
    #                 (X, Y - 14 * idx),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.45,
    #                 (0, 128, 0),
    #                 2,
    #             )
    #
    #     # Display the resulting input
    #     cv2.imshow("input", input)
    #
    #     # Escape keys
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    #
    # cv2.destroyAllWindows()
    #
    # # detect_facial_attributes()
