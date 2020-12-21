# import torch
# import torchvision
# import pandas as pd
# import numpy as np
# from fastai.vision import *
# import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', 500)
#
#
# # Function to filter validation samples
# def validation_func(x):
#     return 'validation' in x
#
# tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate=30, max_lighting=0.3)
#
# src = (ImageList.from_csv('./data/celeba/faces/training/', csv_name='labels.csv')
#        .split_by_valid_func(validation_func)
#        .label_from_df(cols='tags', label_delim=' '))
#
# data = (src.transform(tfms, size=128).databunch(bs=256).normalize(imagenet_stats))
#
# print(data.c, '\n', data.classes)
#
# # Training Algorithm
# data.show_batch(rows=2, figsize=(20, 12))
#
# arch = models.resnet50
# acc_02 = partial(accuracy_thresh, thresh=0.2)
# acc_03 = partial(accuracy_thresh, thresh=0.3)
# acc_04 = partial(accuracy_thresh, thresh=0.4)
# acc_05 = partial(accuracy_thresh, thresh=0.5)
# f_score = partial(fbeta, thresh=0.2)
#
# learn = create_cnn_model(data, arch, metrics=[acc_02, acc_03, acc_04, acc_05, f_score])
# learn.lr_find()
# learn.recorder.plot()
#
# lr = 1e-2
# learn.fit_one_cycle(1, slice(lr))
# learn.fit_one_cycle(4, slice(lr))
# learn.save('ff_stage-1-rn50')
#
# learn.unfreeze()
#
# learn.lr_find()
# learn.recorder.plot()
#
# learn.fit_one_cycle(5, slice(1e-5, lr/5))
# learn.save('ff_stage-2-rn50')
#
# # Training Part 2
# data = (src.transform(tfms, size=256).databunch(bs=64).normalize(imagenet_stats))
#
# acc_05 = partial(accuracy_thresh, thresh=0.5)
# f_score = partial(fbeta, thresh=0.5)
# learn = create_cnn_model(data, models.resnet50, pretrained=False, metrics=[acc_05, f_score])
# learn.load("ff_stage-2-rn50")
#
# learn.freeze()
# learn.lr_find()
# learn.recorder.plot()
#
# lr = 0.01
# learn.fit_one_cycle(1, slice(lr))
# learn.save('ff_stage-1-256-rn50')
#
