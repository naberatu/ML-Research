# import torch
# import torchvision
# import pandas as pd
# import numpy as np
# from fastai.vision import *
# import matplotlib.pyplot as plt
# pd.set_option('display.max_columns', 500)
#
#
# # Parameters
# # ============================================================================================
# PATH = './data/celeba/faces'
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # Loading CIFAR-10
# # ============================================================================================
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
