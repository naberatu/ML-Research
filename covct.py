#
# from __future__ import print_function, division
# from PIL import Image
# import os
# import os.path
# import numpy as np
# import pickle
# from typing import Any, Callable, Optional, Tuple
# import pandas as pd
# import matplotlib as plt
# import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
#
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
#
# def sample_scans():
#     covid_files_path = './data/covct/'
#     covid_files = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
#     covid_images = [cv2.imread(x) for x in random.sample(covid_files, 5)]
#
#     plt.figure(figsize=(20, 10))
#     columns = 5
#     for i, image in enumerate(covid_images):
#         plt.subplot(len(covid_images) / columns + 1, columns, i + 1)
#         plt.imshow(image)
#
#
# class CovCT(Dataset):
#     def __init__(self, root_dir="./data/covct/", train=True, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.train = True
#         self.covid_list = []
#         self.nocov_list = []
#
#         start1, start2, finish1, finish2 = 0, 0, 0, 0
#         if train:
#             start1, start2, finish1, finish2 = 1, 1, 626, 615
#         else:
#             start1, start2, finish1, finish2 = 627, 616, 1252, 1230
#
#         for i in range(start1, finish1):
#             self.covid_list.append("Covid (" + str(i) + ").png")
#         for i in range(start2, finish2):
#             self.nocov_list.append("Non-Covid (" + str(i) + ").png")
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         return self.root_dir + self.covid_list[idx]
#
#
#
#
#
