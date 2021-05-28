from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm


class SegSet(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        # self.mapping = {
        #                 (0, 0, 0)       : 0,            # Background
        #                 (235, 66, 66)   : 1,            # Class 1:  Ground Glass
        #                 (151, 216, 121) : 2,            # Class 2:  Consolidation
        #                 (102, 204, 255) : 3             # Class 3:  Pleural Effusion
        #                 }

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def mapToRGB(self, mask):
        class_mask = mask
        h, w = class_mask.shape[1], class_mask.shape[2]

        # NOTE: All experimental from here.
        # classes = 4 - 1           # 3 classes + background.
        # classes = 3 - 1             # Only the three classes.
        classes = 6 - 1             # Only the three classes.     <--- Much better results.
        idx = np.linspace(0., 1., classes)
        cmap = cm.get_cmap('viridis')
        rgb = cmap(idx, bytes=True)[:, :3]  # Remove alpha value

        h, w = 256, 256
        rgb = rgb.repeat(1000, 0)
        target = np.zeros((h * w, 3), dtype=np.uint8)
        target[:rgb.shape[0]] = rgb
        target = target.reshape(h, w, 3)

        target = torch.from_numpy(target)
        colors = torch.unique(target.view(-1, target.size(2)), dim=0).numpy()
        # target = target.permute(2, 0, 1).contiguous()

        mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}
        # NOTE: End of Experiment (ptrblk).

        # # TODO: Check if this works better.
        # mask_out = np.zeros((h, w, 3))          # Creates empty template tensor
        # for i in range(mask_out.shape[0]):
        #     for j in range(mask_out.shape[1]):
        #         # TODO: figure out what an output should be.
        #         mask, idx = class_mask[i, j]
        #         mask_out[i, j] = self.mapping[idx]

        # NOTE: this is the original
        mask_out = torch.empty(h, w, dtype=torch.long)          # Creates empty template tensor
        # for k in self.mapping:
        for k in mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)
            mask_out[validx] = torch.tensor(mapping[k], dtype=torch.long)  # Fills in tensor

        return mask_out

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # Reorders dimensions from H, W, C to C, H, W
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        mask = self.mapToRGB(mask).float()

        return {
            'image': img,
            'mask': mask
        }