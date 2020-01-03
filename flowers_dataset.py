import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import preprocess_img


def prepare_flowers_dataset(data_path='./data'):
    """Download and unzip flower datas."""
    flowers_data_path = os.path.join(data_path, 'flower_data')
    if not os.path.exists(flowers_data_path):
        os.makedirs(flowers_data_path)
        # download
        print('Downloading flower dataset...')
        os.system('wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -P {}'
                  .format(flowers_data_path))
        # extract
        print('Unzipping flower dataset...')
        os.system('tar -C {} -xvzf {}'.format(
            flowers_data_path, os.path.join(flowers_data_path, '102flowers.tgz')))
    else:
        print('Dataset already prepared in {}'.format(flowers_data_path))

    img_dir = os.path.join(flowers_data_path, 'jpg')
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    img_paths.sort()
    return img_paths


class Flowers64Dataset(Dataset):
    """Dataset object for 64x64 pixel flower images."""

    def __init__(self, img_paths, mirror=True):
        self.img_paths = img_paths
        self.size = 64
        self.mirror = mirror

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)

        # center crop
        h, w = img.shape[:2]
        min_side = min(h, w)
        top, bot = (h - min_side)//2, h - (h - min_side)//2
        left, right = (w - min_side) // 2, w - (w - min_side) // 2
        img = img[top:bot, left:right, :]

        # mirror img with a 50% chance
        if self.mirror:
            if random.random() > 0.5:
                img = img[:, ::-1, :]

        # resize
        img = cv2.resize(img, (self.size, self.size))

        # normalize
        img = preprocess_img(img)

        return torch.tensor(img.astype(np.float32))
