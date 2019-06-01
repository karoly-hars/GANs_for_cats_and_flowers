import os
import os.path as osp
import cv2
import numpy as np
import math
import random
import torch
from torch.utils.data import Dataset


def rot_image(img, angle, center):
    (height, width) = img.shape[:2]
    # Perform the rotation
    rot_mat = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, (width, height), borderValue=(128, 128, 128))
    return rotated_img


def rotate_image_and_keypoints(img, keypoints, angle):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    # Perform the rotation on image
    rot_mat = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
    img = cv2.warpAffine(img, rot_mat, (width, height), borderValue=(128, 128, 128))

    # Rotate keypoints
    for point_name, (px, py) in keypoints.items():
        px, py
        qx = center[0] + math.cos(-angle) * (px - center[0]) - math.sin(-angle) * (py - center[1])
        qy = center[1] + math.sin(-angle) * (px - center[0]) + math.cos(-angle) * (py - center[1])
        keypoints[point_name] = (int(qx), int(qy))
    return img, keypoints


def flip_image_and_keypoints(img, keypoints):
    height = img.shape[0]
    # Flip image
    img = cv2.flip(img, 0)
    # Flip keypints
    for point_name, (px, py) in keypoints.items():
        keypoints[point_name] = (px, height - py)
    return img, keypoints


# crops and resizes the image part that contains the cat's face, based on the annotations
def crop_catface(img_path, annotation_path):
    # load img
    img = cv2.imread(img_path)
    # get parse annotation
    annotation = open(annotation_path, "r")
    line = annotation.readline()
    line = line.strip().split()[1:]
    line = [int(x) for x in line]
    keypoints = {"left_eye": (line[0], line[1]),
                 "right_eye": (line[2], line[3]),
                 "left_ear_base": (line[6], line[7]),
                 "right_ear_base": (line[16], line[17]),
                 "mouth": (line[4], line[5])}

    # rotate image, so the x axis is parallel with the line that connects the ears
    eyeline_h = (keypoints["right_eye"][1] - keypoints["left_eye"][1])
    eyeline_w = (keypoints["right_eye"][0] - keypoints["left_eye"][0])
    angle = math.atan2(eyeline_h, eyeline_w)
    img, keypoints = rotate_image_and_keypoints(img, keypoints, angle)

    # if the cat is upside-down
    if keypoints["mouth"][1] < keypoints["right_eye"][1]:
        img, keypoints = flip_image_and_keypoints(img, keypoints)

    # for some reason after turning the images, the ears might be mixed up (maybe somthing with the dataset?)
    # quick fix
    if keypoints["right_ear_base"][0] < keypoints["left_ear_base"][0]:
        keypoints["left_ear_base"], keypoints["right_ear_base"] = keypoints["right_ear_base"], keypoints[
            "left_ear_base"]

    # calculate the new width
    new_width = keypoints["right_ear_base"][0] - keypoints["left_ear_base"][0]
    new_height = new_width

    # calculate a center point for the face is calculated based on the eyes
    center = (keypoints["right_eye"][0] * 0.5 + keypoints["left_eye"][0] * 0.5, keypoints["left_eye"][1])

    # make sure we are not trying to crop outside the image
    if center[0] - new_width * 0.5 < 0:
        new_width += center[0] - new_width * 0.5
    if center[1] - new_height * 0.4 < 0:
        new_height += center[1] - new_height * 0.4

    top_left_x = int(max(0, center[0] - new_width * 0.5))
    top_left_y = int(max(0, center[1] - new_height * 0.4))
    bottom_right_x = int(min(img.shape[1], center[0] + new_width * 0.5))
    bottom_right_y = int(min(img.shape[0], center[1] + new_height * 0.6))

    # crop
    img = img[top_left_y: bottom_right_y, top_left_x: bottom_right_x, :]

    # resize to square
    new_size = int(max(new_width, new_height))
    img = cv2.resize(img, (new_size, new_size))
    return img


# downloads and unzips the cat database, and extracts the cat faces from the annotations
def prepare_dataset(data_path="./data"):
    if not osp.exists(data_path):
        os.makedirs(data_path)
        # download
        print("Downloading dataset part 1...")
        os.system(
            "wget https://web.archive.org/web/20150520175555/"
            "http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_01.zip -P {}".format(data_path))
        print("Downloading dataset part 2...")
        os.system(
            "wget https://web.archive.org/web/20150520175645/"
            "http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_02.zip -P {}".format(data_path))
        os.system(
            "wget https://web.archive.org/web/20130527104257/"
            "http://137.189.35.203/WebUI/CatDatabase/Data/00000003_015.jpg.cat -P {}".format(data_path))

        # unzip
        print("Unziping data partset 1...")
        os.system("unzip {}/CAT_DATASET_01.zip -d {}/CAT_DATASET_01".format(data_path, data_path))
        print("Unziping data partset 2...")
        os.system("unzip {}/CAT_DATASET_02.zip -d {}/CAT_DATASET_02".format(data_path, data_path))

        # bugfix according to
        # https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html
        os.system("mv {}/00000003_015.jpg.cat {}/CAT_DATASET_01/CAT_00".format(data_path, data_path))
        os.system("rm {}/CAT_DATASET_01/CAT_00/00000003_019.jpg.cat".format(data_path))

    else:
        print("Dataset already prepared in {}".format(data_path))

    img_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn if
                 f.endswith(".jpg")]
    annotation_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn if
                        f.endswith(".jpg.cat")]
    img_paths.sort()
    annotation_paths.sort()

    return img_paths, annotation_paths


# Extracts the faces from each cat image
def extract_catfaces(img_paths, annotation_paths, catfaces_path="./data/catfaces"):
    if not osp.exists(catfaces_path):
        # crop around the faces and save the new imageset
        print("Extracting catfaces...")
        os.makedirs(catfaces_path)

        for idx in range(len(img_paths)):
            img_path = img_paths[idx]
            annotation_path = annotation_paths[idx]
            # print("{}: {},{}".format(idx+1, img_path, annotation_path))
            img = crop_catface(img_path, annotation_path)
            cv2.imwrite("{}/{}_{}".format(catfaces_path, img_path.split('/')[-2], img_path.split('/')[-1]), img)
    else:
        print("Catfaces dataset is ready.")

    catface_img_paths = [osp.join(catfaces_path, f) for f in os.listdir(catfaces_path) if f.endswith(".jpg")]
    return catface_img_paths


# transform network into displayable img
def postprocess_img(img):
    img = img.transpose((1, 2, 0))
    img += 1.0
    img = (img * 128.0).astype(np.uint8)
    return img


def preprocess_img(img):
    # normalize
    img = img / 128.0  # between 0 and 2
    img -= 1.0  # between -1 and 1
    # transpose
    img = img.transpose((2, 0, 1))
    return img


class CatfaceDataset(Dataset):
    def __init__(self, img_paths, size=64, mirror=True, random_crop=True):
        self.img_paths = img_paths
        self.size = size
        self.mirror = mirror
        self.random_crop = random_crop

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)

        # mirror img with a 50% chance
        if self.mirror:
            if random.random() > 0.5:
                img = img[:, ::-1, :]
        img = cv2.resize(img, (self.size, self.size))

        # normalize
        img = preprocess_img(img)
        img = torch.Tensor(img)
        return img
