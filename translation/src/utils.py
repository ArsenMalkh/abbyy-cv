from src import superpoint
import os
from tqdm import tqdm
import cv2
import numpy as np
import os


def profiler(start_time, end_time, size):
    time = end_time - start_time

    return time, time / size

def prepare_img(img):
    interp = cv2.INTER_AREA
    img = cv2.resize(img, (160, 120), interpolation=interp)
    img = (img.astype('float32') / 255.)
    return img


def extract_image_kp(img, keypoint_extractor):
    pts, desc, *_ = keypoint_extractor.run(img)
    return pts, desc


def find_trans(pts_1, pts_2, matches):
    trans_x = 0
    trans_y = 0
    trans = []
    for i in range(matches.shape[1]):
        first_image_index = int(matches[0, i])
        second_image_index = int(matches[1, i])
        x_1, y_1 = pts_1[:2, first_image_index]
        x_2, y_2 = pts_2[:2, second_image_index]
        trans_x_loc = (x_2 - x_1) ** 2
        trans_y_loc = (y_2 - y_1) ** 2
        trans.append(np.sqrt(trans_x_loc + trans_y_loc))

    mask = np.logical_and(trans >= np.quantile(trans, 0.2),
                          trans <= np.quantile(trans, 0.8))
    for i in range(len(mask)):
        if mask[i]:
            first_image_index = int(matches[0, i])
            second_image_index = int(matches[1, i])
            x_1, y_1 = pts_1[:2, first_image_index]
            x_2, y_2 = pts_2[:2, second_image_index]
            trans_x += (x_2 - x_1)
            trans_y += (y_2 - y_1)
    trans_x = round(trans_x / mask.sum(), 1)
    trans_y = round(trans_y / mask.sum(), 1)
    return (trans_x, trans_y)