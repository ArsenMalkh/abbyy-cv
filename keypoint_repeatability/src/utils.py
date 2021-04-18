import cv2
import numpy as np
import os
from numba import njit
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt




def plot_distribution(res, label, y, save_name, save_dir):
    figure=plt.figure(figsize=(10, 10))
    for method in res:
        plt.plot(res[method]["x"], res[method][y], label=method)
    plt.legend()
    plt.title(label)
    plt.savefig(f"{save_dir}/{save_name}.jpg")


def find_descriptors(img, method):
    if method == "shi-thomasi":
        start = time()
        pts = cv2.goodFeaturesToTrack(img,1000,0.001,2)
        end = time()
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        sift = cv2.SIFT_create()
        _, des = sift.compute(img,kp)
    elif method == "sift":
        sift = cv2.SIFT_create()
        start = time()
        kp = sift.detect(img,None)
        end = time()
        kp, des = sift.compute(img, kp)
    elif method == "orb":
        orb = cv2.ORB_create()
        start = time()
        kp = orb.detect(img, None)
        end = time()
        kp, des = orb.compute(img, kp)
    return kp, des, end-start


@njit
def get_distance_matrix(first_arr, second_arr):
    distances = np.zeros((first_arr.shape[0], second_arr.shape[0]))
    for i in range(first_arr.shape[0]):
        for j in range(second_arr.shape[0]):
            distances[i, j] = np.sqrt((first_arr[i][0] - second_arr[j][0])**2 + (first_arr[i][1] - second_arr[j][1])**2)
    return distances


def compute_repeatability_motion(img_list, method):
    kp_0, _, time_for_pict = find_descriptors(img_list[0], method)
    kp_0 = cv2.KeyPoint_convert(kp_0)
    result = np.zeros((kp_0.shape[0], len(img_list) - 1))
    all_time = time_for_pict
    N = len(kp_0)
    for k, img in enumerate(tqdm(img_list[1:])):
        (x, y), *_ = cv2.phaseCorrelate(np.float32(img_list[0]), np.float32(img))
        kp, _, time_for_pict = find_descriptors(img, method)
        all_time += time_for_pict
        N += len(kp)
        kp = cv2.KeyPoint_convert(kp)
        kp[:, 0] -= x
        kp[:, 1] -= y

        distances = get_distance_matrix(kp_0, kp).min(axis=1)
        result[distances < 2.5, k] += 1
    return result, all_time / N


def compute_repeatability_descriptors(img_list, method):
    threshold = {"shi-thomasi": 70, "sift": 70, "orb": 60}
    _, des_0, time_for_pict = find_descriptors(img_list[0], method)
    dess = [des_0]
    result = np.zeros((dess[0].shape[0], len(img_list) - 1))
    all_time = time_for_pict
    N = des_0.shape[0]
    for k, image in enumerate(tqdm(img_list[1:])):
        _, des, time_for_pict = find_descriptors(image, method)
        N += des.shape[0]
        all_time += time_for_pict
        if method == "orb":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(dess[0], des)
        distances = []
        for i, match in enumerate(matches):
            distances.append(match.distance)
            if match.distance < threshold[method]:
                result[i, k] += 1
    return result, all_time / N


def get_images(path):
    img_paths = sorted(os.listdir(path))
    img_list = []
    for img_path in img_paths:
        img_list.append(cv2.imread(os.path.join(path, img_path), cv2.IMREAD_GRAYSCALE))
    return img_list

