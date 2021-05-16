import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def check_area(space_arr):
    for i in range(1):
        for j in range(i+1, len(space_arr) - 1):
            for k in range(j+1, len(space_arr)):
                first_to_second = space_arr[i] / space_arr[j]
                first_to_third = space_arr[i] / space_arr[k]
                second_to_third = space_arr[j] / space_arr[k]
                flag_1 = np.isclose(first_to_second, 49/25, 49/25 * 0.4)
                flag_2 = np.isclose(first_to_third, 49/9, 49/9 * 0.4)
                flag_3 = np.isclose(second_to_third, 25/9, 25/9 * 0.4)
                if flag_1 and flag_2 and flag_3:
                    return True
    return False


def finder_pattern_det(args):
    imgs = sorted([file for file in os.listdir(args.input) if
                   file.split(".")[-1] in ['png', 'jpg', 'JPEG', 'JPG', 'jpeg']])
    for img_path in tqdm(imgs):
        img = cv2.imread(os.path.join(args.input, img_path))
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grey_img = cv2.GaussianBlur(grey_img, (7, 7), 0)

        grey_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        grey_img = cv2.morphologyEx(grey_img, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        grey_img = cv2.erode(grey_img, np.ones((3, 3), np.uint8), iterations=1)

        detected_edges = cv2.Canny(grey_img, 80, 110, 7)
        mask = (detected_edges != 0).astype(grey_img.dtype) * 255

        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        finded = []
        cs = []
        hierarchy = hierarchy[0]
        used = []
        for i in range(hierarchy.shape[0]):
            k = i
            c = 0
            spaces = []
            used_local = []

            while hierarchy[k][2] != -1:
                spaces.append(cv2.contourArea(contours[k]))
                used_local.append(k)
                k = hierarchy[k][2]
                c += 1

            if hierarchy[k][2] != -1:
                c += 1
                spaces.append(cv2.contourArea(contours[k]))

            if 4 <= c <= 7:
                flag = check_area(spaces)
                if flag and i not in used:
                    finded.append(contours[i])
                    used.extend(used_local)
        for cont in finded:
            cv2.drawContours(img, [cont], 0, (217, 255, 19), 6, cv2.LINE_8, hierarchy, 0)
        cv2.imwrite(os.path.join(args.output, img_path), img)


def parse_args():
    parser = argparse.ArgumentParser(description='count errors in training data')
    parser.add_argument('--input', default="./data/TestSet1",
                        help='path input dir')
    parser.add_argument('--output', default="./data/results",
                        help='save path')
    return parser.parse_args()


def main():
    args = parse_args()
    finder_pattern_det(args)


if __name__ == "__main__":
    main()
