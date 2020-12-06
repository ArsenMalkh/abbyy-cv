import argparse
import os
from tqdm import tqdm

import numpy as np
import cv2
from pathlib import Path
import time


class PyramidLayer:
    def __init__(self, previous_layer, first_layer=False, noise_const=10, noise_order_mult=3, noise_variance_mult=2,
                 mean_w=0.25, diff_w=0.75, shift=2):
        if first_layer:
            self.noise_constant = noise_const
            self.noise_order_mult = noise_order_mult
            self.noise_variance_mult = noise_variance_mult
            self.order = 0
            self.minimums = previous_layer
            self.maximums = previous_layer
            self.means = previous_layer
            self.shape = (previous_layer.shape[0], previous_layer.shape[1])
            self.variances = np.zeros(self.shape)
            self.pad()
            self.mean_w = mean_w
            self.diff_w = diff_w
            self.shift = shift
            assert np.abs(mean_w + diff_w - 1) < 1e-8

        else:
            self.shift = shift
            self.noise_constant = noise_const
            self.noise_order_mult = noise_order_mult
            self.noise_variance_mult = noise_variance_mult
            self.order = previous_layer.order + 1
            self.shape = (previous_layer.shape[0] // 2, previous_layer.shape[1] // 2)
            self.minimums = np.zeros(self.shape)
            self.maximums = np.zeros(self.shape)
            self.means = np.zeros(self.shape)
            self.previous_layer = previous_layer
            self.thresholds = np.zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.minimums[i, j] = np.min(self.previous_layer.minimums[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2])
                    self.maximums[i, j] = np.max(self.previous_layer.maximums[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2])
                    self.means[i, j] = np.mean(self.previous_layer.means[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2])
            self.interpolate_variances()
            self.pad()
            self.mean_w = mean_w
            self.diff_w = diff_w
            assert np.abs(mean_w + diff_w - 1) < 1e-8

    def pad(self):
        if self.shape[0] % 2 == 1:
            self.maximums = np.vstack((self.maximums, self.maximums[-1, :]))
            self.minimums = np.vstack((self.minimums, self.minimums[-1, :]))
            self.means = np.vstack((self.means, self.means[-1, :]))
            self.variances = np.vstack((self.variances, self.variances[-1, :]))
            self.shape = (self.shape[0] + 1, self.shape[1])
        if self.shape[1] % 2 == 1:
            self.maximums = np.hstack((self.maximums, self.maximums[:, -1].reshape((self.maximums.shape[0], 1))))
            self.minimums = np.hstack((self.minimums, self.minimums[:, -1].reshape((self.maximums.shape[0], 1))))
            self.means = np.hstack((self.means, self.means[:, -1].reshape((self.maximums.shape[0], 1))))
            self.variances = np.hstack((self.variances, self.variances[:, -1].reshape((self.maximums.shape[0], 1))))
            self.shape = (self.shape[0], self.shape[1] + 1)

    def interpolate_variances(self):
        h, w = self.shape
        self.variances = np.zeros((self.shape))
        for i in range(h):
            for j in range(w):
                previous_variances = np.mean(self.previous_layer.variances[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2])
                previous_squared_mean = np.mean(self.previous_layer.means[i * 2: (i + 1) * 2, j * 2: (j + 1) * 2] ** 2)
                self.variances[i, j] = previous_variances + previous_squared_mean - self.means[i, j] ** 2

    def resize_threshold(self, deeper_layer):
        h, w = self.shape
        self.thresholds = np.zeros((h, w))
        for i in range(0, h, 2):
            for j in range(w - 1):
                if j % 2 == 0:
                    self.thresholds[i, j] = deeper_layer.thresholds[i // 2, j // 2]
                else:
                    self.thresholds[i, j] = 0.25 * deeper_layer.thresholds[i // 2, j // 2] + 0.75 * \
                                            deeper_layer.thresholds[
                                                i // 2, j // 2 + 1]
            self.thresholds[i, -1] = self.thresholds[i, -2]
        for i in range(1, h - 1, 2):
            self.thresholds[i] = 0.75 * self.thresholds[i - 1] + 0.25 * self.thresholds[i + 1]
        self.thresholds[-1] = self.thresholds[-2]

    def get_noise_constant(self):
        return self.noise_constant + self.order * self.noise_order_mult + self.noise_variance_mult * np.sqrt(
            self.variances)

    def change_threshold(self):
        h, w = self.shape
        noise = self.get_noise_constant()
        for i in range(h):
            for j in range(w):
                if self.maximums[i, j] - self.minimums[i, j] > noise[i, j]:
                    self.thresholds[i, j] = self.mean_w * self.means[i, j] + 0.5 * self.diff_w * (
                            self.maximums[i, j] + self.minimums[i, j]) + self.shift


class ImagePyramid:
    def __init__(self, img, letter_layer=3, mean_w=0.15, diff_w=0.85, shift=2):
        self.img = img
        self.layers = [PyramidLayer(img, True)]
        self.letter_layer = letter_layer
        self.mean_w = mean_w
        self.diff_w = diff_w
        self.shift = shift

        while self.layers[-1].shape[0] > 2 and self.layers[-1].shape[1] > 2:
            self.layers.append(PyramidLayer(self.layers[-1], mean_w=mean_w, diff_w=diff_w))
        self.pad_image()

    def pad_image(self):
        if self.img.shape[0] % 2 == 1:
            self.img = np.vstack((self.img, self.img[-1, :]))
        if self.img.shape[1] % 2 == 1:
            self.img = np.hstack((self.img, self.img[:, -1].reshape((self.img.shape[0], 1))))

    def calc_thresholds(self):
        self.layers[-1].thresholds = self.mean_w * self.layers[-1].means + 0.5 * self.diff_w * (
                self.layers[-1].maximums + self.layers[-1].minimums)
        for i in range(len(self.layers) - 2, self.letter_layer, -1):
            self.layers[i].resize_threshold(self.layers[i + 1])
            self.layers[i].change_threshold()
        for i in range(self.letter_layer, -1, -1):
            self.layers[i].resize_threshold(self.layers[i + 1])

    def binarize_image(self):
        return self.img > self.layers[0].thresholds - self.shift


def read_image(path):
    image_name = path.split("/")[-1].split(".")[0]
    return cv2.imread(path, 0), image_name


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgs_path', required=True,
                        help='input images path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='save dir path')


    return parser.parse_args()


def save_result(img, save_path, image_name):
    image_dir = Path(save_path)
    image_dir.mkdir(exist_ok=True, parents=True)

    cv2.imwrite(os.path.join(save_path, f"{image_name}.tiff"), img)


def contrast_adjustment(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]


def main():
    args = get_args()
    paths = [os.path.join(args.imgs_path, image_path) for image_path in os.listdir(args.imgs_path)]
    for image_path in tqdm(paths):
        img, image_name = read_image(image_path)
        img_c = contrast_adjustment(img)
        img = ImagePyramid(img.astype(np.float64))
        img.calc_thresholds()
        img = img.binarize_image()
        # print("no correction made")
        # img_c = ImagePyramid(img_c.astype(np.float64))
        # img_c.calc_thresholds()
        # img_c = img_c.binarize_image()
        # print("correction made")
        save_result(255 * img, args.save_dir,
                    image_name)
        time.sleep(60)


if __name__ == '__main__':
    main()