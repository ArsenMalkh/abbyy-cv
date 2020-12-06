import cv2
import numpy as np
from numba import njit
import os
from pathlib import Path
import argparse
from scipy import fftpack
from PIL import Image, ImageDraw
from tqdm import tqdm
import time


def read_image(path):
    image_name = path.split("/")[-1].split(".")[0]
    return cv2.imread(path, 0),  image_name


@njit
def find_skew(x, mu, sigma):
    return np.mean(np.power((x - mu) / sigma, 3))


@njit
def denoise(image, block_size):
    h, w = image.shape
    sigma = []
    for i in range(h // block_size):
        for j in range(w // block_size):
            block = image[i * block_size: (i + 1) * block_size,
                    j * block_size: (j + 1) * block_size]
            sigma.append(block.std())
    nu_2 = np.median(np.array(sigma)**2)

    result = np.zeros((h // block_size * block_size, w // block_size * block_size))
    for i in range(h // block_size):
        for j in range(w // block_size):
            block = image[i * block_size: (i + 1) * block_size,
                          j * block_size: (j + 1) * block_size]
            block_mean = np.mean(block)
            block_std = np.std(block)
            result[i * block_size: (i + 1) * block_size,
                   j * block_size: (j + 1) * block_size] = block_mean + ((block_std ** 2 - nu_2) * (block - block_mean) / (block_std ** 2))
    return result


@njit
def find_t(block, image_mean, bin_type):
    mu = np.mean(block)
    sigma = np.std(block)

    if bin_type == "sauvola":
        k = 0.2
        R = 128
        return mu * (1 + k * (sigma / R - 1))
    elif bin_type == "wolf":
        k = 0.25
        R = 110
        m = np.min(block)
        return (1 - k)*mu + k * m + k * sigma / R * (mu - m)
    elif bin_type == "2018":
        k = 0.2
        R = 110
        return (image_mean + np.max(block)) / 2 * (1 + k * (sigma/R-1))

@njit
def binarization(image, block_size, bin_type):
    h, w = image.shape
    image_mean = np.mean(image)
    result = np.zeros((h // block_size * block_size, w // block_size * block_size))
    for i in range(h // block_size):
        for j in range(w // block_size):
            block = image[i * block_size: (i + 1) * block_size,
                          j * block_size: (j + 1) * block_size]
            T = find_t(block, image_mean, bin_type)
            result[i * block_size: (i + 1) * block_size,
                  j * block_size: (j + 1) * block_size] = image[i * block_size: (i + 1) * block_size,
                                                                j * block_size: (j + 1) * block_size] > T
    return result


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgs_path', required=True,
                        help='input images path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='save dir path')
    parser.add_argument('--block_size', type=int, default=50,
                        help='Block size for patterns.')
    parser.add_argument('--binarization_type', type=str, choices=["sauvola", "2018", "wolf"],
                        help='type of binarization.')
    return parser.parse_args()


def save_result(img, save_path, image_name):
    image_dir = Path(save_path)
    image_dir.mkdir(exist_ok=True, parents=True)

    cv2.imwrite(os.path.join(save_path, f"{image_name}.tiff"), img)


def low_pass_filter(img):
    fft1 = fftpack.fftshift(fftpack.fft2(img))
    x, y = img.shape
    e_x, e_y = 10, 10
    bbox = ((x / 2) - (e_x / 2), (y / 2) - (e_y / 2), (x / 2) + (e_x / 2), (y / 2) + (e_y / 2))

    low_pass = Image.new("L", (img.shape[0], img.shape[1]), color=0)

    draw1 = ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np = np.array(low_pass)
    # multiply both the images
    filtered = np.multiply(fft1, low_pass_np.T)

    # inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    return ifft2.astype(np.uint8)


def contrast_adjustment(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]


def main():
    args = get_args()
    pathes = [os.path.join(args.imgs_path, image_path) for image_path in os.listdir(args.imgs_path)]
    for image_path in tqdm(pathes):
        img, image_name = read_image(image_path)
        img_c = contrast_adjustment(img)
        img_c = np.clip(img_c, 0, 255)
        img_c = binarization(img_c.astype(np.float64), args.block_size, args.binarization_type)
        img = binarization(img.astype(np.float64), args.block_size, args.binarization_type)
        save_result(np.vstack((255 * img, np.zeros((100, img.shape[1])), 255 * img_c)).astype("uint"), args.save_dir, image_name)


if __name__ == '__main__':
    main()
