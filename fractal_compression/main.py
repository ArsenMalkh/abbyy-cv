import cv2
import argparse
import numpy as np
from numba import njit, prange
from time import time
import tqdm
import os
from matplotlib import pyplot as plt


@njit(parallel=True, fastmath=True)
def MSE(output, original):
    h, w = output.shape
    return 1 / (h * w) * np.power(output - original, 2).sum()


@njit(parallel=True, fastmath=True)
def PSNR(output, original):
    mse = MSE(output, original)
    Y_max = 255

    return 10 * np.log10(Y_max * Y_max / mse)


@njit()
def get_all_possible_transforms():
    """
    this function returns all possible variants of orientation
    :return: list of tuples (flip, rotations)
    """
    rotations = (0, 1, 2, 3)
    flips = (-1, 1)
    transforms = []
    for flip in flips:
        for r in rotations:
            transforms.append((flip, rotations[r]))

    return transforms


@njit(parallel=True, fastmath=True)
def calculate_contrast_brightness(src_image, dst_image):
    """
    solve linear transform from one image to another
    :param src_image:
    :param dst_image:
    :return: coefficents k, b for brightness-contrast transform
    """
    c_size = src_image.shape[0]
    src_matrix = np.concatenate((src_image.copy().reshape((c_size ** 2, 1)), np.ones((c_size ** 2, 1))), axis=1)
    dst_matrix = dst_image.copy().reshape((c_size ** 2,)).astype(np.float64)

    return np.linalg.lstsq(src_matrix, dst_matrix)[0]


def read_image(path, fastmath=True):
    image_name = path.split("/")[-1].split(".")[0]
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE), image_name


@njit(parallel=True, fastmath=True)
def l1_distance(img1, img2):
    return np.mean(np.abs(img1 - img2))


@njit()
def spatial_transform(img, flip, angle):
    img = img[:, ::flip]
    for i in range(angle):
        img = img.T[::-1, :]
    return img


@njit(parallel=True, fastmath=True)
def color_transform(img, contrast, brightness):
    return img * contrast + brightness


@njit(parallel=True, fastmath=True)
def transform(img, transformation, contrast=1.0, brightness=0):
    img = spatial_transform(img, *transformation)
    return color_transform(img, contrast, brightness)


@njit(parallel=True, fastmath=True)
def resize(img, scale):
    """
    image compression by mean
    :param img: img np.array
    :param scale: scale in range 1.0 +
    :return: replace pixel with mean of block
    """
    rescaled_image = np.zeros((img.shape[0] // scale, img.shape[1] // scale))

    for i in prange(rescaled_image.shape[0]):
        for j in prange(rescaled_image.shape[1]):
            rescaled_image[i, j] = np.mean(img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale])

    return rescaled_image


@njit(fastmath=True)
def large_blocks(img, block_size):
    """
    create tuple of large blocks (i, j, orientation,
    :param img: source image
    :param block_size: small block size
    :return: list with info about every large block i, j, transformation number and resized  block itself
    """
    blocks = []
    large_block_size = block_size * 2

    for i in range(img.shape[0] // large_block_size):
        for j in range(img.shape[1] // large_block_size):
            block = img[i * large_block_size: (i + 1) * large_block_size,
                    j * large_block_size: (j + 1) * large_block_size]
            resized_block = resize(block, 2)
            for trans_number, transformation in enumerate(get_all_possible_transforms()):
                blocks.append((i, j, trans_number, spatial_transform(resized_block, *transformation)))
    return blocks


@njit(parallel=True, fastmath=True)
def encode_image(img, block_size):
    large_blocks_list = large_blocks(img, block_size)
    transforms = np.zeros((img.shape[0] // block_size, img.shape[1] // block_size, 5))
    k = 0
    size = img.shape[0] // block_size * img.shape[1] // block_size * len(large_blocks_list)
    for i in prange(img.shape[0] // block_size):
        for j in prange(img.shape[1] // block_size):
            target_block = img[i * block_size: (i + 1) * block_size,
                           j * block_size: (j + 1) * block_size]
            dist = np.inf
            for y, x, trans_number, block in large_blocks_list:
                contrast, brightness = calculate_contrast_brightness(block, target_block)
                block = color_transform(block, contrast, brightness)
                k = k + 1
                error = l1_distance(block, target_block)
                if error < dist:
                    dist = error
                    transforms[i, j] = (float(y), float(x), float(trans_number), contrast, brightness)
            print(round(k/size, 4))
    return transforms


@njit()
def decoder_step(img, block_size, transforms):
    result_image = np.zeros((transforms.shape[0] * block_size, transforms.shape[1] * block_size))
    all_possible_transforms = get_all_possible_transforms()
    for i in prange(transforms.shape[0]):
        for j in prange(transforms.shape[1]):
            y, x, trans_number, contrast, brightness = transforms[i, j]
            y, x, trans_number = int(y), int(x), int(trans_number)

            block = resize(img[y * 2 * block_size: (y + 1) * 2 * block_size,
                           x * 2 * block_size: (x + 1) * 2 * block_size], 2)
            result_block = transform(block, all_possible_transforms[trans_number], contrast, brightness)

            result_image[i * block_size: (i + 1) * result_image,
            j * block_size: (j + 1) * result_image] = result_block

    return result_image


def save_result(img, idx, save_path, image_name):
    cv2.imwrite(os.path.join(save_path, image_name, f"{idx}.bmp"), img)


def decoder(block_size, transforms, n_iterations, save_path, image_name, original_image):
    image = np.random(0, 256, original_image.shape).astype('uint8')
    image.save_result(image, 0, save_path, image_name)
    results = [PSNR(image, original_image)]
    for iteration in tqdm(range(1, n_iterations + 1)):
        image = decoder_step(image, block_size, transforms).astype('uint8')
        results.append(PSNR(image, original_image))
        image.save_result(image, iteration, save_path, image_name)

    plt.plot(results)
    plt.savefig(os.path.join(save_path, image_name, "psnr.png"))


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img_path', required=True,
                        help='imput image path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='save dir path')
    parser.add_argument('--block_size', type=int, default=4, choices=[4, 8],
                        help='Block size for patterns.')
    parser.add_argument('--n_iterations', type=int, default=10,
                        help='Restoring iterations.')

    return parser.parse_args()


@njit
def test(image):
    return image[::-1, :]


def main():
    args = get_args()

    img, image_name = read_image(args.img_path)
    print("encoding...")
    start_time = time()
    transforms = encode_image(img, args.block_size)
    print(f"encoding takes {time() - start_time}s")

    print("decoding...")
    start_time = time()
    decoder(args.block_size, transforms, args.n_iterations, args.save_dir, image_name, img)
    print(f"encoding takes {time() - start_time}s")


if __name__ == '__main__':
    main()
