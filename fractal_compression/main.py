import cv2
import argparse
import numpy as np
from numba import njit
from time import time
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from pathlib  import Path
import scipy


@njit(parallel=True, fastmath=False)
def MSE(output, original):
    h, w = output.shape
    return 1 / (h * w) * np.power(output - original, 2).sum()


@njit(fastmath=False)
def PSNR(output, original):
    mse = MSE(output, original)
    Y_max = 255
    return 10 * np.log10(Y_max * Y_max / np.power(output - original, 2).mean())


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
def classify(img):
    threshold = np.mean(img)
    first_mean = np.mean(img[:img.shape[0] // 2, :img.shape[1] // 2])
    second_mean = np.mean(img[:img.shape[0] // 2, img.shape[1] // 2:])
    third_mean = np.mean(img[img.shape[0] // 2:, :img.shape[1] // 2])
    fourth_mean = np.mean(img[img.shape[0] // 2:, img.shape[1] // 2:])

    return first_mean > threshold, second_mean > threshold, third_mean > threshold, fourth_mean > threshold


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


def read_image(path):
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


@njit(fastmath=True)
def transform(img, transformation, contrast=1.0, brightness=0):
    img = spatial_transform(img, *transformation)
    return color_transform(img, contrast, brightness)


@njit(parallel=False, fastmath=True)
def resize(img, scale):
    """
    image compression by mean
    :param img: img np.array
    :param scale: scale in range 1.0 +
    :return: replace pixel with mean of block
    """
    rescaled_image = np.zeros((img.shape[0] // scale, img.shape[1] // scale), dtype=np.float64)

    for i in range(rescaled_image.shape[0]):
        for j in range(rescaled_image.shape[1]):
            rescaled_image[i, j] = np.mean(img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale])

    return rescaled_image



@njit()
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
                transformed_block = spatial_transform(resized_block, *transformation)
                bin_hash = classify(transformed_block)
                blocks.append((i, j, trans_number, bin_hash, transformed_block))
    return blocks


@njit(parallel=False, fastmath=True)
def encode_image(img, block_size):
    large_blocks_list = large_blocks(img, block_size)
    transforms = np.zeros((img.shape[0] // block_size, img.shape[1] // block_size, 5), dtype=np.float64)
    size = img.shape[0] // block_size
    for i in range(img.shape[0] // block_size):
        for j in range(img.shape[1] // block_size):
            target_block = img[i * block_size: (i + 1) * block_size,
                               j * block_size: (j + 1) * block_size]
            block_hash = classify(target_block)
            dist = np.inf
            for y, x, trans_number, bin_hash, block in large_blocks_list:
                if block_hash == bin_hash:
                    contrast, brightness = calculate_contrast_brightness(block, target_block)
                    block = color_transform(block, contrast, brightness)
                    error = l1_distance(block, target_block)
                    if error < dist:
                        dist = error
                        transforms[i, j] = (float(y), float(x), float(trans_number), contrast, brightness)
        print(round(i / size, 2))
    return transforms


@njit(parallel=False)
def decoder_step(img, block_size, transforms):
    result_image = np.zeros((transforms.shape[0] * block_size, transforms.shape[1] * block_size), dtype=np.float64)
    all_possible_transforms = get_all_possible_transforms()
    for i in range(transforms.shape[0]):

        for j in range(transforms.shape[1]):
            y, x, trans_number, contrast, brightness = transforms[i, j]
            y, x, trans_number = int(y), int(x), int(trans_number)

            block = resize(img[y * 2 * block_size: (y + 1) * 2 * block_size,
                           x * 2 * block_size: (x + 1) * 2 * block_size], 2)
            result_block = transform(block, all_possible_transforms[trans_number], contrast, brightness)

            result_image[i * block_size: (i + 1) * block_size,
            j * block_size: (j + 1) * block_size] = result_block

    return result_image


def save_result(img, idx, save_path, image_name, block_size):
    image_dir = Path(os.path.join(save_path, f"{image_name}_{block_size}"))
    image_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(os.path.join(save_path, f"{image_name}_{block_size}", f"{idx}.tiff"), img)


def decoder(block_size, transforms, n_iterations, save_path, image_name, original_image):
    image = np.random.randint(0, 255, (transforms.shape[0] * block_size, transforms.shape[1] * block_size))
    images = [image]
    save_result(image.astype('uint8'), 0, save_path, image_name, block_size)
    results = [PSNR(image.astype(np.float64), original_image.astype(np.float64))]
    save_result(original_image, "original", save_path, image_name, block_size)
    for iteration in tqdm(range(1, n_iterations + 1)):
        image = decoder_step(image.astype(np.float64), block_size, transforms)
        images.append(image)
        results.append(PSNR(image.astype(np.float64), original_image.astype(np.float64)))
        save_result(image.astype('uint8'), iteration, save_path, image_name, block_size)
    print(len(results))
    save_result(images[np.argmax(results)].astype('uint8'), "best", save_path, image_name, block_size)
    print(len(images))
    print(f"Best PSNR value {max(results)}")
    plt.plot(results)
    plt.xlabel("iteration numbers")
    plt.ylabel("PSNR")
    plt.savefig(os.path.join(save_path, f"{image_name}_{block_size}", "psnr.png"))


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img_path', required=True,
                        help='imput image path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='save dir path')
    parser.add_argument('--block_size', type=int, default=4, choices=[4, 8],
                        help='Block size for patterns.')
    parser.add_argument('--n_iterations', type=int, default=100,
                        help='Restoring iterations.')
    parser.add_argument('--debug_scale', type=int, default=None,
                        help='scale for debugging resize')

    return parser.parse_args()


@njit
def test(image):
    return image[::-1, :]


def main():
    args = get_args()

    img, image_name = read_image(args.img_path)
    if args.debug_scale is not None:
        img = resize(img, args.debug_scale)
    print("encoding...")
    start_time = time()
    transforms = encode_image(img.astype(np.float64), args.block_size)
    print("encoding takes {}s".format(time() - start_time))

    print("decoding...")
    start_time = time()
    decoder(args.block_size, transforms, args.n_iterations, args.save_dir, image_name, img)
    print("encoding takes {}s".format(time() - start_time))


if __name__ == '__main__':
    main()
