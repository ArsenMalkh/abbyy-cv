import numpy as np
from numba import njit
import cv2


def MSE(output, original):
    Y_ref = cv2.cvtColor(output[2:-2, 2:-2, :].astype('uint8'), cv2.COLOR_RGB2GRAY)
    Y_out = cv2.cvtColor(original[2:-2, 2:-2, :].astype('uint8'), cv2.COLOR_RGB2GRAY)
    h, w = Y_out.shape
    
    return 1 / (h * w) * np.power(Y_ref - Y_out, 2).sum()

def PSNR(output, original):
    mse = MSE(output, original)
    Y_max = 255
    
    return 10 * np.log10(Y_max * Y_max / mse)
    
