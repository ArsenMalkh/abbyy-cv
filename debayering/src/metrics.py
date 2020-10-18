import numpy as np
from numba import njit

def Y(image):
    result = 0.299 * image[:,:,0] + 0.5879 * image[:,:,1] + 0.114 * image[:,:,2]
    return result

def MSE(output, original):
    Y_ref = Y(original[2:-2, 2:-2, :])
    Y_out = Y(output[2:-2, 2:-2, :])
    h, w = Y_out.shape
    
    return 1 / (h * w) * np.sum((Y_out - Y_ref) ** 2)

def PSNR(output, original):
    mse = MSE(output, original)
    Y_max = 255
    
    return 10 * np.log10(Y_max * Y_max / mse)
    
