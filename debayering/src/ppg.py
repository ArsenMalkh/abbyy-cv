import numpy as np
from numba import njit, prange

@njit
def normalize_color(value):
    value = min(value, 255)
    return max(value, 0)

@njit
def patch_3_3(m, x, y):
    return m[y - 1 : y + 2, x - 1 : x + 2].flatten()

@njit
def patch_5_5(m, x, y):
    return m[y - 2 : y + 3, x - 2 : x + 3].flatten()

@njit
def get_color(x, y):
    if y % 2 == 0:
        if x % 2 == 0:
            return "red"
        else:
            return "green"
    else:
        if x % 2 == 0:
            return "green"
        else:
            return "blue"

@njit
def first_step(green_matrix, red_blue_matrix):
    N = 2 * abs(red_blue_matrix[12]  - red_blue_matrix[2]) + abs(green_matrix[7] - green_matrix[17])
    E = 2 * abs(red_blue_matrix[12] - red_blue_matrix[14]) + abs(green_matrix[11] - green_matrix[13])
    W = 2 * abs(red_blue_matrix[12] - red_blue_matrix[10]) + abs(green_matrix[11] - green_matrix[13])
    S = 2 * abs(red_blue_matrix[12] - red_blue_matrix[22]) + abs(green_matrix[7] - green_matrix[17])
    
    minimum = min([N, E, W, S])
    if N == minimum:
        g = (green_matrix[7]  * 3 + green_matrix[17] + red_blue_matrix[12] - red_blue_matrix[2]) / 4
    elif E == minimum:
        g = (green_matrix[13] * 3 + green_matrix[11] + red_blue_matrix[12] - red_blue_matrix[14]) / 4
    elif W == minimum:
        g = (green_matrix[11] * 3 + green_matrix[13] + red_blue_matrix[12] - red_blue_matrix[10]) / 4
    elif S == minimum:
        g = (green_matrix[17] * 3 + green_matrix[7]  + red_blue_matrix[12] - red_blue_matrix[22]) / 4
    
    return normalize_color(g)
        
@njit
def hue_transit(L1, L2, L3, V1, V3):
    if (L1 < L2 and L2 < L3) or (L1 > L2 and L2 > L3):
        return V1 + (V3 - V1) * (L2 - L1) / (L3 - L1)
    else:
        return (V1 + V3) / 2 + (L2 - (L1 + L3) / 2) / 2
        
@njit
def second_step(y, green_matrix, red_matrix, blue_matrix):
    
    if y % 2 == 1:
        r = hue_transit(green_matrix[1], green_matrix[4], green_matrix[7], red_matrix[1], red_matrix[7])
        b = hue_transit(green_matrix[3], green_matrix[4], green_matrix[5], blue_matrix[3], blue_matrix[5])
    else:
        r = hue_transit(green_matrix[3], green_matrix[4], green_matrix[5], red_matrix[3], red_matrix[5])
        b = hue_transit(green_matrix[1], green_matrix[4], green_matrix[7], blue_matrix[1], blue_matrix[7])
    return normalize_color(r), normalize_color(b)
    
@njit
def third_step(g, rb):
    NE = abs(rb[8] - rb[16]) + abs(rb[4] - rb[12]) + abs(rb[12] - rb[20]) + abs(g[8] - g[12]) + abs(g[12] - g[16])
    NW = abs(rb[6] - rb[18]) + abs(rb[0] - rb[12]) + abs(rb[12] - rb[24]) + abs(g[6] - g[12]) + abs(g[12] - g[18])
    if NE < NW:
        return normalize_color(hue_transit(g[8], g[12], g[16], rb[8], rb[16]))
    else:
        return normalize_color(hue_transit(g[6], g[12], g[18], rb[6], rb[18]))
        
@njit
def PPG(source_image, orig_image):
    img_red = source_image[:, :, 0].copy()
    img_green = source_image[:, :, 1].copy()
    img_blue = source_image[:, :, 2].copy()
    img_rb_sum = img_blue + img_red
    h, w, channels = source_image.shape
    
    target_image = np.zeros((h, w, 3))
    
    
    for y in prange(2, h-2):
        for x in prange(2, w-2):
            if get_color(x, y) != "green":
                img_green[y, x] = first_step(green_matrix=patch_5_5(img_green, x, y), red_blue_matrix=patch_5_5(img_rb_sum, x, y))
                
        
    for y in prange(2, h-2):
        for x in prange(2, w-2):
            if get_color(x, y) == "green":
                img_red[y, x], img_blue[y, x] = second_step(y, green_matrix=patch_3_3(img_green, x, y), red_matrix=patch_3_3(img_red, x, y), blue_matrix=patch_3_3(img_blue, x, y))
            
    for y in range(2, h-2):
        for x in range(2, w-2):
            color = get_color(x, y)
            if color == "blue":
                img_red[y, x] =  third_step(g=patch_5_5(img_green, x, y), rb=patch_5_5(img_rb_sum, x, y))
            elif color == "red":
                img_blue[y, x] = third_step(g=patch_5_5(img_green, x, y), rb=patch_5_5(img_rb_sum, x, y))
                
    for y in prange(2, h-2):
        for x in prange(2, w-2):
            target_image[y, x, 0] = img_red[y, x]
            target_image[y, x, 1] = img_green[y, x]
            target_image[y, x, 2] = img_blue[y, x]

    return target_image
