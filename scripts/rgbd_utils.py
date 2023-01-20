import cv2 as cv
import numpy as np

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def img2float(img: cv.Mat) -> cv.Mat:
    """ This function converts input image to normalized float """
    return cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)


def bilinear_interpolation(img: cv.Mat, x: float, y: float, width: int, height: int) -> float:
    """ This function perform the bilinear interpolation intensity for input coordinates that do not exactly 
    locate on the pixel coordinate (in other words, not interger).
    (x0, y0) ---- (h0_x, h0_y) -------- (x1, y0)
       |               |                   |
       |             (x, y)                |
       |               |                   | 
       |               |                   |
    (x0, y1) ---- (h1_x, h1_y) -------- (x1, y1)
    """
    
    # Consider the pixel as invalid to begin with
    valid = np.nan

    # Get the four corner coordinates for the current floating point coordinate x, y
    x0 = np.floor(x).astype(np.uint16)
    y0 = np.floor(y).astype(np.uint16)
    x1 = x0 + 1
    y1 = y0 + 1

    # Perform repeated linear interpolation.
    # Compute weights for each corner, inversely propotional to the distance
    x1_weight = x - x0 #  / 1, since grid size is in general 1
    y1_weight = y - y0
    x0_weight = x1 - x
    y0_weight = y1 - y

    # Check if the warped points lie within the image
    if x0 < 0 or x0 >= width:
        return np.nan
    if x1 < 0 or x1 >= width:
        return np.nan
    if y0 < 0 or y0 >= height:
        return np.nan
    if y1 < 0 or y1 >= height:
        return np.nan

    # Bilinear interpolate intensities
    h0 = x0_weight * img.item((y0, x0)) + x1_weight * img.item((y0, x1))
    h1 = x0_weight * img.item((y1, x0)) + x1_weight * img.item((y1, x1))
    valid = y0_weight * h0 + y1_weight * h1

    return valid
    

def bilinear_interpolation_test(img, x, y, width, height):
    """ This function is taken from https://github.com/krrish94/dvo_python, which serves here as a counter check 
    for my implementation
    """
    # Consider the pixel as invalid, to begin with
    valid = np.nan

    # Get the four corner coordinates for the current floating point values x, y
    x0 = np.floor(x).astype(np.uint16)
    y0 = np.floor(y).astype(np.uint16)
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute weights for each corner location, inversely proportional to the distance
    x1_weight = x - x0
    y1_weight = y - y0
    x0_weight = 1 - x1_weight
    y0_weight = 1 - y1_weight   

    # Check if the warped points lie within the image
    if x0 < 0 or x0 >= width:
        x0_weight = 0
    if x1 < 0 or x1 >= width:
        x1_weight = 0
    if y0 < 0 or y0 >= height:
        y0_weight = 0
    if y1 < 0 or y1 >= height:
        y1_weight = 0

    # Bilinear weights
    w00 = x0_weight * y0_weight
    w10 = x1_weight * y0_weight
    w01 = x0_weight * y1_weight
    w11 = x1_weight * y1_weight

    # Bilinearly interpolate intensities
    sum_weights = w00 + w10 + w01 + w11
    total = 0
    if w00 > 0:
        total += img.item((y0, x0)) * w00
    if w01 > 0:
        total += img.item((y1, x0)) * w01
    if w10 > 0:
        total += img.item((y0, x1)) * w10
    if w11 > 0:
        total += img.item((y1, x1)) * w11

    if sum_weights > 0:
        valid = total / sum_weights

    return valid


def ConvertDepthFromEXR(exr):
    if len(exr.shape) > 2:
        exr = exr[:, :, 0]
    
    depth = cv.normalize(exr, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return depth


if __name__ == '__main__':
    dir = '../data'
    for file in os.listdir(dir):
        if file.endswith(".exr"):
            print(os.path.join(dir, file))
            exr = cv.imread(os.path.join(dir, file), cv.IMREAD_UNCHANGED)
            depth = ConvertDepthFromEXR(exr)
            new_file_name = os.path.join(dir, file)[:-4] + '.png'
            cv.imwrite(new_file_name, depth)
