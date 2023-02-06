"""
Helper function to construct image pyramids
"""

import cv2 as cv
import numpy as np


def downsampleGray(img: cv.Mat) -> cv.Mat:
    """ Function to downsample an intensity (grayscale) image
    The downsampling strategy eventually chosen is a navie block averaging method.
    That is, for each pixel in the target image, we choose a block comprising 4 neighbors
    in the source image, and simply average their intensities. For each target image point 
    (y, x), where x indexes the width and y the height dimensions, we consider the following
    four neighbors: (2*y, 2*x), (2*y+1, 2*x), (2*y, 2*x+1), (2*y+1, 2*x+1).
    Note: The image must be float to begin with.
    """
    img_new = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) / 4.

    return img_new

def downsampleDepth(img: cv.Mat) -> cv.Mat:
    """
    For depth images, we do not average all pixels; rather, we average only pixels with non-zero
    depth values.
    """

    # Perform block-averaging, but not across depth boundaries. (i.e., compute average only over
    # non-zero elements)
    img_ = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=2)
    num_nonzero = np.count_nonzero(img_, axis=2)
    num_nonzero[np.where(num_nonzero == 0)] = 1
    img_new = np.sum(img_, axis=2) / 4 / num_nonzero

    return img_new.astype(np.uint8)


def buildPyramid(gray, depth, num_levels, focal_legth, cx, cy):
    """ Function to construct a pyramid of intensity and depth images with a given level
    
    """

    # Lists to store each level of a pyramid 
    pyramid_gray = []
    pyramid_depth = []
    pyramid_intrinsics = []

    current_gray = gray
    current_depth = depth
    current_f = focal_legth
    current_cx = cx
    current_cy = cy

    # Build levels of the pyramid 
    for level in range(num_levels):
        pyramid_gray.append(current_gray)
        pyramid_depth.append(current_depth)
        current_K = dict() # Use a dict to store the intrinsic parameters
        current_K['f'] = current_f / 2
        current_K['cx'] = current_cx / 2
        current_K['cy'] = current_cy / 2
        current_K['scaling_factor'] = 5000
        pyramid_intrinsics.append(current_K)
        if level < num_levels - 1:
            current_gray = downsampleGray(current_gray)
            current_depth = downsampleDepth(current_depth)
        
    return pyramid_gray, pyramid_depth, pyramid_intrinsics
