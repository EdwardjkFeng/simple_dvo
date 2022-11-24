import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import rgbd_utils
import image_pyramid
import direct_image_alignment
import se3utils


# Parse command-line arguments 
def parse_arg():

    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', help='Path to a TUM RGB-D bechmark sequence', required=True)
    parser.add_argument('-startFrameRGB', help='Filename (sans the .png extension) of the first \
        RGB frame to be processed', required = True)
    parser.add_argument('-startFrameDepth', help='Filename (sans the .png extension) of the first \
        depth frame to be processed', required = True)
    parser.add_argument('-endFrameRGB', help='Filename (sans the .png extension) of the last \
        RGB frame to be processed', required = True)
    parser.add_argument('-endFrameDepth', help='Filename (sans the .png extension) of the last \
        depth frame to be processed', required = True)
    parser.add_argument('-numPyramidLevels', help='Number of the levels used in the pyramid', default=3)
    parser.add_argument('-stepsize', help='Stepsize for gradient descent solver', default=1e-6)
    parser.add_argument('-numIters', help='Number of iterations to run each optimization \
        routine for', default=50)
    parser.add_argument('-tol', help='Tolerance parameter for gradient-based optimization. \
        Specifies the amount by which loss must change across successive iterations', default=1e-8)

    args = parser.parse_args()

    return args


def main(args):

    img_gray_prev = cv.imread(os.path.join(args.datapath, args.startFrameRGB + '.png'), cv.IMREAD_GRAYSCALE)
    img_gray_cur = cv.imread(os.path.join(args.datapath, args.endFrameRGB + '.png'), cv.IMREAD_GRAYSCALE)
    img_depth_prev = cv.imread(os.path.join(args.datapath, args.startFrameDepth + '.png'), cv.IMREAD_GRAYSCALE)
    img_depth_cur = cv.imread(os.path.join(args.datapath, args.endFrameDepth + '.png'), cv.IMREAD_GRAYSCALE)

    # Convert the intensity images to float
    img_gray_prev = rgbd_utils.img2float(img_gray_prev)
    img_gray_cur = rgbd_utils.img2float(img_gray_cur)
    
    # Use default camera intrinsics
    f = 525.0
    cx = 319.5
    cy = 239.5
    scaling_factor = 5000

    # Construct a downsampled pyramid using the specified number of pyramid levels
    pyramid_gray, pyramid_depth, pyramid_intrinsics = image_pyramid.buildPyramid(img_gray_prev, \
        img_depth_prev, num_levels=args.numPyramidLevels, focal_legth=f, cx=cx, cy=cy)

    # Compute residuals 
    K = dict()
    K['f'] = f
    K['cx'] = cx
    K['cy'] = cy
    K['scaling_factor'] = scaling_factor
    xi_init = np.zeros((6, 1))

    residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, img_depth_prev, \
        img_gray_cur, K, xi_init)

    # Test image gradient computations
    grad_x, grad_y = direct_image_alignment.computeImageGradient(img_gray_prev)
    cv.imshow('img', img_gray_prev)
    cv.imshow('grad_x', grad_x)
    cv.imshow('grad_y', grad_y)
    cv.waitKey(1000 * 20)

    # Test Jacobian computations
    J = direct_image_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, K, xi_init, residuals, cache_point3d)

    # Simple gradient descent test
    stepsize = 1e-6
    max_iters = 100
    tol = 1e-8
    err_prev = 1e24
    for it in range(max_iters):
        residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, \
            img_depth_prev, img_gray_cur, K, xi_init)
        J = direct_image_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, \
            K, xi_init, residuals, cache_point3d)
        # Normalize the error and the jacobian
        err_cur = 0.5 * (1 / (img_gray_cur.shape[0] * img_gray_cur.shape[1])) * np.sum(np.abs(residuals))
        grad = (1 / (img_gray_cur.shape[0] * img_gray_cur.shape[1])) * np.reshape(np.sum(J, axis=(0, 1)).T, (6, 1))
        print("Error: ", err_cur)
        print('Jacobian: ', np.sum(J, axis=(0, 1)))
        xi_init += stepsize * grad
        if np.abs(err_prev - err_cur) < tol:
            break
        err_prev = err_cur

    print('xi: ', xi_init)


if __name__ == '__main__':
    args = parse_arg()
    main(args)



