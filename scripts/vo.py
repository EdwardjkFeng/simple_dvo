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
import pcd_utils

import open3d as o3d


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
    parser.add_argument('-numPyramidLevels', type=int, help='Number of the levels used in the pyramid', default=1)
    parser.add_argument('-stepsize', type=float, help='Stepsize for gradient descent solver', default=1e-6)
    parser.add_argument('-numIters', type=int, help='Number of iterations to run each optimization \
        routine for', default=50)
    parser.add_argument('-tol', type=float, help='Tolerance parameter for gradient-based optimization. \
        Specifies the amount by which loss must change across successive iterations', default=1e-8)

    args = parser.parse_args()

    # python vo.py -datapath ../data/cofusion -startFrameRGB Color0001 -startFrameDepth Depth0001 -endFrameRGB Color0002 -endFrameDepth Depth0002

    return args


def main(args):

    # Read images
    img_prev = cv.cvtColor(cv.imread(os.path.join(args.datapath, args.startFrameRGB + '.png')), cv.COLOR_BGR2RGB)
    img_cur = cv.cvtColor(cv.imread(os.path.join(args.datapath, args.endFrameRGB + '.png')), cv.COLOR_BGR2RGB)
    img_depth_prev = cv.imread(os.path.join(args.datapath, args.startFrameDepth + '.png'), cv.IMREAD_GRAYSCALE)
    img_depth_cur = cv.imread(os.path.join(args.datapath, args.endFrameDepth + '.png'), cv.IMREAD_GRAYSCALE)

    img_gray_prev = cv.cvtColor(img_prev, cv.COLOR_RGB2GRAY)
    img_gray_cur = cv.cvtColor(img_cur, cv.COLOR_RGB2GRAY)
    # Convert the intensity images to float
    # img_prev = rgbd_utils.img2float(img_prev)
    # img_cur = rgbd_utils.img2float(img_cur)
    img_gray_prev = rgbd_utils.img2float(img_gray_prev)
    img_gray_cur = rgbd_utils.img2float(img_gray_cur)
    
    # Use default camera intrinsics
    f = 360 #525.0
    cx = 320 #319.5
    cy = 240 #239.5
    scaling_factor = 1 #5000

    # Construct a downsampled pyramid using the specified number of pyramid levels
    pyramid_gray, pyramid_depth, pyramid_intrinsics = image_pyramid.buildPyramid(img_gray_prev, \
        img_depth_prev, num_levels=args.numPyramidLevels, focal_legth=f, cx=cx, cy=cy)

    # Set camera parameters
    K = dict()
    K['f'] = f
    K['cx'] = cx
    K['cy'] = cy
    K['scaling_factor'] = scaling_factor

    # Compute residuals
    xi_init = np.zeros((6, 1))

    residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, img_depth_prev, img_gray_cur, K, xi_init)
    # residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, img_depth_prev, img_gray_cur, img_depth_cur, K, xi_init)
    # print(cache_point3d.shape) # Debug

    source_points = np.reshape(cache_point3d, (-1, 3))
    # print(points.shape)

    s_pcd = o3d.geometry.PointCloud()
    s_pcd.points = o3d.utility.Vector3dVector(source_points)
    s_colors = np.reshape(rgbd_utils.img2float(img_prev), (-1, 3))
    s_pcd.colors = o3d.utility.Vector3dVector(s_colors)
    # o3d.visualization.draw_geometries([s_pcd])

    t_pcd = direct_image_alignment.rgbd_pointcloud(img_cur, img_depth_cur, f, cx, cy, scaling_factor)

    # # Test image gradient computations
    # grad_x, grad_y = direct_image_alignment.computeImageGradient(img_gray_prev)
    # # cv.imshow('img', img_gray_prev)
    # # cv.imshow('grad_x', grad_x)
    # # cv.imshow('grad_y', grad_y)
    # # cv.waitKey(1000 * 20)

    # # Test Jacobian computations
    # J = direct_image_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, img_depth_cur, K, xi_init, residuals, cache_point3d)

    # Simple gradient descent test
    stepsize = args.stepsize
    max_iters = args.numIters
    tol = args.tol
    err_prev = 1e24
    # for it in range(max_iters):
    #     residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, img_depth_prev, img_gray_cur, K, xi_init)
    #     J = direct_image_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, K, xi_init, residuals, cache_point3d)
    #     # residuals, cache_point3d = direct_image_alignment.computeResiduals(img_gray_prev, img_depth_prev, img_gray_cur, img_depth_cur, K, xi_init)
    #     # J = direct_image_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, img_depth_cur, K, xi_init, residuals, cache_point3d)
    #     # Normalize the error and the jacobian
    #     err_cur = 0.5 * (1 / (img_gray_cur.shape[0] * img_gray_cur.shape[1])) * np.sum(np.dot(residuals.T, residuals))
    #     grad = (1 / (img_gray_cur.shape[0] * img_gray_cur.shape[1])) * np.reshape(np.sum(J, axis=(0, 1)).T, (6, 1))
    #     print("Error: ", err_cur)
    #     print('Jacobian: ', np.sum(J, axis=(0, 1)))
    #     xi_init += stepsize * grad

    #     # T_init = se3utils.SE3_exp(xi_init) * se3utils.SE3_exp(grad)
    #     # print('T: ', T_init)
    #     T_init = se3utils.SE3_exp(xi_init)
    #     pcd_utils.draw_registration_result_original_color(s_pcd, t_pcd,
    #                                         T_init)
    #     # xi_init = se3utils.SE3_log(T_init)
    #     if np.abs(err_prev - err_cur) < tol:
    #         break
    #     err_prev = err_cur

    # T = se3utils.SE3_exp(xi_init)
    # print('T: ', se3utils.SE3_exp(xi_init))
    # pcd_utils.draw_registration_result_original_color(s_pcd, t_pcd,
    #                                         T)

    gray_prev, depth_prev, pyramid_intrinsics = image_pyramid.buildPyramid(img_gray_prev, img_depth_prev, num_levels=args.numPyramidLevels, focal_legth=f, cx=cx, cy=cy)

    gray_cur, depth_cur, _ = image_pyramid.buildPyramid(img_gray_cur, img_depth_cur, num_levels=args.numPyramidLevels, focal_legth=f, cx=cx, cy=cy)
            
    for i in range(1, args.numPyramidLevels + 1):
        T, xi_init = direct_image_alignment.do_gaussian_newton(gray_prev[-i], depth_prev[-i], gray_cur[-i], xi_init, pyramid_intrinsics[-i], max_iters)


    # T = direct_image_alignment.do_gaussian_newton(img_gray_prev, img_depth_prev, img_gray_cur, xi_init, K, max_iters)
    # print('T: ', T)
    pcd_utils.draw_registration_result_original_color(s_pcd, t_pcd, T)



if __name__ == '__main__':
    args = parse_arg()
    main(args)



