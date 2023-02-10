import numpy as np
import cv2 as cv
import sophus
import open3d as o3d
import imageio.v2 as imageio

import torch

import rgbdutils
import pcdutils


INITIAL_SIGMA = 5.0
DEFAUFLT_DOF = 5.0

def calc_gradient(img: np.ndarray):
    H, W = img.shape
    grad_x = np.zeros(img.shape, dtype=np.float32)
    grad_y = np.zeros(img.shape, dtype=np.float32)


    grad_x[:, 1:W-1] = 0.5 * (img[:, 2:] - img[:, 0:W-2])
    grad_y[1:H-1, :] = 0.5 * (img[2:, :] - img[0:H-2, :])

    return grad_x, grad_y

def calc_residuals(prev_img, curr_img, curr_depth, K, transform):
    H, W = prev_img.shape
    residuals = np.zeros((H, W), dtype=np.float32)
    cached_points3d = np.zeros((H, W, 3), dtype=np.float32)
    cached_points2d = np.zeros((H, W, 2), dtype=np.float32)

    R = transform[:3, :3].reshape(3, 3)
    t = transform[:3, 3].reshape(3, 1)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    fx_inv = 1 / fx
    fy_inv = 1 / fy

    for y in range(H):
        for x in range(W):
            Z_curr = curr_depth[y, x]
            X_curr = fx_inv * Z_curr * (x - cx)
            Y_curr = fy_inv * Z_curr * (y - cy)
            point3d_curr = np.array([X_curr, Y_curr, Z_curr]).reshape((3, 1))
            point3d_warped = np.matmul(R, point3d_curr) + t

            if point3d_warped[2] > 0:
                cached_points3d[y, x] = point3d_warped.squeeze()
                point2d_warped = np.matmul(K,  point3d_warped)
                point2d_warped = (point2d_warped[:2] / point2d_warped[2]).reshape((2, 1))
                cached_points2d[y, x] = point2d_warped.squeeze()
                u, v = point2d_warped

                I_warped = rgbdutils.bilinear_interpolation(prev_img, u, v, W, H)
                if not np.isnan(I_warped):
                    residuals[y, x] = I_warped - curr_img[y, x]

    return residuals, cached_points3d, cached_points2d


def calc_Jacobian(prev_img, points3d_curr, points2d_curr, K):
    H, W = prev_img.shape
    # Compute the gradient of the previous intensity map
    grad_x, grad_y = calc_gradient(prev_img)

    fx = K[0, 0]
    fy = K[1, 1]

    J = np.zeros((H, W, 6), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            X, Y, Z = points3d_curr[y, x]
            if Z <= 0:
                continue
            J_w = np.array(
                [[fx/Z, 0, -fx*X/(Z*Z), -fx*(X*Y)/(Z*Z), fx*(1 + (X*X)/(Z*Z)), -fx*Y/Z],
                 [0, fy/Z, -fy*Y/(Z*Z), -fy*(1 + (Y*Y)/(Z*Z)), fy*X*Y/(Z*Z), fy*X/Z]]
            ).reshape((2, 6))
            u, v = points2d_curr[y, x]
            dx = rgbdutils.bilinear_interpolation(grad_x, u, v, W, H)
            if np.isnan(dx):
                continue
            dy = rgbdutils.bilinear_interpolation(grad_y, u, v, W, H)
            if np.isnan(dy):
                continue
            J_I = np.array([dx, dy]).reshape((1, 2))
            J[y, x] = np.matmul(J_I, J_w)

    return J


def weighting(residual: np.ndarray):
    n = len(residual)
    lambda_init = 1 / (INITIAL_SIGMA * INITIAL_SIGMA)
    lambda_ = lambda_init
    num = 0.0
    dof = DEFAUFLT_DOF
    weights = np.ones(residual.shape)
    itr = 0
    while np.abs(lambda_ - lambda_init) > 1e-3:
        itr += 1
        lambda_init = lambda_
        lambda_ = 0.0
        num = 0.0
        for i in range(n):
            if residual[i] != 0:
                num += 1
                lambda_ += residual[i] * residual[i] * ((dof + 1) / (dof + lambda_init * residual[i] * residual[i]))

        lambda_ /= num
        lambda_ = 1 / lambda_
    
    weights = (dof + 1) / (dof + lambda_ * residual * residual)

    return weights



def solve_linear_system(A: np.ndarray, b: np.ndarray):
    At = A.transpose()
    At_A = np.matmul(At, A) 
    At_A += np.eye(At_A) * 1e-8
    At_A_inv = np.linalg.inv(At_A)

    return np.matmul(At_A_inv.transpose(), np.matmul(At, b))


def solve_Gauss_Nowton(curr_img, prev_img, curr_depth, prev_depth, K, transform, num_iters: int=20):

    err_prev = np.inf
    transform_prev = transform

    # Compute residuals
    for i in range(num_iters):
        r, cached_points3d, cached_points2d = calc_residuals(prev_img, curr_img, curr_depth, K, transform_prev)

        J = calc_Jacobian(prev_img, cached_points3d, cached_points2d, K)

        r = r.reshape(-1, 1)
        J = J.reshape(-1, 6)
        num_valid_entries = np.sum(r != 0)
        # print(r.shape, J.shape, num_valid_entries)

        # Normalize J and r over valid entries
        # r = r / num_valid_entries
        # J = J / num_valid_entries

        weights = weighting(r)
        # r = r * weights
        # J = J * weights
        
        Jt = J.transpose()
        err = np.sum(r.transpose() @ r)
        print('{}. iteration, error = {}'.format(i+1, err))

        b = Jt @ r
        A = Jt @ J
        inc = - np.linalg.solve(A, b)

        transform = np.matmul(transform_prev, sophus.SE3.exp(inc).matrix())
        transform_prev = transform

        # TODO terminate criteria
        delta = np.abs(err - err_prev)
        if delta < 1e-6:
            break
    
        err_prev = err

    return transform

def apply_direct_image_alignment(curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr, initial_transform, num_iters: int = 20):

    # Run iterations for each pyramid
    transform_prev = initial_transform

    num_pyr_levels = len(curr_img_pyr)

    for i in range(1, num_pyr_levels+1):
        curr_img = curr_img_pyr[-i]
        curr_depth = curr_depth_pyr[-i]
        prev_img = prev_img_pyr[-i]
        prev_depth = prev_depth_pyr[-i]
        K = K_pyr[-1]

        transform = solve_Gauss_Nowton(curr_img, prev_img, curr_depth, prev_depth, K, transform_prev, num_iters)
        transform_prev = transform
        
        print('=====================')
        print("{}. pyramid level".format(i))
        print("transform = \n", transform)
        print('=====================')
        
    return transform




if __name__ == '__main__':
    # Set config parameters
    NUM_ITERS = 10
    NUM_PYR_LEVELS = 3
    H = 480
    W = 640

    prev_rgb_path = '../data/cofusion/Color0600.png'
    curr_rgb_path = '../data/cofusion/Color0601.png'
    prev_depth_path = '../data/cofusion/Depth0600.exr'
    curr_depth_path = '../data/cofusion/Depth0601.exr'


    # Read rgb images
    prev_image = cv.imread(prev_rgb_path).astype(np.float32)
    curr_image = cv.imread(curr_rgb_path).astype(np.float32)
    curr_image_rgb = cv.cvtColor(curr_image, cv.COLOR_BGR2RGB)
    prev_image_rgb = cv.cvtColor(prev_image, cv.COLOR_BGR2RGB)
    
    # Convert rgb to grayscale
    curr_image = cv.cvtColor(curr_image, cv.COLOR_BGR2GRAY) / 255
    prev_image = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY) / 255
    # Read depth
    # prev_depth = cv.imread(prev_depth_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    # curr_depth = cv.imread(curr_depth_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    prev_depth = np.asarray(imageio.imread(prev_depth_path), dtype=float)
    curr_depth = np.asarray(imageio.imread(curr_depth_path), dtype=float)
    # Intrinsics
    K_ = np.array(
        [[360.0,  0, 320.0],
         [0,  360.0, 240.0],
         [0,    0,   1]]
    )
    
    depth_scale_factor = 1
    prev_depth = prev_depth / depth_scale_factor
    curr_depth = curr_depth / depth_scale_factor

    # Construct pyramid
    curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr = rgbdutils.construct_pyramid(curr_image, prev_image, curr_depth, prev_depth, K_, NUM_PYR_LEVELS)

    # Apply direct image alignment
    initial_transform = np.eye(4, dtype=np.float32)

    transform = apply_direct_image_alignment(curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr, initial_transform, NUM_ITERS)

    # Visualize the point clouds
    # Generate point clouds for each frame
    # target_pcd = pcdutils.generate_o3d_pcd(prev_rgb_path, prev_depth_path, H, W, K_)
    # source_pcd = pcdutils.generate_o3d_pcd(curr_rgb_path, curr_depth_path, H, W, K_)
    
    target_pcd = pcdutils.rgbd_to_pcd(prev_image_rgb, prev_depth, K_)
    source_pcd = pcdutils.rgbd_to_pcd(curr_image_rgb, curr_depth, K_)
    # Paint uniform color for better comparison
    # red = np.array([1, 0, 0])
    # green = np.array([0, 1, 0])
    # target_pcd.paint_uniform_color(red)
    # source_pcd.paint_uniform_color(green)

    o3d.visualization.draw_geometries([source_pcd, target_pcd])

    # T = np.eye(4)
    # T[:3, 3] += 0.00005

    pcdutils.draw_registration_result_original_color(source_pcd, target_pcd, transformation=transform)