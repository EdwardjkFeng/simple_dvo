"""
Functions to compute photometric error residuals and jacobians 
"""

import numpy as np

import rgbd_utils 
import se3utils

import open3d as o3d

def rgbd_pointcloud(rgb, depth, focal_legth, cx, cy, scaling_factor):
    """
    Takes in an intensity image and a registered depth image, and outputs a
    pointcloud. Intrinsics must be provided, else we use the defaults.
    """
    pointcloud = o3d.geometry.PointCloud()
    points = np.zeros((rgb.shape[0], rgb.shape[1], 3))
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            # intensity = rgb.item((u, v))
            Z = depth.item((v, u)) / scaling_factor
            if Z == 0:
                continue

            X = (u - cx) * Z / focal_legth
            Y = (v - cy) * Z / focal_legth
            # pointcloud.append((X, Y, Z, intensity)) 
            points[v, u, :] = [X, Y, Z]
    points = np.reshape(points, (-1, 3))
    colors = np.reshape(rgb, (-1, 3))
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pointcloud])
    
    return pointcloud


def computeResiduals(gray_prev, depth_prev, gray_cur, K, xi):
    """ Compute photometric error
    Compute the image alignment error (residuals). Takes in the previous intensity image and 
    first backprojects it to 3D to obtain a pointcloud. This pointcloud is then rotated by an 
    SE(3) transform "xi", and then projected down to the current image. After this step, an intensity
    interpolation step is performed and we compute the error between the projected image and the actual
    current intensity image.

    While performing the residuals, also cache information to speedup Jacobian computation.
    """

    width = gray_cur.shape[1]
    height = gray_cur.shape[0]
    residuals = np.zeros(gray_cur.shape, dtype=np.float32)

    # Cache to store computed 3D points
    cache_point3d = np.zeros((height, width, 3), dtype=np.float32)

    one_by_f = 1. / K['f']

    # Use the SE(3) Exponential map to compute a 4 x 4 matrix from the vector xi
    T = se3utils.SE3_exp(xi)

    K_mat = np.array([[K['f'], 0,      K['cx']],
                      [0,      K['f'], K['cy']],
                      [0,      0,      1]])

    # Warp each point in the previous image, to the current image
    for v in range(height):
        for u in range(width):
            intensity_prev = gray_prev.item((v, u))
            Z = depth_prev.item((v, u)) / K['scaling_factor']
            if Z <= 0:
                continue
            
            Y = one_by_f * Z * (v - K['cy'])
            X = one_by_f * Z * (u - K['cx'])
            # Transform the 3D point 
            point_3d = np.dot(T[0:3, 0:3], np.asarray([X, Y, Z])) + T[0:3, 3]
            point_3d = np.reshape(point_3d, (3, 1))
            cache_point3d[v, u, :] = np.reshape(point_3d, (3))
            # Project it down to 2D
            point_2d_warped = np.dot(K_mat, point_3d)
            px = point_2d_warped[0] / point_2d_warped[2]
            py = point_2d_warped[1] / point_2d_warped[2]

            # Interpolate the intensity value bilinearly 
            intensity_warped = rgbd_utils.bilinear_interpolation(gray_cur, px[0], py[0], width, height)

            # If the pixel is valid (i.e., interpolation return a non-NaN value), compute residual
            if not np.isnan(intensity_warped):
                residuals.itemset((v, u), intensity_prev - intensity_warped)

    return residuals, cache_point3d


def computeResiduals_color(gray_prev, depth_prev, gray_cur, depth_cur, K, xi):
    """ Compute photometric error and geometric error
    Compute the image alignment error (residuals). Takes in the previous intensity image and 
    first backprojects it to 3D to obtain a pointcloud. This pointcloud is then rotated by an 
    SE(3) transform "xi", and then projected down to the current image. After this step, an intensity
    interpolation step is performed and we compute the error between the projected image and the actual
    current intensity image.

    While performing the residuals, also cache information to speedup Jacobian computation.
    """

    width = gray_cur.shape[1]
    height = gray_cur.shape[0]
    residuals = np.zeros(gray_cur.shape, dtype=np.float32)
    geometric_lambda = 0.5
    photometric_lambda = 1 - geometric_lambda

    # Cache to store computed 3D points
    cache_point3d = np.zeros((height, width, 3), dtype=np.float32)

    one_by_f = 1. / K['f']

    # Use the SE(3) Exponential map to compute a 4 x 4 matrix from the vector xi
    T = se3utils.SE3_exp(xi)

    K_mat = np.array([[K['f'], 0,      K['cx']],
                      [0,      K['f'], K['cy']],
                      [0,      0,      1]])

    # Warp each point in the previous image, to the current image
    for v in range(height):
        for u in range(width):
            intensity_prev = gray_prev.item((v, u))
            d_prev = depth_prev.item((v, u))
            Z = depth_prev.item((v, u)) / K['scaling_factor']
            if Z <= 0:
                continue
            
            Y = one_by_f * Z * (v - K['cy'])
            X = one_by_f * Z * (u - K['cx'])
            # Transform the 3D point 
            point_3d = np.dot(T[0:3, 0:3], np.asarray([X, Y, Z])) + T[0:3, 3]
            point_3d = np.reshape(point_3d, (3, 1))
            cache_point3d[v, u, :] = np.reshape(point_3d, (3))
            # Project it down to 2D
            point_2d_warped = np.dot(K_mat, point_3d)
            px = point_2d_warped[0] / point_2d_warped[2]
            py = point_2d_warped[1] / point_2d_warped[2]

            # Interpolate the intensity value bilinearly 
            intensity_warped = rgbd_utils.bilinear_interpolation(gray_cur, px[0], py[0], width, height)

            depth_warped = rgbd_utils.bilinear_interpolation(depth_cur, px[0], py[0],width, height)

            # If the pixel is valid (i.e., interpolation return a non-NaN value), compute residual
            if not np.isnan(intensity_warped) and not np.isnan(depth_warped):
                residuals.itemset((v, u), np.sqrt(photometric_lambda) * (intensity_prev - intensity_warped) + np.sqrt(geometric_lambda) * (d_prev - depth_warped))

    return residuals, cache_point3d

def computeImageGradient(img):
    """
    We use a simple form for the image gradient. For instance, a gradient along the X-direction
    at location (y, x) is computed as I(y, x + 1) - I(y, x -1).
    """

    gradX = np.zeros(img.shape, dtype=np.float32)
    gradY = np.zeros(img.shape, dtype=np.float32)

    width = img.shape[1]
    height = img.shape[0]

    # Exploit the fact that we can perform matrix operations on images ,to compute gradients quicker
    gradX[:, 1:width-1] = 0.5 * (img[:, 2:] - img[:, 0:width-2])
    gradY[1:height-1, :] = 0.5* (img[2:, :] - img[0:height-2, :])

    return gradX, gradY


def computeJacobian(gray_prev, depth_prev, gray_cur, K, xi, residuals, cache_point3d):

    width = gray_prev.shape[1]
    height = gray_prev.shape[0]

    K_mat = np.asarray([[K['f'], 0, K['cx']],
                        [0, K['f'], K['cy']],
                        [0, 0, 1]])

    f = K['f']
    cx = K['cx']
    cy = K['cy']

    # Initialize memory to store the Jacobian
    J = np.zeros((height, width, 6))

    # Compute image gradients
    grad_x, grad_y = computeImageGradient(gray_cur)

    # For each pixel, compute one Jacobian term
    for v in range(height):
        for u in range(width):
            X = cache_point3d.item((v, u, 0))
            Y = cache_point3d.item((v, u, 1))
            Z = cache_point3d.item((v, u, 2))
            if Z <= 0:
                continue

            J_img = np.reshape(np.asarray([[grad_x[v, u], grad_y[v, u]]]), (1, 2))
            J_pi = np.reshape(np.asarray([[f/Z, 0, -f*X/(Z*Z)], [0, f/Z, -f*Y/(Z*Z)]]), (2, 3))

            J_w = np.reshape(np.asarray([[f/Z, 0, -f*X/(Z*Z), -f*(X*Y)/(Z*Z), f*(1+(X*X)/(Z*Z)), -f*Y/Z], [0, f/Z, -f*Y/(Z*Z), -f*(1+(Y*Y)/(Z*Z)), f*X*Y/(Z*Z), f*X/Z]]), (2, 6))
            J_exp = np.concatenate((np.eye(3), se3utils.SO3_hat(-np.asarray([X, Y, Z]))), axis=1)
            J_exp = np.dot(J_exp, se3utils.SE3_left_jacobian(xi))
            # J[v, u, :] = residuals[v, u] * np.reshape(np.dot(J_img, np.dot(J_pi, J_exp)), (6))
            J[v, u, :] = - np.reshape(np.matmul(J_img, J_w), (6))
            if not np.isfinite(J[v, u, 0]):
                J[v, u, :] = 0

    return J


def computeJacobian_color(gray_prev, depth_prev, gray_cur, depth_cur, K, xi, residuals, cache_point3d):

    width = gray_prev.shape[1]
    height = gray_prev.shape[0]

    K_mat = np.asarray([[K['f'], 0, K['cx']],
                        [0, K['f'], K['cy']],
                        [0, 0, 1]])

    f = K['f']
    cx = K['cx']
    cy = K['cy']

    geometric_lambda = 0.5
    photometric_lambda = 1 - geometric_lambda

    # Initialize memory to store the Jacobian
    J = np.zeros((height, width, 6))

    # Compute image gradients
    grad_x, grad_y = computeImageGradient(gray_cur)
    grad_u, grad_v = computeImageGradient(depth_cur)

    # For each pixel, compute one Jacobian term
    for v in range(height):
        for u in range(width):
            X = cache_point3d.item((v, u, 0))
            Y = cache_point3d.item((v, u, 1))
            Z = cache_point3d.item((v, u, 2))
            if Z <= 0:
                continue

            J_img = np.reshape(np.asarray([[grad_x[v, u], grad_y[v, u]]]), (1, 2))
            J_pi = np.reshape(np.asarray([[f/Z, 0, -f*X/(Z*Z)], [0, f/Z, -f*Y/(Z*Z)]]), (2, 3))
            J_exp = np.concatenate((np.eye(3), se3utils.SO3_hat(-np.asarray([X, Y, Z]))), axis=1)
            J_exp = np.dot(J_exp, se3utils.SE3_left_jacobian(xi))

            J_depth = np.reshape(np.asarray([[grad_u[v, u], grad_v[v, u]]]), (1, 2))
            J[v, u, :] = residuals[v, u] * (np.sqrt(photometric_lambda)* np.reshape(np.dot(J_img, np.dot(J_pi, J_exp)), (6)) + np.sqrt(geometric_lambda) * np.reshape(np.dot(J_depth, np.dot(J_pi, J_exp)), (6)))

    
    return J


def do_gaussian_newton(img_gray_prev, img_depth_prev, img_gray_cur, xi, K, max_iters):
    # Gaussian Newton solver for direct image alignment
    xi_prev = xi
    # H = np.zeros((6, 6)) # Hessian for GN optimization
    # inc = np.zeros((6, 1)) # step increments

    err_prev = np.inf
    for iter in range(max_iters):
        residuals, pcd = computeResiduals(img_gray_prev, img_depth_prev, img_gray_cur, K, xi_prev)

        J = computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, K, xi, residuals, pcd).reshape(-1, 6)

        Jt = J.transpose()

        err = np.sum(np.matmul(residuals.transpose(), residuals))

        b = np.matmul(Jt, residuals.reshape(-1))
        H = np.matmul(Jt, J)
        inc = - np.linalg.solve(H, b)

        xi_prev = xi
        xi = se3utils.SE3_log(se3utils.SE3_exp(xi) @ se3utils.SE3_exp(inc))

        if (err / err_prev > 0.995):
            break
        err_prev = err

    return se3utils.SE3_exp(xi), xi