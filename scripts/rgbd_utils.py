import cv2 as cv
import numpy as np
import torch

import open3d as o3d
# import taichi as ti
# ti.init()

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
    # print(y0, x0, y1, x1)
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


def constructCameraMatrix(K):
    return np.array([[K['f'], 0,      K['cx']],
                     [0,      K['f'], K['cy']],
                     [0,      0,      1]])


def depth_ambigious_backprojection(
    height: int, 
    width: int, 
    intrinsics: torch.Tensor
) -> torch.Tensor:

    device = intrinsics.device
    intrinsics = intrinsics.squeeze().type(torch.FloatTensor)       # ([4, 4])

    # Back-projection
    K_inv = torch.linalg.inv(intrinsics[:3, :3]) # shape([3, 3])

    n_rows = torch.arange(width).view(1, width).repeat(height, 1).to(device)
    n_cols = torch.arange(height).view(height, 1).repeat(1, width).to(device)
    pixel = torch.stack((n_rows, n_cols, torch.ones(height, width, device=device)), dim=2).type(torch.FloatTensor) # shape ([H, W, 3])

    # Cache computed 3D points
    depth_ambigious_point3d = torch.matmul(pixel.view(height, width, 1, 3), K_inv.T).squeeze()  # ([H, W, 3])

    return depth_ambigious_point3d


def backprojection(depth, K):
    inv_K_mat = np.linalg.inv(constructCameraMatrix(K))
    n_rows = np.arange(rgb.shape[1]).reshape(1, rgb.shape[1]).repeat(rgb.shape[0], axis=0)
    n_cols = np.arange(rgb.shape[0]).reshape(rgb.shape[0], 1).repeat(rgb.shape[1], axis=1)
    pixel = np.stack((n_rows, n_cols, np.ones(rgb.shape[:2])), axis=0)

    points_3d = (inv_K_mat @ pixel.reshape(3, -1))

    nonzero_depth = (depth.reshape(1, -1) != 0).squeeze()
    points_3d = (depth.reshape(1, -1)[:, nonzero_depth] / K['scaling_factor'] * points_3d[:, nonzero_depth]).T

    # height, width = depth.size()[-2:]
    # K = torch.from_numpy(constructCameraMatrix(K))
    # print(K, 'with shape: ', K.size())
    # points_3d = depth_ambigious_backprojection(height, width, K)

    # nonzero_depth = (depth !=0)
    # points_3d = points_3d[nonzero_depth, :]
    # points_3d = depth[nonzero_depth].view(-1, 1) * points_3d

    return points_3d, nonzero_depth


if __name__ == '__main__':
    # dir = '../data/cofusion/'
    # for file in os.listdir(dir):
    #     if file.endswith(".exr"):
    #         print(os.path.join(dir, file))
    #         exr = cv.imread(os.path.join(dir, file), cv.IMREAD_UNCHANGED)
    #         depth = ConvertDepthFromEXR(exr)
    #         new_file_name = os.path.join(dir, file)[:-4] + '.png'
    #         cv.imwrite(new_file_name, depth)

    import time, pcd_utils, direct_image_alignment
    # Compare the execuate time
    dir = '../data/cofusion/'
    rgb = dir + 'Color0001.png'
    depth = dir + 'Depth0001.png'
    rgb = cv.cvtColor(cv.imread(rgb, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB)
    depth = img2float(cv.imread(depth, cv.IMREAD_UNCHANGED))

    f = 360.0 
    cx = 320.0
    cy = 240.0
    scaling_factor = 5000
    K = dict()
    K['f'] = f
    K['cx'] = cx
    K['cy'] = cy
    K['scaling_factor'] = scaling_factor

    start = time.time()
    points, mask = backprojection(depth, K)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        rgb.astype(np.float64).reshape(-1, 3)[mask, :] / 255.0)
    print("matrix time ", time.time() - start)
    o3d.visualization.draw_geometries([pcd])

    start = time.time()
    pcd1 = direct_image_alignment.rgbd_pointcloud(rgb, depth, f, cx, cy, scaling_factor)
    print("Iteration time ", time.time() - start)
    o3d.visualization.draw_geometries([pcd1])

    start = time.time()
    pcd2 = pcd_utils.generate_o3d_pcd([480, 640], rgb_path='../data/cofusion/Color0001.png', depth_path='../data/cofusion/Depth0001.png', K=K, visualize=False)
    print("o3d time ", time.time() - start)
    o3d.visualization.draw_geometries([pcd2])

