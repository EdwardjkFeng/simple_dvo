import copy
import numpy as np
import open3d as o3d 


def generate_o3d_pcd(rgb_path, depth_path, H, W, K: np.ndarray, depth_scale: int=1):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width = W, height = H, intrinsic_matrix = K)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=depth_scale, convert_rgb_to_intensity = False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    return pcd

def rgbd_to_pcd(rgb, depth, K):
    H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    fx_inv = 1 / fx
    fy_inv = 1 / fy

    o3d_pcd = o3d.geometry.PointCloud()
    pcd = np.zeros((H, W, 3))

    for y in range(H):
        for x in range(W):
            Z_curr = depth[y, x]
            if Z_curr <= 0:
                continue
            X_curr = fx_inv * Z_curr * (x - cx)
            Y_curr = fy_inv * Z_curr * (y - cy)

            pcd[y, x, :] = [X_curr, Y_curr, Z_curr]
    
    pcd = pcd.reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pcd


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def generate_o3d_pcd_test(size, rgb_path, depth_path, K, visualize=False):
    # Read rgb and depth
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    print(rgb)

    # Read camera intrinsic
    K_mat = np.array([[K['f'], 0,      K['cx']],
                      [0,      K['f'], K['cy']],
                      [0,      0,      1]])
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width = size[0], height = size[1], intrinsic_matrix = K_mat)
    print(camera_intrinsic)

    # Create rgbd
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity = False)

    # Create point cloud from rgbd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    if visualize:
        pass

    return pcd