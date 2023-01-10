import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


def generate_o3d_pcd(size, rgb_path, depth_path, K, visualize=False):
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
        plt.subplot(1, 2, 1)
        plt.title('Redwood grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Redwood depth image')
        plt.imshow(rgbd_image.depth)
        plt.show()
        o3d.visualization.draw_geometries([pcd])

    return pcd


if __name__ == '__main__':
    rgb_path = "../data/rgb2.png"
    depth_path = "../data/depth2.png"

    K = dict()
    K['f'] = 525.0
    K['cx'] = 319.5
    K['cy'] = 239.5
    K['scaling_factor'] = 5000

    img_size = [640, 480]

    generate_o3d_pcd(img_size, rgb_path, depth_path, K, True)