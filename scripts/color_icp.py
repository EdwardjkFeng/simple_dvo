import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

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
    parser.add_argument('-numPyramidLevels', help='Number of the levels used in the pyramid', default=3)
    parser.add_argument('-stepsize', help='Stepsize for gradient descent solver', default=1e-6)
    parser.add_argument('-numIters', help='Number of iterations to run each optimization \
        routine for', default=50)
    parser.add_argument('-tol', help='Tolerance parameter for gradient-based optimization. \
        Specifies the amount by which loss must change across successive iterations', default=1e-8)

    args = parser.parse_args()

    return args


def main(args):
    img_size = [640, 480]

    # Use default camera intrinsics
    K = dict()
    K['f'] = 360 # 525.0
    K['cx'] = 320 # 319.5
    K['cy'] = 240 # 239.5
    K['scaling_factor'] = 5000

    icp_transformation = []
    pcds = []

    rgbs = sorted(glob.glob(os.path.join(args.datapath, "Color*.png")))
    depths = sorted(glob.glob(os.path.join(args.datapath, "Depth*.png")))
    cur_transformation = None
    start = False

    for i in range(len(rgbs) - 1):
        if rgbs[i] == os.path.join(args.datapath, args.startFrameRGB + '.png') and depths[i] == os.path.join(args.datapath, args.startFrameDepth + '.png'):
            start = True
        if start:
            print('source: ', depths[i], 'target: ', depths[i + 1])
            # Generate point clouds
            pcd_source = pcd_utils.generate_o3d_pcd(
                img_size, rgbs[i], depths[i], K, False)
            pcd_target = pcd_utils.generate_o3d_pcd(
                img_size, rgbs[i + 1], depths[i + 1], K, False)

            # Init transfromation
            cur_transformation = np.identity(4)
            # pcd_utils.draw_registration_result_original_color(pcd_source, pcd_target, cur_transformation)

            # Perform colored ICP
            voxel_radius = [0.04, 0.02, 0.01]
            max_iter = [50, 30, 14]
            print("\n3. Colored point cloud registration")
            for scale in range(3):
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                print([iter, radius, scale])

                print("3-1. Downsample with a voxel size %.2f" % radius)
                source_down = pcd_source.voxel_down_sample(radius)
                target_down = pcd_target.voxel_down_sample(radius)

                print("3-2. Estimate normal.")
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                print("3-3. Applying colored point cloud registration")
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, cur_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=iter))
                cur_transformation = result_icp.transformation
                print(result_icp)
            pcd_utils.draw_registration_result_original_color(pcd_source, pcd_target, result_icp.transformation)
            icp_transformation.append(cur_transformation)
            pcds.append(pcd_source)

            print(cur_transformation)
    pcds.append(pcd_target)

    for n in range(len(icp_transformation)):
        cur_idx = n
        pcd = pcds[n]
        while cur_idx < len(icp_transformation):
            pcd = pcd.transform(icp_transformation[cur_idx])
            cur_idx += 1
        pcds[n] = pcd
    
    o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    args = parse_arg()
    main(args)



