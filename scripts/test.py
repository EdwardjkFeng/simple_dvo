import cv2 as cv
import open3d as o3d
import sys

import rgbd_utils as utils
import image_pyramid
import matplotlib.pyplot as plt

if __name__ ==  '__main__':
            
    rgb1 = cv.imread('../data/rgb.png')
    rgb2 = cv.imread('../data/rgb2.png')

    depth1 = cv.imread('../data/depth.png', 0)
    depth2 = cv.imread('../data/depth2.png', 0)
    n_depth1 = utils.img2float(depth1)

    show = False
    if show:
        cv.imshow("rgb1", rgb1)
        cv.imshow("rgb2", rgb2)
        cv.imshow("depth1", depth1)
        cv.imshow("depth2", depth2)
        cv.imshow("normalized", image_pyramid.downsampleGray(rgb1))
        cv.waitKey(5000) # wait 5 sec before closing the windows


    """ Check bilinear interpolation implementation
    x = 30.5
    y = 400.7
    width, height = depth1.shape
    # print(rgb1[int(x), int(y)])
    print("Image shape : ", width, height)
    print("Bi-linear interpolation test\n")
    my_imp = utils.bilinear_interpolation(depth1, x, y, width, height)
    test = utils.bilinear_interpolation_test(depth1, x, y, width, height)
    print("My implementation : ", my_imp, " == Counter test :", test, " ? : ", my_imp == test) 
    """
    rgb = o3d.io.read_image("../data/rgb2.png")
    depth = o3d.io.read_image("../data/depth2.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image("../data/rgb.png"), o3d.io.read_image("../data/depth.png"))
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()