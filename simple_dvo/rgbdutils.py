import cv2 as cv
import numpy as np


# TODO width and height may be not needed as parameters
def bilinear_interpolation(img: np.ndarray, x: float, y: float, width: int, height: int):
    valid = np.nan

    x0 = np.floor(x).astype(np.uint16)
    y0 = np.floor(y).astype(np.uint16)
    x1 = x0 + 1
    y1 = y0 + 1

    w_x0 = x1 - x
    w_x1 = 1 - w_x0
    w_y0 = y1 - y
    w_y1 = 1 - w_y0

    if x0 < 0 or x0 >= width:
        w_x0 = 0
    if x1 < 0 or x1 >= width:
        w_x1 = 0
    if y0 < 0 or y0 >= height:
        w_y0 = 0
    if y1 < 0 or y1 >= height:
        w_y1 = 0

    w00 = w_x0 * w_y0
    w10 = w_x1 * w_y0
    w01 = w_x0 * w_y1
    w11 = w_x1 * w_y1

    w_all = w00 + w01 + w10 + w11
    total = 0
    if w00 > 0:
        total += img[y0, x0] * w00
    if w01 > 0:
        total += img[y1, x0] * w01
    if w10 > 0:
        total += img[y0, x1] * w10
    if w11 > 0:
        total += img[y1, x1] * w11
    
    if w_all > 0:
        valid = total / w_all
    
    return valid


def downsample_image(img: np.ndarray):

    img_ds = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) / 4.

    return img_ds


def downsample_depth(depth: np.ndarray):
    depth_ds = np.stack([depth[0::2, 0::2], depth[0::2, 1::2], depth[1::2, 0::2], depth[1::2, 1::2]], axis=2)
    num_valid_depth = np.count_nonzero(depth_ds, axis=2)
    num_valid_depth[np.where(num_valid_depth == 0)] = 1 # To avoid divid by 0

    depth_ds = np.sum(depth_ds, axis=2) / num_valid_depth

    return depth_ds



def construct_pyramid(curr_img, prev_img, curr_depth, prev_depth, K, num_levels):
    curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr = [], [], [], [], []
    curr_img_pyr.append(curr_img)
    prev_img_pyr.append(prev_img)
    curr_depth_pyr.append(curr_depth)
    prev_depth_pyr.append(prev_depth)
    K_pyr.append(K)

    for _ in range(num_levels - 1):
        K_ds = np.copy(K_pyr[-1])
        K_ds[:2, :3] = K_ds[:2, :3] / 2
        K_pyr.append(K_ds)

        curr_img_ds = downsample_image(curr_img_pyr[-1])
        curr_img_pyr.append(curr_img_ds)
        curr_depth_ds = downsample_depth(curr_depth_pyr[-1])
        curr_depth_pyr.append(curr_depth_ds)

        prev_img_ds = downsample_image(prev_img_pyr[-1])
        prev_img_pyr.append(prev_img_ds)
        prev_depth_ds = downsample_depth(prev_depth_pyr[-1])
        prev_depth_pyr.append(prev_depth_ds)
    
    return curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr


def test():
    prev_image = cv.imread('../data/cofusion/Color0001.png').astype(np.float32)
    prev_image = cv.cvtColor(prev_image, cv.COLOR_BGR2GRAY)
    curr_image = cv.imread('../data/cofusion/Color0002.png').astype(np.float32)
    curr_image = cv.cvtColor(curr_image, cv.COLOR_BGR2GRAY)
    prev_depth = cv.imread('../data/cofusion/Depth0001.png', cv.IMREAD_UNCHANGED).astype(np.float32)
    curr_depth = cv.imread('../data/cofusion/Depth0002.png', cv.IMREAD_UNCHANGED).astype(np.float32)

    patch = prev_image[:2, :2]

    K = np.asarray(
        [[360,  0, 320],
         [0,  360, 240],
         [0,    0,   1]]
        )

    curr_img_pyr, prev_img_pyr, curr_depth_pyr, prev_depth_pyr, K_pyr = construct_pyramid(curr_image, prev_image, curr_depth, prev_depth, K, 2)

    cv.imshow("downsample rgb", curr_img_pyr[-1].astype(np.uint8))
    cv.waitKey(20000)

    cv.imshow('downsample depht', curr_depth_pyr[-1].astype(np.uint8))
    cv.waitKey(20000)
    cv.destroyAllWindows()
    print(curr_img_pyr[-1])
    print(K_pyr[-1])
    print(patch, '\n', downsample_image(patch))




if __name__ == '__main__':
    a = np.arange(9).reshape(3, 3)
    h, w = a.shape
    print(a)
    u = 1.5
    v = 1.5
    print(bilinear_interpolation(a, u, v, w, h))

    b = np.arange(16).reshape(4, 4)
    b_ds = downsample_image(b)
    print('raw matrix: \n', b, "\ndownsampled matrix: \n", b_ds)

    test()

    