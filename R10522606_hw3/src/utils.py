import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux, uy = u[:, 0].reshape((N, 1)), u[:, 1].reshape((N, 1))
    vx, vy = v[:, 0].reshape((N, 1)), v[:, 1].reshape((N, 1))

    A1 = np.concatenate(
        (ux, uy, np.ones((N, 1)), np.zeros((N, 3)), -np.multiply(ux, vx), -np.multiply(uy, vx), -vx), axis=1)
    A2 = np.concatenate(
        (np.zeros((N, 3)), ux, uy, np.ones((N, 1)), -np.multiply(ux, vy), -np.multiply(uy, vy), -vy), axis=1)
    A = np.concatenate((A1, A2), axis=0)

    # TODO: 2.solve H with A
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xc, yc = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1))
    dst_coor = np.stack((xc, yc), axis=-1)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    dst_coor = dst_coor.reshape((-1, 2))
    dst_coor = np.concatenate((dst_coor, np.ones((dst_coor.shape[0], 1))), axis=1)

    if direction == 'b':
        # TODO: 3.apply H_inv to the dest pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_coor = np.dot(H_inv, dst_coor.T)
        src_coor = np.divide(src_coor, src_coor[-1, :])
        src_x = np.rint(src_coor[0, :].reshape((ymax - ymin, xmax - xmin))).astype(int)
        src_y = np.rint(src_coor[1, :].reshape((ymax - ymin, xmax - xmin))).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = ((0 < src_y) * (src_y < h_src)) * ((0 < src_x) * (src_x < w_src))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking
        dst[yc[mask], xc[mask]] = src[src_y[mask], src_x[mask]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_coor = np.dot(H, dst_coor.T)
        src_coor = np.divide(src_coor, src_coor[-1, :])
        dst_x = np.rint(src_coor[0, :].reshape(ymax - ymin, xmax - xmin)).astype(int)
        dst_y = np.rint(src_coor[1, :].reshape(ymax - ymin, xmax - xmin)).astype(int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of dst image)

        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indexing
        dst[np.clip(dst_y, 0, h_dst - 1), np.clip(dst_x, 0, w_dst - 1)] = src

    return dst
