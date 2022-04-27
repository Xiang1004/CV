import numpy as np
import cv2
import random
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    min_w = 0

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        keypoints1, descriptors1 = cv2.ORB_create().detectAndCompute(im1, None)
        keypoints2, descriptors2 = cv2.ORB_create().detectAndCompute(im2, None)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors1, descriptors2, k=2)
        goodu, goodv = [], []

        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                goodu.append(keypoints1[m.queryIdx].pt)
                goodv.append(keypoints2[m.trainIdx].pt)
        goodu, goodv = np.array(goodu), np.array(goodv)

        # TODO: 2. apply RANSAC to choose best H
        max_in = 0
        best_H = np.eye(3)
        for i in range(2000):
            random_u = np.zeros((4, 2))
            random_v = np.zeros((4, 2))
            for j in range(4):
                rint = random.randint(0, len(goodu) - 1)
                random_u[j] = goodu[rint]
                random_v[j] = goodv[rint]
            H = solve_homography(random_v, random_u)

            u = np.concatenate((goodu, np.ones((len(goodu), 1))), axis=1)
            v = np.concatenate((goodv, np.ones((len(goodv), 1))), axis=1)

            est_u = np.dot(H, v.T)
            est_u = np.divide(est_u, est_u[-1])

            errors = np.linalg.norm(u - est_u.T, ord=1, axis=1)
            in_liner = sum(errors < 4)
            if in_liner > max_in:
                max_in = in_liner
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        min_w += im1.shape[1]
        max_w = min_w + im2.shape[1]
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], min_w, max_w, direction='b')

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
