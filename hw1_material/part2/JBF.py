import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w,
                                             BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros(img.shape)     # Create a zeros of the same size as img
        k = int((self.wndw_size - 1) / 2)

        # spatial kernel Gs
        denominator_s = (2 * self.sigma_s ** 2)
        x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
        Gs = np.exp(- (x ** 2 + y ** 2) / denominator_s)

        # range kernel Gr
        denominator_r = (2 * self.sigma_r ** 2)
        T = np.linspace(0, 1, 256)
        Gr = np.exp(- (T ** 2 / denominator_r))      # 0~255

        # JBF
        h, w, _ = img.shape
        for i in range(self.pad_w, self.pad_w + h):
            for j in range(self.pad_w, self.pad_w + w):
                if padded_guidance.ndim == 3:
                    G = Gr[abs(padded_guidance[(i - k):(i + k + 1), (j - k):(j + k + 1), 0] -
                                   padded_guidance[i, j, 0])] * \
                         Gr[abs(padded_guidance[(i - k):(i + k + 1), (j - k):(j + k + 1), 1] -
                                   padded_guidance[i, j, 1])] * \
                         Gr[abs(padded_guidance[(i - k):(i + k + 1), (j - k):(j + k + 1), 2] -
                                   padded_guidance[i, j, 2])] * Gs
                else:
                    G = Gr[abs(padded_guidance[(i - k):(i + k + 1), (j - k):(j + k + 1)] -
                                 padded_guidance[i, j])] * Gs

                output[i - self.pad_w, j - self.pad_w, 0] = \
                    np.sum(G * padded_img[(i - k): (i + k + 1), (j - k):(j + k + 1), 0]) / np.sum(G)
                output[i - self.pad_w, j - self.pad_w, 1] = \
                    np.sum(G * padded_img[(i - k): (i + k + 1), (j - k):(j + k + 1), 1]) / np.sum(G)
                output[i - self.pad_w, j - self.pad_w, 2] = \
                    np.sum(G * padded_img[(i - k): (i + k + 1), (j - k):(j + k + 1), 2]) / np.sum(G)

        return np.clip(output, 0, 255).astype(np.uint8)
