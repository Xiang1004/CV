import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        gaussian_images = []
        for i in range(self.num_octaves):
            gaussian_images.append(image)
            for j in range(1, self.num_guassian_images_per_octave):
                img_GB = cv2.GaussianBlur(image, (0, 0), self.sigma**j)
                gaussian_images.append(img_GB)
                # cv2.imwrite('{}.jpg'.format(i * 10 + j), image)
            image = cv2.resize(img_GB, (int(img_GB.shape[1]/2), int(img_GB.shape[0]/2)),
                               interpolation=cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                img_DoG = cv2.subtract(gaussian_images[5*i + j], gaussian_images[5*i + j+1])
                # cv2.normalize(src=img_DoG, dst=img_DoG, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
                dog_images.append(img_DoG)
                # cv2.imwrite('{}_DoG.jpg'.format(i*10+j+100), img_DoG)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        keypoints = []
        for oct in range(self.num_octaves):
            for num in range(4*oct + 1, 4*oct + self.num_DoG_images_per_octave - 1):
                for i in range(1, dog_images[num].shape[0]-1):
                    for j in range(1, dog_images[num].shape[1]-1):
                        if abs(dog_images[num][i][j]) > self.threshold:
                            pixel = []     # 3x3 points
                            for x in range(-1, 2):
                                for y in range(-1, 2):
                                    for z in range(-1, 2):
                                        pixel.append(dog_images[num+z][i+x][j+y])
                            if np.asarray(pixel).max() <= dog_images[num][i][j] or \
                                    np.asarray(pixel).min() >= dog_images[num][i][j]:
                                keypoints.append((i*(oct+1), j*(oct+1)))

        # Step 4: Delete duplicate keypoints
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]

        return np.array(keypoints, dtype=object)  # np.array(dog_images, dtype=object)
