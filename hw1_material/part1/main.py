import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def plot_dog(dog_images, save_path):
    for i in range(len(dog_images)):
        path = (save_path + '{}.png'.format(i+1))
        cv2.normalize(src=dog_images[i], dst=dog_images[i], beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(path, dog_images[i])

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=7.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--save_path', default='./output/', help='path to output image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###

    DoG = Difference_of_Gaussian(args.threshold)

    keypoints = DoG.get_keypoints(img)
    #keypoints, dog_images = DoG.get_keypoints(img)

    save_path = (args.save_path+'threshold{}.png'.format(int(args.threshold)))
    plot_keypoints(img, keypoints, save_path)
    # plot_dog(dog_images, args.save_path)


if __name__ == '__main__':
    main()
