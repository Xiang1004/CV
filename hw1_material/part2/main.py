import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ### TODO ###
    # Read the setting.txt about different sigma
    R, G, B = [], [], []
    with open(args.setting_path) as f:
        for sigma in f.readlines()[1:]:
            s = sigma.split(',')
            if len(s) == 3:
                R.append(float(s[0]))
                G.append(float(s[1]))
                B.append(float(s[2]))
            else:
                sigma_s = int(s[1])
                sigma_r = float(s[3])

    # Create 5 types of guidance and img_gray
    h, w = img_gray.shape
    guidance = np.zeros([h, w, len(R)+1])
    for i in range(len(R)):
        guidance[:, :, i] = (R[i] * img_rgb[:, :, 0] + G[i] * img_rgb[:, :, 1] + B[i] * img_rgb[:, :, 2])
    guidance[:, :, i+1] = img_gray

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    num_guidance = guidance.shape[-1]

    # BF ground truth
    output_gt = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    # JBF guidance
    output_guid = np.zeros([h, w, 3, num_guidance])
    for i in range(num_guidance):
        output_guid[:, :, :, i] = JBF.joint_bilateral_filter(img_rgb, guidance[:, :, i])

    # Calculate Cost
    cost = []
    for i in range(num_guidance):
        cost.append(np.sum(np.abs(output_gt.astype(np.int32) - output_guid[:, :, :, i].astype(np.int32))))
        if i == num_guidance-1:
            print('img_gray', '\n', 'Cost = ', cost[i])
        else:
            print('R = ', R[i], 'G = ', G[i], 'B = ', B[i], '\n', "Cost = ", cost[i])

    highest_cost = cost.index(max(cost))
    lowest_cost = cost.index(min(cost))

    # Plot
    plt.subplot(2, 3, 1)
    plt.title('Ground truth')
    plt.imshow(output_gt)

    plt.subplot(2, 3, 2)
    plt.title('Highest cost (guidance)' )
    plt.imshow(guidance[:, :, highest_cost].astype(np.uint8), cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Highest cost (JBF image)')
    plt.imshow(output_guid[:, :, :, highest_cost].astype(np.uint8))

    plt.subplot(2, 3, 4)
    plt.title('Lowest cost (guidance)' + str(lowest_cost+1))
    plt.imshow(guidance[:, :, lowest_cost].astype(np.uint8), cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('Lowest cost (JBF image)')
    plt.imshow(output_guid[:, :, :, lowest_cost].astype(np.uint8))

    plt.show()


if __name__ == '__main__':
    main()
