import sys
import torch
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import argparse

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="path to save ")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])

    for num in os.listdir(args.save_path):
        remove_root = os.path.join(args.save_path, str(num), 'conf.txt')
        if os.path.isfile(remove_root):
            os.remove(remove_root)

    none_path = os.path.join('.', 'none.png')

    for num in os.listdir(args.save_path):
        root = os.path.join(args.save_path, str(num))
        conf_txt = os.path.join(root,'conf.txt')
        nr_image = len([name for name in os.listdir(root) if name.endswith('.png')])
        result = []

        for idx in range(nr_image):
            image_name = os.path.join(root, f'{idx}.png')
            image = Image.open(image_name).convert('RGB')
            image = transforms.ToTensor()(image)

            if float(torch.max(image[0]))+float(torch.max(image[1]))+float(torch.max(image[2])) == float(0):
                ans = int(0)
            else:
                ans = int(1)
            result.append(ans)

            image_0 = tensor_to_image(image[0])
            if os.path.isfile(image_name):
                os.remove(image_name)
            image_0 = image_0.save(image_name)
            img = cv2.imread(image_name)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            img = cv2.erode(img, kernel)
            img = cv2.dilate(img, kernel)
            if os.path.isfile(image_name):
                os.remove(image_name)
            cv2.imwrite(image_name, img)
            if ans == int(0) and not os.path.isfile(none_path):
                cv2.imwrite(none_path, img)

        output = np.ones(nr_image)
        final_out = np.zeros(nr_image)

        for i in range(2,nr_image-2):
            if int(result[i]) == int(0):
                output[i-2] = output[i-1] = output[i] = output[i+1] = output[i+2] = int(0)

        for j in range(2, nr_image-2):
            if int(output[j]) == int(1):
                final_out[j - 2] = final_out[j - 1] = final_out[j] = final_out[j + 1] = final_out[j + 2] = int(1)

        for k in range(nr_image):
            if int(final_out[k]) == int(1):
                ans = '1.0'
            else:
                ans = '0.0'
                image_name = os.path.join(root, f'{k}.png')
                img = cv2.imread(none_path)
                if os.path.isfile(image_name):
                    os.remove(image_name)
                cv2.imwrite(image_name, img)
            print(ans, end=' \n', file=open(conf_txt, 'a'))








