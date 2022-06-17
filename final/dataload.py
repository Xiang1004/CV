import torch
import torchvision.transforms as transforms
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class Dataset(Dataset):
    def __init__(self, dataset_path, mode="val", cfg=None):
        self.mode = mode  # train mode or validation mode

        if self.mode == "train":
            self.rgb_path = os.path.join(dataset_path, self.mode, 'rgb')
            self.rgb_list = list(sorted(os.listdir(self.rgb_path)))
            self.seg_path = os.path.join(dataset_path, self.mode, "seg")
            self.seg_list = list(sorted(os.listdir(self.seg_path)))
        else:
            self.rgb_path = os.path.join(dataset_path)
            self.rgb_list = list(sorted(os.listdir(self.rgb_path), key=lambda info: (int(info[0:-4]), info[-4:])))
            #self.rgb_list = list(sorted(os.listdir(self.rgb_path)))


        print(mode, ":", len(self.rgb_list), "images")

        self.width = cfg["width"]
        self.height = cfg["height"]

        if mode == "train":
            self.rgb_transform = transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.GaussianBlur(21, 10),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, idx):

        rgb = Image.open(os.path.join(self.rgb_path, self.rgb_list[idx])).convert("RGB")
        rgb = rgb.resize((self.width, self.height))
        img = self.rgb_transform(rgb)
        if self.mode == "train":
            seg_mask_path = os.path.join(self.seg_path, self.seg_list[idx])
            seg_mask = Image.open(seg_mask_path).convert("L")
            seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
            seg_mask = np.array(seg_mask)
            # instances are encoded as different colors
            obj_ids = np.unique(seg_mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set of binary masks
            seg_masks = seg_mask == obj_ids[:, None, None]
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            temp_obj_ids = []
            temp_masks = []
            boxes = []
            for i in range(num_objs):
                pos = np.where(seg_masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if int(xmax - xmin) < 1 or int(ymax - ymin) < 1:
                    continue
                temp_masks.append(seg_masks[i])
                temp_obj_ids.append(obj_ids[i])
                boxes.append([xmin, ymin, xmax, ymax])
            obj_ids = temp_obj_ids
            seg_masks = np.asarray(temp_masks)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = []
            for obj_id in obj_ids:
                if 1 <= obj_id:
                    labels.append(1)
                else:
                    print("miss value error")
                    exit(0)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            seg_masks = torch.as_tensor(seg_masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            target = {"boxes": boxes,
                      "labels": labels,
                      "masks": seg_masks,
                      "image_id": image_id,
                      "area": area,
                      "iscrowd": iscrowd}
            return img, target
        return img, img

    def __len__(self):
        return len(self.rgb_list)
