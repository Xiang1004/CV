import torch
import argparse
import cv2
import os
import pathlib
import yaml

from tqdm import tqdm
from models import maskrcnn
from dataload import Dataset
from utils.visualizer import draw_prediction

if __name__ == '__main__':
    # load arguments and configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb', help="file name of configuration file")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dataset_path", type=str, help="path to dataset for evaluation")
    parser.add_argument("--weight_path", type=str, default=None, help="if it is given, evaluate this")
    parser.add_argument("--save_path", type=str, help="path to save ")
    args = parser.parse_args()
    
    with open('cfgs/' + args.cfg + '.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    # fix seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    torch.manual_seed(7777)

    # load model
    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, args.gpu)
    model.to(device)

    # load dataset
    for num in os.listdir(args.dataset_path):
        dataset_path = os.path.join(args.dataset_path, num)
        val_dataset = Dataset(dataset_path=dataset_path, mode="val", cfg=cfg)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=1, shuffle=False)

        # path to checkpoint files
        save_dir = os.path.join(args.save_path, num)
        print("Images are saved at", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        cpu_device = torch.device("cpu")

        # inference

        print("Inference with", args.weight_path)
        model.load_state_dict(torch.load(args.weight_path))
        model.eval()
        for img_idx, (image, _) in enumerate(tqdm(val_loader)):
            image = list(img.to(device) for img in image)
            outputs = model(image)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # draw result image
            vis_image = draw_prediction(image[0].to(cpu_device), outputs[0], args.thresh, num)
            # save image
            save_name = "{:01d}.png".format(img_idx)
            path = os.path.join(save_dir, save_name)
            cv2.imwrite(path, vis_image)
