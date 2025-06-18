import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import re


class Dataset(Dataset):
    def __init__(self, client_idx=None, split="train", transform=None):
        self.image_val = []
        self.root_dir = "./robotics_site_npy"
        self.transform = transform
        self.split = split
        self.client_idx = client_idx
        self.client_name = ["site1", "site2", "site3"]
        self.datapath = self.root_dir + "/{}/{}/".format(self.client_name[client_idx], split)

        self.image_list = sorted(glob(self.root_dir + "/{}/{}/image/*".format(self.client_name[client_idx], split)))

        print("total {} slices".format(len(self.image_list)))
        self.crop_size = {"h": 512, "w": 640}
        self.test = split == "test"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        raw_file = self.image_list[idx]
        ins_pattern = re.compile(r"(?<=video)\d+")
        ins_num = ins_pattern.findall(raw_file)
        ins_num = list(map(int, ins_num))[0]
        frame_pattern = re.compile(r"(?<=frame)\d+")

        frame_num = frame_pattern.findall(raw_file)
        frame_num = list(map(int, frame_num))[0]
        image, mask = self._load_data(ins_num, frame_num, 1)

        sample = {"image": image, "label": mask}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_data(self, ins, frame, t=1, global_n=0):
        if self.client_idx == 2:
            r_im = os.path.join(self.datapath, "image/video{}frame{:09d}.npy")
            r_lb = os.path.join(self.datapath, "mask/video{}frame{:09d}.npy")
        elif self.client_idx == 3:
            r_im = os.path.join(self.datapath, "image/video{:03d}frame{:03d}.npy")
            r_lb = os.path.join(self.datapath, "mask/video{:03d}frame{:03d}.npy")
        else:
            r_im = os.path.join(self.datapath, "image/video{}frame{:03d}.npy")
            r_lb = os.path.join(self.datapath, "mask/video{}frame{:03d}.npy")

        imgs = []

        if self.client_idx == 2:
            if t > frame or (False in [os.path.exists(r_im.format(ins, i)) for i in range(frame - 60 * t + 60, frame + 1, 60)]):  # when t > frame index, use future frame
                imgs += [Image.fromarray(np.load(r_im.format(ins, i))) for i in range(frame + 60 * t - 60, frame - 1, -60)]
            else:
                imgs += [Image.fromarray(np.load(r_im.format(ins, i))) for i in range(frame - 60 * t + 60, frame + 1, 60)]
        elif self.client_idx == 3:
            if t > frame or (False in [os.path.exists(r_im.format(ins, i)) for i in range(frame - 25 * t + 25, frame + 1, 25)]):  # when t > frame index, use future frame
                imgs += [Image.fromarray(np.load(r_im.format(ins, i if i != 0 else 1))) for i in range(frame + 25 * t - 25, frame - 1, -25)]
            else:
                imgs += [Image.fromarray(np.load(r_im.format(ins, i if i != 0 else 1))) for i in range(frame - 25 * t + 25, frame + 1, 25)]
        else:
            if t > frame or (False in [os.path.exists(r_im.format(ins, i)) for i in range(frame - t + 1, frame + 1)]):  # when t > frame index, use future frame
                imgs += [Image.fromarray(np.load(r_im.format(ins, i))) for i in range(frame + t - 1, frame - 1, -1)]
            else:
                imgs += [Image.fromarray(np.load(r_im.format(ins, i))) for i in range(frame - t + 1, frame + 1)]

        for i in range(len(imgs)):
            imgs[i] = imgs[i].resize((self.crop_size["w"], self.crop_size["h"]), Image.BILINEAR)

        if self.test:
            imgs = [np.array(i) for i in imgs]
            masks = Image.fromarray(np.load(r_lb.format(ins, frame)))  # 1024,1280,4
        else:
            masks = Image.fromarray(np.load(r_lb.format(ins, frame)))
            masks = masks.resize((self.crop_size["w"], self.crop_size["h"]), Image.NEAREST)

        return imgs, masks
