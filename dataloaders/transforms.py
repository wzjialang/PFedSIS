import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = np.array(sample["image"])
        mask = sample["label"]

        img1 = np.array(img1.astype(np.float32).transpose((0, 3, 1, 2))) / 255.0
        mask = np.array(mask)

        img1 = torch.from_numpy(img1).float()
        mask = torch.from_numpy(mask).long()

        return {"image": img1, "label": mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            for i in range(len(img1)):
                img1[i] = img1[i].transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img1, "label": mask}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            for i in range(len(img1)):
                img1[i] = img1[i].transpose(Image.FLIP_TOP_BOTTOM)

            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {"image": img1, "label": mask}


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {"image": (img1, img2), "label": mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample["image"]
        mask = sample["label"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        for i in range(len(img1)):
            img1[i] = img1[i].rotate(rotate_degree, Image.BILINEAR)

        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": img1, "label": mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {"image": (img1, img2), "label": mask}


def _random_scale(self, imgs, mask):
    base_size_w = self.base_size["w"]
    crop_size_w = self.crop_size["w"]
    crop_size_h = self.crop_size["h"]
    # random scale (short edge)

    w, h = imgs[0].size

    long_size = random.randint(int(base_size_w * 0.5), int(base_size_w * 2.0))
    if h > w:
        oh = long_size
        ow = int(1.0 * w * long_size / h + 0.5)
        short_size = ow
    else:  # here
        ow = long_size
        oh = int(1.0 * h * long_size / w + 0.5)
        short_size = oh
    for i in range(len(imgs)):
        imgs[i] = imgs[i].resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # print(ow,oh) #926,521

    # pad crop
    if short_size < crop_size_w:
        padh = crop_size_h - oh if oh < crop_size_h else 0
        padw = crop_size_w - ow if ow < crop_size_w else 0
        for i in range(len(imgs)):
            imgs[i] = ImageOps.expand(imgs[i], border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size, if has the previous padding above, then do nothing
    w, h = imgs[0].size
    x1 = random.randint(0, w - crop_size_w)
    y1 = random.randint(0, h - crop_size_h)
    for i in range(len(imgs)):
        imgs[i] = np.array(imgs[i].crop((x1, y1, x1 + crop_size_w, y1 + crop_size_h)))
    mask = np.array(mask.crop((x1, y1, x1 + crop_size_w, y1 + crop_size_h)))
    # final transform
    return imgs, mask


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size_w = base_size["w"]
        self.crop_size_w = crop_size["w"]
        self.crop_size_h = crop_size["h"]
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        # random scale (short edge)

        long_size = random.randint(int(self.base_size_w * 0.5), int(self.base_size_w * 2.0))
        w, h = img[0].size

        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:  # here
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        for i in range(len(img)):
            img[i] = img[i].resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # print(ow,oh) #926,521

        # pad crop
        if short_size < self.crop_size_w:
            padh = self.crop_size_h - oh if oh < self.crop_size_h else 0
            padw = self.crop_size_w - ow if ow < self.crop_size_w else 0
            for i in range(len(img)):
                img[i] = ImageOps.expand(img[i], border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size, if has the previous padding above, then do nothing
        w, h = img[0].size
        x1 = random.randint(0, w - self.crop_size_w)
        y1 = random.randint(0, h - self.crop_size_h)
        for i in range(len(img)):
            img[i] = np.array(img[i].crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h)))
        mask = np.array(mask.crop((x1, y1, x1 + self.crop_size_w, y1 + self.crop_size_h)))

        return {"image": img, "label": mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]

        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {"image": (img1, img2), "label": mask}
