import torch
import numpy as np
import medpy.metric as md

import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

color_map = {
    0: [0, 0, 0],  # background-tissue
    1: [0, 0, 255],  # instrument-shaft
    2: [0, 255, 0],  # instrument-wrist
    3: [255, 0, 0],  # instrument-clasper
}


@torch.no_grad()
def compute_scores(pred, mask, num_classes, eval_funcs, score_values, fid):
    for c in range(num_classes):
        _pred_y = (pred[0] == c).astype(np.uint8)
        _mask = (mask == c).astype(np.uint8)
        if np.max(_pred_y) == 0:
            _pred_y[0, 0] = 1
        if np.max(_mask) == 0:
            _mask[0, 0] = 1
            _pred_y[0, 0] = 1

        for e_i in range(len(eval_funcs)):
            score_values[e_i, c, fid] = eval_funcs[e_i](_pred_y, _mask)

    return score_values


@torch.no_grad()
def label2rgb(ind_im, color_map=color_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im


@torch.no_grad()
def set_alpha(image, alpha=128):
    """
    Set the alpha (transparency) of a segmentation image.

    Parameters:
    segmentation_image (Image): The input image.
    alpha (int): The alpha value to set. Default is 128.

    Returns:
    Image: The image with the alpha channel modified.
    """
    r, g, b, a = image.split()
    a = a.point(lambda i: alpha)
    image.putalpha(a)
    return image


@torch.no_grad()
def process_pred(pred, mask):
    if pred.shape[2] == mask.shape[0] and pred.shape[3] == mask.shape[1]:
        return pred.argmax(dim=1).cpu().numpy()
    else:
        pred = F.interpolate(pred, size=mask.shape)
        return pred.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def test_vis(site_index, args, dataloader, test_net1, visulization=False):
    test_net1.eval()

    eval_funcs = [md.dc, md.jc, md.assd, md.hd95]

    if args.dataset == "robotics":
        score_values1 = np.zeros((len(eval_funcs), args.num_classes, len(dataloader)))

    for fid, filename in enumerate(dataloader):
        if args.dataset == "robotics":
            image, mask = (
                filename["image"][:, 0, :, :, :].cuda(),
                filename["label"].squeeze(0).numpy(),
            )

            pred1, _, _ = test_net1(image)
            pred1 = process_pred(pred1, mask)

            if visulization:
                torch_resize = torchvision.transforms.Resize([mask.shape[0], mask.shape[1]])
                image = torch_resize(image).squeeze(0).cpu()

                mask_ = label2rgb(mask)
                pred1_ = label2rgb(pred1[0, :, :])

                image = transforms.ToPILImage()(image)
                mask_ = transforms.ToPILImage()(mask_.astype(np.uint8))
                pred1_ = transforms.ToPILImage()(pred1_.astype(np.uint8))

                image = image.convert("RGBA")
                mask_ = mask_.convert("RGBA")
                pred1_ = pred1_.convert("RGBA")

                mask_ = set_alpha(mask_, alpha=128)
                pred1_ = set_alpha(pred1_, alpha=128)

                mask_ = Image.alpha_composite(image, mask_)
                pred1_ = Image.alpha_composite(image, pred1_)

                total_width = image.size[0] + mask_.size[0] + pred1_.size[0] + 10
                max_height = max(image.size[1], mask_.size[1], pred1_.size[1])
                new_im = Image.new("RGBA", (total_width, max_height), "white")
                new_im.paste(image, (0, 0))
                new_im.paste(mask_, (image.size[0] + 5, 0))
                new_im.paste(pred1_, (image.size[0] + mask_.size[0] + 10, 0))

                if not os.path.exists(args.load_path + "/vis"):
                    os.makedirs(args.load_path + "/vis")
                new_im.save(args.load_path + "/vis/site{}_{}.png".format(site_index + 1, fid))
                new_im.close()

            score_values1 = compute_scores(pred1, mask, args.num_classes, eval_funcs, score_values1, fid)

            # print("site{}_{}".format(site_index + 1, fid))
            if visulization:
                with open(args.load_path + "/log_test.txt", "a") as f:
                    f.write("site{}_{}:".format(site_index + 1, fid) + " " + "ours:" + str(score_values1[:, 1:4, fid].mean(-1)))
                    f.write("\n")

    score_values1 = np.mean(score_values1, axis=-1)
    score_values1 = score_values1[:, 1:4].mean(1)

    if visulization:
        with open(args.load_path + "/log_test_overall.txt", "a") as f:
            f.write("[Overall]" + "ours:" + str(score_values1))
