import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from .pvtv2 import pvt_v2_b2, pvt_v2_b0
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
import segmentation_models_pytorch as smp


def BuildFPN(num_classes, encoder="pvtb2", decoder="fpn"):
    if encoder == "pvtb0":
        backbone = pvt_v2_b0()
        path = "./pretrained/pvt_v2_b0.pth"
        chs = [16, 32, 80, 128]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == "pvtb2":
        backbone = pvt_v2_b2()
        path = "weights/pvt_v2_b2.pth"
        chs = [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == "resnet50":
        backbone = smp.encoders.get_encoder("resnet50")
        chs = [256, 512, 1024, 2048]
    elif encoder == "resnet18":
        backbone = smp.encoders.get_encoder("resnet18")
        chs = [64, 128, 256, 512]
    else:
        raise NotImplementedError

    trans = "resnet" not in encoder
    head = _head(num_classes, in_chs=128)
    head_p = _head(num_classes, in_chs=128)
    head_res = _head(3, in_chs=128)

    decoder = FPNDecoder(chs)
    decoder_p = FPNDecoder(chs)
    model = _SimpleSegmentationModel(trans, backbone, decoder, head, head_p, decoder_p, head_res)
    return model


class _head(nn.Module):
    def __init__(self, num_classes, in_chs):
        super(_head, self).__init__()
        self.p_head = nn.Conv2d(in_chs, num_classes, 1)

    def forward(self, feature):
        return self.p_head(feature)


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, trans, backbone, decoder, head, head_p, decoder_p, head_res):
        super(_SimpleSegmentationModel, self).__init__()
        self.trans = trans
        self.backbone = backbone
        self.head = head
        self.head_p = head_p
        self.decoder = decoder
        self.decoder_p = decoder_p
        self.head_res = head_res

    def forward(self, x):
        input_shape = x.shape[-2:]
        features, _ = self.backbone(x)

        C1, C2, C3, C4 = (
            features[-4].shape[1],
            features[-3].shape[1],
            features[-2].shape[1],
            features[-1].shape[1],
        )
        feature_list = [
            features[-4][:, : C1 // 2, ...],
            features[-3][:, : C2 // 2, ...],
            features[-2][:, : C3 // 2, ...],
            features[-1][:, : C4 // 2, ...],
        ]
        x_g_f = self.decoder(feature_list)
        feature_list_p = [
            features[-4][:, C1 // 2 :, ...],
            features[-3][:, C2 // 2 :, ...],
            features[-2][:, C3 // 2 :, ...],
            features[-1][:, C4 // 2 :, ...],
        ]
        x_p_f = self.decoder_p(feature_list_p)

        x_g = self.head(x_g_f)
        x_p = self.head_p(x_p_f)
        x_res = self.head_res(x_p_f)
        x_g = F.interpolate(x_g, size=input_shape, mode="bilinear", align_corners=False)
        x_p = F.interpolate(x_p, size=input_shape, mode="bilinear", align_corners=False)
        x_res = F.interpolate(x_res, size=input_shape, mode="bilinear", align_corners=False)

        return x_g + x_p, x_res, x_g
