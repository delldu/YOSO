import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from .resnet import build_resnet50
from .neck import YOSONeck
from .head import YOSOHead

from typing import Dict

import todos
import pdb


class YOSO(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 32
        # GPU -- 3G, 60ms

        self.num_classes = 150
        self.object_mask_threshold = 0.8
        self.overlap_threshold = 0.98
        self.thing_ids = [
            7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33, 35, 36, 37, 38, 39,
            41, 42, 43, 44, 45, 47, 49, 50, 53, 55, 56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72,
            73, 74, 75, 76, 78, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103,
            104, 107, 108, 110, 111, 112, 115, 116, 118, 119, 120, 121, 123, 124, 125, 126, 127, 129,
            130, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 146, 147, 148, 149,
        ]

        self.backbone = build_resnet50()

        self.yoso_neck = YOSONeck()
        self.yoso_head = YOSOHead(num_stages=2)

        self.normal = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)

        self.load_weights()

    def load_weights(self, model_path="models/image_panoptic.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)["model"]
        new_sd = {}
        for k in sd.keys():
            if not "yoso_head.criterion" in k:
                new_sd[k] = sd[k]
        self.load_state_dict(new_sd)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.normal(x)

        backbone_feats = self.backbone(x)

        features = list()
        for f in ["res2", "res3", "res4", "res5"]:
            features.append(backbone_feats[f])

        neck_feats = self.yoso_neck(features)
        cls_scores, mask_preds = self.yoso_head(neck_feats)

        # upsample masks
        mask_preds = F.interpolate(
            mask_preds,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        panoptic_r = self.panoptic_inference(cls_scores, mask_preds)

        return panoptic_r.unsqueeze(0).unsqueeze(0)

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg

        # Now we have some objects !!!
        current_segment_id = 0
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list: Dict[int, int] = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()

            isthing = pred_class in self.thing_ids
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:  # 0.98
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = int(pred_class)  # current_segment_id

        return panoptic_seg
