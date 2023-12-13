import os
import torch
from torch import nn
import torch.nn.functional as F

from .resnet import build_resnet_backbone
from .neck import YOSONeck
from .head import YOSOHead

import todos
import pdb


def sem_seg_postprocess(result, img_size, output_height, output_width):
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


class YOSO(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1280
        self.MAX_W = 1280
        self.MAX_TIMES = 32
        # GPU ?
        
        self.num_classes = 150
        self.object_mask_threshold = 0.8
        self.overlap_threshold = 0.98
        self.thing_ids = [7, 8, 10, 12, 14, 15, 18, 19, 20, 22, 23, 24, 27, 30, 31, 32, 33, 35, 36, 37, 38, 39, 
                41, 42, 43, 44, 45, 47, 49, 50, 53, 55, 56, 57, 58, 62, 64, 65, 66, 67, 69, 70, 71, 72, 
                73, 74, 75, 76, 78, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 97, 98, 102, 103, 
                104, 107, 108, 110, 111, 112, 115, 116, 118, 119, 120, 121, 123, 124, 125, 126, 127, 129, 
                130, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 146, 147, 148, 149]
        
        self.backbone = build_resnet_backbone()
        # self.backbone -- ResNet(...)

        self.yoso_neck = YOSONeck()
        self.yoso_head = YOSOHead(num_stages=2)

        # PIXEL_MEAN: [123.675/255.0, 116.280/255.0, 103.530/255.0]
        # PIXEL_STD: [58.395/255.0, 57.120/255.0, 57.375/255.0]
        self.register_buffer("pixel_mean", torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), False)
        self.load_weights()


    def load_weights(self, model_path="models/image_panoptic.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)['model']
        new_sd = {}
        for k in sd.keys():
            if not 'yoso_head.criterion' in k:
                new_sd[k] = sd[k]
        self.load_state_dict(new_sd)


    def forward(self, x):
        B, C, H, W = x.size()
        # torch.save(self.state_dict, "/tmp/image_yoso.pth")

        x = (x - self.pixel_mean) / self.pixel_std

        #tensor [x] size: [1, 3, 640, 864], min: -2.03228, max: 2.64, mean: 0.033599
        # Good

        # todos.debug.output_var("x", x)
        # Good
        # tensor [x] size: [1, 3, 960, 1280], min: -2.117904, max: 2.64, mean: 0.034018

        backbone_feats = self.backbone(x)
        # backbone_feats.keys() -- ['res2', 'res3', 'res4', 'res5']

        # Good
        # tensor [images.tensor] size: [1, 3, 960, 1280], min: -2.117904, max: 2.64, mean: 0.034018
        # tensor [res2] size: [1, 256, 240, 320], min: 0.0, max: 2.523891, mean: 0.14508
        # tensor [res3] size: [1, 512, 120, 160], min: 0.0, max: 2.850623, mean: 0.076335
        # tensor [res4] size: [1, 1024, 60, 80], min: 0.0, max: 2.428513, mean: 0.039654
        # tensor [res5] size: [1, 2048, 30, 40], min: 0.0, max: 11.654506, mean: 0.033195

        # todos.debug.output_var("res2", backbone_feats['res2'])
        # todos.debug.output_var("res3", backbone_feats['res3'])
        # todos.debug.output_var("res4", backbone_feats['res4'])
        # todos.debug.output_var("res5", backbone_feats['res5'])

        # Bad
        # tensor [x] size: [1, 3, 960, 1280], min: -2.117904, max: 2.64, mean: 0.034018
        # tensor [res2] size: [1, 256, 240, 320], min: 0.0, max: 2.523891, mean: 0.14508
        # tensor [res3] size: [1, 512, 120, 160], min: 0.0, max: 2.850623, mean: 0.076335
        # tensor [res4] size: [1, 1024, 60, 80], min: 0.0, max: 2.428513, mean: 0.039654
        # tensor [res5] size: [1, 2048, 30, 40], min: 0.0, max: 11.654509, mean: 0.033195


        # print(features)
        features = list()
        for f in ['res2', 'res3', 'res4', 'res5']:
            features.append(backbone_feats[f])
        # outputs = self.sem_seg_head(features)
        neck_feats = self.yoso_neck(features)
        # tensor [neck_feats] size: [1, 256, 160, 216], min: -60.158829, max: 78.699196, mean: 0.579509

        losses, cls_scores, mask_preds = self.yoso_head(neck_feats, None)

        # todos.debug.output_var("cls_scores", cls_scores)
        # todos.debug.output_var("mask_preds", mask_preds)

        # Good
        # tensor [cls_scores] size: [1, 100, 151], min: -34.820499, max: 3.981438, mean: -17.157784
        # tensor [mask_preds] size: [1, 100, 240, 320], min: -290.145142, max: 23.088482, mean: -31.55003

        # Bad
        # tensor [cls_scores] size: [1, 100, 151], min: -34.820499, max: 3.981438, mean: -17.157784
        # tensor [mask_preds] size: [1, 100, 240, 320], min: -290.144806, max: 23.088469, mean: -31.550026

        # losses -- {}
        # cls_scores.size() -- [1, 100, 151]
        # mask_preds.size() -- [1, 100, 160, 216]
        mask_cls_results = cls_scores #outputs["pred_logits"]
        mask_pred_results = mask_preds #outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(x.shape[-2], x.shape[-1]), # x.shape -- [1, 3, 640, 864]
            mode="bilinear",
            align_corners=False,
        )

        for mask_cls_result, mask_pred_result, in zip(mask_cls_results, mask_pred_results):
            panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result)

            # Good
            # (Pdb) panoptic_r[0].size() -- [960, 1280]
            # (Pdb) panoptic_r[0]
            # tensor([[2, 2, 2,  ..., 2, 2, 2],
            #         [2, 2, 2,  ..., 2, 2, 2],
            #         [2, 2, 2,  ..., 2, 2, 2],
            #         ...,
            #         [4, 4, 4,  ..., 6, 6, 6],
            #         [4, 4, 4,  ..., 6, 6, 6],
            #         [4, 4, 4,  ..., 6, 6, 6]], device='cuda:0', dtype=torch.int32)
            # panoptic_r[1]
            # {'id': 1, 'isthing': False, 'category_id': 17}, 
            # {'id': 2, 'isthing': False, 'category_id': 2}, 
            # {'id': 3, 'isthing': False, 'category_id': 1}, 
            # {'id': 4, 'isthing': False, 'category_id': 9}, 
            # {'id': 5, 'isthing': False, 'category_id': 0}, 
            # {'id': 6, 'isthing': False, 'category_id': 6}, 
            # {'id': 7, 'isthing': False, 'category_id': 4}

            # Bad
            # {'id': 1, 'isthing': False, 'category_id': 17}, 
            # {'id': 2, 'isthing': False, 'category_id': 2}, 
            # {'id': 3, 'isthing': False, 'category_id': 1}, 
            # {'id': 4, 'isthing': False, 'category_id': 9}, 
            # {'id': 5, 'isthing': False, 'category_id': 0}, 
            # {'id': 6, 'isthing': False, 'category_id': 6}, 
            # {'id': 7, 'isthing': False, 'category_id': 4}

            return panoptic_r[0].unsqueeze(0).unsqueeze(0)


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
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()

                isthing = pred_class in self.thing_ids
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold: # 0.98
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

