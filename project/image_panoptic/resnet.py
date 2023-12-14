# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

import pdb

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        return F.batch_norm(x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=self.eps,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.norm(x)


class CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=FrozenBatchNorm2d(out_channels),
            )
        else: # support torch.jit.script
            self.shortcut = nn.Identity() # None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=FrozenBatchNorm2d(bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=FrozenBatchNorm2d(bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=FrozenBatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)

        shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__(in_channels, out_channels, 4)
        # in_channels = 3
        # out_channels = 64
        # norm = 'FrozenBN'

        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=FrozenBatchNorm2d(out_channels),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(nn.Module):
    """
    Implement :paper:`ResNet`.
    """
    def __init__(self, stem, stages, out_features):
        super().__init__()
        self.stem = stem
        
        self.stage_names = []
        self.stages = []
        for i, blocks in enumerate(stages):
            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

        # self.out_features = out_features # ['res2', 'res3', 'res4', 'res5']


    def forward(self, x) -> Dict[str, torch.Tensor]:
        outputs = {}
        x = self.stem(x)

        # torch.jit.script
        # for name, stage in zip(self.stage_names, self.stages):
        #     x = stage(x)
        #     if name in self.out_features:
        #         outputs[name] = x
        x = self.res2(x)
        outputs['res2'] = x
        x = self.res3(x)
        outputs['res3'] = x
        x = self.res4(x)
        outputs['res4'] = x
        x = self.res5(x)
        outputs['res5'] = x

        return outputs


    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs):
        # if first_stride is not None:
        #     pdb.set_trace()
        #     assert "stride" not in kwargs and "stride_per_block" not in kwargs
        #     kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
        #     logger = logging.getLogger(__name__)
        #     logger.warning(
        #         "ResNet.make_stage(first_stride=) is deprecated!  "
        #         "Use 'stride_per_block' or 'stride' instead."
        #     )

        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks



def make_stage(*args, **kwargs):
    return ResNet.make_stage(*args, **kwargs)


def build_resnet50():
    """
    Build resnet50
    """
    norm = 'FrozenBN'
    stem = BasicStem(in_channels=3, out_channels=64)

    # fmt: off
    freeze_at           = 0
    out_features        = ['res2', 'res3', 'res4', 'res5']
    depth               = 50
    num_groups          = 1
    width_per_group     = 64
    bottleneck_channels = num_groups * width_per_group # 64 --> 1024
    in_channels         = 64
    out_channels        = 256
    stride_in_1x1       = False
    res5_dilation       = 1

    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3], # True
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
    ] # [2, 3, 4, 5]

    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx], # [3, 4, 6, 3]
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels, # 2048
            "out_channels": out_channels, # 4096
            "norm": norm, # 'FrozenBN'
        }

        stage_kargs["bottleneck_channels"] = bottleneck_channels # 1024
        stage_kargs["stride_in_1x1"] = stride_in_1x1 # False
        stage_kargs["dilation"] = dilation # 1
        stage_kargs["num_groups"] = num_groups # 1
        stage_kargs["block_class"] = BottleneckBlock

        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)


    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)
