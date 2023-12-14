"""Image Weather Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import numpy as np

import redos
import todos

from . import segmentator
from . import ade20k

import pdb



def get_segment_model():
    """Create model."""

    device = todos.model.get_device()
    model = segmentator.YOSO()
    model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    if 'cpu' in str(device.type):
        model.float()
   
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_panoptic.torch"):
        model.save("output/image_panoptic.torch")

    return model, device


def blender_segment(input_tensor, output_tensor):
    palette = np.array(ade20k.ADE20K.PALETTE)
    B, C, H, W = input_tensor.size()

    # input_tensor.size() -- [1, 3, 512, 512]
    color_numpy = np.zeros((H, W, 3), dtype=np.uint8)
    mask_numpy = output_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    for label, color in enumerate(palette):
        color_numpy[mask_numpy == label, :] = color
    color_tensor = torch.from_numpy(color_numpy).permute(2, 0, 1).unsqueeze(0)

    return 0.5 * input_tensor.cpu() + 0.5 * color_tensor / 255.0


def model_forward(model, device, input_tensor):
    output_tensor = todos.model.forward(model, device, input_tensor)
    final_tensor = blender_segment(input_tensor.cpu(), output_tensor.cpu())

    return final_tensor


def image_predict(input_files, output_dir):
    print(f"Segment predict {input_files} ... ")

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # Load model
    model, device = get_segment_model()

    # Load files
    image_filenames = todos.data.load_files(input_files)

    # Start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # Original input
        input_tensor = todos.data.load_tensor(filename)
        # Pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)

    todos.model.reset_device()
