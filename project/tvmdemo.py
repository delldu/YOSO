# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
from tqdm import tqdm
import numpy as np

import torch
import todos
import image_panoptic

SO_B, SO_C, SO_H, SO_W = 1, 3, 800, 800

# RuntimeError: CUDA out of memory. Tried to allocate 600.00 MiB (GPU 0; 10.76 GiB total capacity; 7.83 GiB already allocated; 283.88 MiB free; 7.87 GiB reserved in total by PyTorch)

def blender_segment(input_tensor, output_tensor):
    palette = np.array(image_panoptic.ade20k.ADE20K.PALETTE)
    B, C, H, W = input_tensor.size()

    # input_tensor.size() -- [1, 3, 512, 512]
    color_numpy = np.zeros((H, W, 3), dtype=np.uint8)
    mask_numpy = output_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    for label, color in enumerate(palette):
        color_numpy[mask_numpy == label, :] = color
    color_tensor = torch.from_numpy(color_numpy).permute(2, 0, 1).unsqueeze(0)

    return 0.5 * input_tensor.cpu() + 0.5 * color_tensor / 255.0

def compile():
    model, device = image_panoptic.get_tvm_model()

    todos.data.mkdir("output")
    if not os.path.exists("output/image_panoptic.so"):
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        todos.tvmod.compile(model, device, input, "output/image_panoptic.so")
    todos.model.reset_device()


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    tvm_model = todos.tvmod.load("output/image_panoptic.so", "cuda")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    mean_time = 0
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        B, C, H, W = input_tensor.shape
        input_tensor = todos.data.resize_tensor(input_tensor, SO_H, SO_W)

        start_time = time.time()
        predict_tensor = todos.tvmod.forward(tvm_model, input_tensor)
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        final_tensor = blender_segment(input_tensor.cpu(), predict_tensor.cpu())

        # predict_tensor = todos.data.resize_tensor(predict_tensor, H, W)
        todos.data.save_tensor([input_tensor, final_tensor], output_file)

    mean_time = mean_time / len(image_filenames)

    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    todos.model.reset_device()


if __name__ == "__main__":
    compile()
    predict("images/*.png", "output/so")
