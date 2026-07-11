import argparse
import os
import random
import time

import numpy as np
import torch

from . import transformer
from . import utils
from importlib.resources import files

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grayscale textures with a trained fast texture-synthesis network."
    )
    parser.add_argument("--model", default="./models/texture_J/texture_transformer_weight.pth",
                        help="Path to a texture_transformer_weight.pth checkpoint.")
    parser.add_argument("--output", default="images/out/generated_texture2.png",
                        help="Output image path. If --count > 1, an index is added before the extension.")
    parser.add_argument("--width", type=int, default=2484)
    parser.add_argument("--height", type=int, default=2733)
    parser.add_argument("--count", type=int, default=1,
                        help="Number of independent texture samples to generate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for reproducible texture samples.")
    return parser.parse_args()




def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def output_path(base_path, index, count):
    if count == 1:
        return base_path
    stem, ext = os.path.splitext(base_path)
    return "{}_{:03d}{}".format(stem, index + 1, ext or ".png")

def to_opencv_gray_normalized(img):
    img = np.asarray(img)
    img = np.squeeze(img).astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val <= min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - min_val) * (255.0 / (max_val - min_val))
    return img.astype(np.uint8)

def generate_texture(texture_name, width, height, count, seed=None):
    set_seed(seed)

    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

    net = transformer.TransformerNetwork(input_channels=1, output_channels=1)
    model_path = (
        files("texture_infill")
        / "models"
        / f"{texture_name}_transformer_weight.pth"
    )
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device).eval()

    #os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    start_time = time.time()

    textures = []
    with torch.no_grad():
        for i in range(count):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            noise = torch.rand(1, 1, height, width, device=device).mul(255.0)
            generated_tensor = net(noise).detach().cpu()
            generated_image = utils.ttoi(generated_tensor)
            generated_image = to_opencv_gray_normalized(generated_image)
            textures.append(generated_image)
            print("Generation time: {:.2f} seconds".format(time.time() - start_time))

    return textures






