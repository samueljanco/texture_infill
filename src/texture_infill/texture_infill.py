import argparse
from copy import copy

import cv2
import numpy as np
from .dimensions_calculation import calculate_generator_dimensions
from .textute_generator import generate_texture
from .texture_postprocessor import postprocess_texture
from .color_application import color_textures
from .layer_composer import compose_layers
from .arguments import Arguments


def mask_bbox(mask):
    """
    Returns bounding box of a binary mask as:
    x_min, y_min, x_max, y_max

    x_max and y_max are exclusive, so:
    width  = x_max - x_min
    height = y_max - y_min
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max() + 1
    y_min = ys.min()
    y_max = ys.max() + 1

    return x_min, y_min, x_max, y_max




def generate_infill(image, mask, args):
    run_args = copy(args)
    run_args.clamp_arguments()

    if run_args.generate_square:
        target_width = run_args.width
        target_height = run_args.height
        bbox = None
    else:
        bbox = mask_bbox(mask)

        if bbox is None:
            output = np.zeros_like(image)
            return output

        x_min, y_min, x_max, y_max = bbox

        target_width = x_max - x_min
        target_height = y_max - y_min

        run_args.width = target_width
        run_args.height = target_height

    gen_width, gen_height = calculate_generator_dimensions(
        run_args.density_applicable,
        run_args.width,
        run_args.height,
        run_args.density_x,
        run_args.density_y,
        run_args.rotation,
        run_args.global_scale,
        run_args.scale_x,
        run_args.scale_y,
    )

    gen_width *= 1.15
    gen_height *= 1.15

    gen_width, gen_height = int(gen_width), int(gen_height)

    print(gen_width, gen_height)

    textures = generate_texture(
        run_args.texture_name,
        gen_width,
        gen_height,
        run_args.count,
        run_args.seed,
    )

    processed_textures = postprocess_texture(textures, run_args)

    print(processed_textures[0].shape)

    processed_textures = [
        texture[:target_height, :target_width, ...]
        for texture in processed_textures
    ]

    colored_textures = color_textures(
        processed_textures,
        run_args.colors,
        run_args.color_strengths,
    )

    result = compose_layers(
        colored_textures,
        run_args.white_background,
        run_args.background_brightness,
    )

    print(result.shape)

    if run_args.generate_square:
        return result

    x_min, y_min, x_max, y_max = bbox

    output = np.zeros_like(image)

    mask_crop = mask[y_min:y_max, x_min:x_max] > 0
    output_crop = output[y_min:y_max, x_min:x_max, :]

    output_crop[mask_crop] = result[mask_crop]
    output[y_min:y_max, x_min:x_max, :] = output_crop

    return output


if __name__ == "__main__":
    image_path = "./images/test_img.jpg"
    mask_path = "./images/test_mask.png"

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_and(image, image, mask=~mask)

    args = Arguments("texture_J")
    args.count = 10
    args.scale_x = 1.0
    args.scale_y = 1.0
    args.density_x = 0.0
    args.density_y = 0.0
    args.rotation = 0.0
    args.global_scale = 1.0
    args.colors = [(30, 60, 120), (120, 60, 120), (30, 60, 120)]
    args.color_strengths = [1.0, 1.0, 1.0, 1.0]
    args.generate_square = True
    args.background_brightness = 1.0

    output = generate_infill(image, mask, args)


    cv2.imshow("output", output)
    cv2.waitKey(0)
