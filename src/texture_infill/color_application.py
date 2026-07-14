
import numpy as np


def color_textures(textures, colors, strengths):

    colorized_textures = []
    for i in range(len(textures)):
        colorized_texture = colorize_gray(textures[i], colors[min(i, len(colors) - 1)], strengths[min(i, len(strengths) - 1)])
        colorized_textures.append(colorized_texture)

    return colorized_textures



def colorize_gray(gray: np.ndarray, color, texture_strength) -> np.ndarray:
    alpha = (gray.astype(np.float32) / 255.0) * texture_strength
    alpha = np.clip(alpha, 0.0, 1.0)

    color_arr = np.array(color, dtype=np.float32)

    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.float32)
    rgba[..., :3] = color_arr * alpha[..., None]
    rgba[..., 3] = alpha * 255.0

    return np.clip(rgba, 0, 255).astype(np.uint8)
