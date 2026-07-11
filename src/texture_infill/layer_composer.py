import numpy as np


def compose_layers(
    layers,
    white_background=False,
    background_brightness=1.0
) -> np.ndarray:
    """
    Alpha-compose multiple RGBA layers.

    layers:
        list of H x W x 4 uint8 RGBA images.

    white_background:
        False -> return RGBA with transparency.
        True  -> return RGB composited on background_color.
    """

    if len(layers) == 0:
        raise ValueError("layers must contain at least one image")

    h, w = layers[0].shape[:2]

    out_rgb = np.zeros((h, w, 3), dtype=np.float32)
    out_a = np.zeros((h, w), dtype=np.float32)

    for layer in layers:
        if layer.shape[:2] != (h, w):
            raise ValueError("All layers must have the same height and width")

        src_rgb = layer[..., :3].astype(np.float32) / 255.0
        src_a = layer[..., 3].astype(np.float32) / 255.0

        # Standard source-over alpha compositing
        out_rgb = src_rgb * src_a[..., None] + out_rgb * (1.0 - src_a[..., None])
        out_a = src_a + out_a * (1.0 - src_a)

    if white_background:
        b = np.clip(background_brightness, 0.0, 1.0)
        bg = np.array([b, b, b], dtype=np.float32)

        rgb = out_rgb + bg * (1.0 - out_a[..., None])
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = out_rgb
    rgba[..., 3] = out_a

    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)