import numpy as np
import cv2

def rotate_image(image, clockwise=True):
    """
    Rotate 90 degrees.

    clockwise=True is used to convert horizontal seam operations
    into vertical seam operations.
    """
    if clockwise:
        return np.ascontiguousarray(np.rot90(image, 1))
    return np.ascontiguousarray(np.rot90(image, 3))


def global_resize(gray, scale_x=1.0, scale_y=1.0, interpolation=cv2.INTER_LINEAR):
    """
    Globally stretch or shrink the full image.

    scale_x > 1 expands width.
    scale_x < 1 shrinks width.
    scale_y > 1 expands height.
    scale_y < 1 shrinks height.
    """
    h, w = gray.shape[:2]

    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)
    return resized


def get_otsu_threshold(gray):
    threshold_value, _ = cv2.threshold(
        gray.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(threshold_value)
