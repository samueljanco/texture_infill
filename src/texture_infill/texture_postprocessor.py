
import numpy as np
import cv2
from .seams_density_modifier import modify_density
from .helpers import global_resize
import math






def largest_rotated_rect_size(w, h, angle_degrees):
    """
    Size of the largest axis-aligned rectangle inside a w x h rectangle
    rotated by angle_degrees.

    Returns:
        crop_w, crop_h
    """

    if w <= 0 or h <= 0:
        return 0, 0

    angle = math.radians(angle_degrees)

    # Normalize angle to [0, pi/2]
    angle = angle % math.pi
    if angle > math.pi / 2:
        angle = math.pi - angle

    sin_a = abs(math.sin(angle))
    cos_a = abs(math.cos(angle))

    if sin_a < 1e-12 or cos_a < 1e-12:
        return int(w), int(h)

    width_is_longer = w >= h
    side_long = w if width_is_longer else h
    side_short = h if width_is_longer else w

    # Case 1: half-constrained solution
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        if width_is_longer:
            crop_w = x / sin_a
            crop_h = x / cos_a
        else:
            crop_w = x / cos_a
            crop_h = x / sin_a

    # Case 2: fully constrained solution
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a

        crop_w = (w * cos_a - h * sin_a) / cos_2a
        crop_h = (h * cos_a - w * sin_a) / cos_2a

    return int(math.floor(crop_w)), int(math.floor(crop_h))


def rotate_and_crop_no_empty(gray, angle_degrees, interpolation=cv2.INTER_LINEAR):
    """
    Rotate image around its center, expand canvas, then crop the largest
    centered rectangle that contains no empty corner areas.
    """

    #cv2.imshow("gray", gray)

    if abs(angle_degrees) < 1e-8:
        return gray

    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])

    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))

    # Recenter rotated image inside expanded canvas
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        gray,
        M,
        (new_w, new_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    #cv2.imshow("rotated", rotated)

    crop_w, crop_h = largest_rotated_rect_size(w, h, angle_degrees)

    crop_w = min(crop_w, new_w)
    crop_h = min(crop_h, new_h)

    cx = new_w // 2
    cy = new_h // 2

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)

    x2 = min(new_w, x1 + crop_w)
    y2 = min(new_h, y1 + crop_h)

    # Adjust if clipping happened on the right/bottom
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    cropped = rotated[y1:y2, x1:x2]

    #cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)
    return cropped


def rotated_cropped_size(w, h, angle_degrees):
    """
    Return the output size of rotate_and_crop_no_empty without touching pixels.

    Returns:
        out_w, out_h
    """

    w = int(w)
    h = int(h)

    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive.")

    if abs(angle_degrees) < 1e-8:
        return w, h

    angle = math.radians(angle_degrees)
    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))

    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))

    crop_w, crop_h = largest_rotated_rect_size(w, h, angle_degrees)

    crop_w = min(crop_w, new_w)
    crop_h = min(crop_h, new_h)

    return crop_w, crop_h



def postprocess_texture(textures, args):
    # todo: optimize calculations outside the loop

    processed_textures = []

    for texture in textures:
        interpolation = cv2.INTER_NEAREST #if args.nearest else cv2.INTER_LINEAR

        #cv2.imshow("texture", texture)
        #cv2.waitKey(0)

        if args.density_applicable:
            texture = modify_density(
                texture,
                scale_x=args.scale_x,
                scale_y=args.scale_y,
                density_x=args.density_x,
                density_y=args.density_y
            )
        else:
            texture = global_resize(
                texture,
                scale_x=args.scale_x,
                scale_y=args.scale_y,
                interpolation=cv2.INTER_LINEAR
            )

        if abs(args.rotation) > 1e-8:
            texture = rotate_and_crop_no_empty(
                texture,
                angle_degrees=args.rotation,
                interpolation=cv2.INTER_NEAREST
            )

        if abs(args.global_scale - 1.0) > 1e-8:
            texture = global_resize(
                texture,
                scale_x=args.global_scale,
                scale_y=args.global_scale,
                interpolation=cv2.INTER_LINEAR
            )

            texture = np.clip(texture, 0, 255)

        processed_textures.append(texture)



    return processed_textures
