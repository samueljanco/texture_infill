import math

def inverse_largest_rotated_rect_size(crop_w, crop_h, angle_degrees):
    """
    Inverse of largest_rotated_rect_size() for the common fully-constrained case.

    Given the desired cropped size after rotation, estimate the source rectangle
    size before rotation/crop.

    This works well for practical texture-generation use.
    """

    angle = math.radians(angle_degrees % 180.0)

    if angle > math.pi / 2.0:
        angle = math.pi - angle

    if abs(angle) < 1e-12:
        return crop_w, crop_h

    if abs(angle - math.pi / 2.0) < 1e-12:
        return crop_h, crop_w

    sin_a = abs(math.sin(angle))
    cos_a = abs(math.cos(angle))

    src_w = crop_w * cos_a + crop_h * sin_a
    src_h = crop_w * sin_a + crop_h * cos_a

    return src_w, src_h

def calculate_generator_dimensions(
    density_applicable,
    required_w,
    required_h,
    density_x=0.0,
    density_y=0.0,
    rotation_degrees=0.0,
    global_scale=1.0,
    scale_x=1.0,
    scale_y=1.0,
):
    """
    Estimate generator dimensions by directly inverting the postprocessing pipeline.

    Parameters:
        density_applicable:
            If False, density_x and density_y are ignored.

        required_w, required_h:
            desired final dimensions.

        density_x, density_y:
            seam density modulation.
            If density_applicable=True:
                post_density_w = generated_w * (1 + density_x)
                post_density_h = generated_h * (1 + density_y)

            If density_applicable=False:
                density does not change dimensions.

        rotation_degrees:
            final rotation angle before crop.

        global_scale:
            final uniform scale.

        scale_x, scale_y:
            non-uniform stretch scale applied in the x and y directions.

    Returns:
        generated_w, generated_h
    """

    # Undo final global scale
    crop_w = required_w / global_scale
    crop_h = required_h / global_scale

    # Undo directional stretch
    #crop_w = crop_w / scale_x
    #crop_h = crop_h / scale_y

    # Undo rotation + crop
    pre_rotation_w, pre_rotation_h = inverse_largest_rotated_rect_size(
        crop_w,
        crop_h,
        rotation_degrees
    )

    # Undo density seam dimension change only when applicable
    if density_applicable:
        generated_w_float = pre_rotation_w / (1.0 + density_x)
        generated_h_float = pre_rotation_h / (1.0 + density_y)
    else:
        generated_w_float = pre_rotation_w
        generated_h_float = pre_rotation_h

    generated_w = int(math.ceil(generated_w_float))
    generated_h = int(math.ceil(generated_h_float))

    generated_w = max(1, generated_w)
    generated_h = max(1, generated_h)

    return generated_w, generated_h
