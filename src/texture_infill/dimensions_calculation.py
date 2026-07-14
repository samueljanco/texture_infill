import math


def required_pre_rotation_size_for_crop_at_least(
    target_w: float,
    target_h: float,
    angle_degrees: float,
) -> tuple[float, float]:
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target width and height must be positive.")

    angle = abs(math.radians(angle_degrees)) % math.pi
    if angle > math.pi / 2:
        angle = math.pi - angle

    s = abs(math.sin(angle))
    c = abs(math.cos(angle))
    eps = 1e-12

    if s < eps:
        return target_w, target_h

    if c < eps:
        return target_h, target_w

    candidates = []

    # Fully constrained case: exact inverse.
    w_full = target_w * c + target_h * s
    h_full = target_w * s + target_h * c

    side_short = min(w_full, h_full)
    side_long = max(w_full, h_full)

    if side_short > 2.0 * s * c * side_long:
        candidates.append((w_full, h_full))

    # Half-constrained wide-source case.
    # Output aspect tends toward c / s.
    h_wide = max(2.0 * s * target_w, 2.0 * c * target_h)
    w_wide = h_wide / (2.0 * s * c)
    candidates.append((w_wide, h_wide))

    # Half-constrained tall-source case.
    # Output aspect tends toward s / c.
    w_tall = max(2.0 * c * target_w, 2.0 * s * target_h)
    h_tall = w_tall / (2.0 * s * c)
    candidates.append((w_tall, h_tall))

    return min(candidates, key=lambda wh: wh[0] * wh[1])

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

    if required_w <= 0 or required_h <= 0:
        raise ValueError("Required width and height must be positive.")

    if global_scale <= 0 or scale_x <= 0 or scale_y <= 0:
        raise ValueError("Scale values must be positive.")

    # Undo final global scale.
    crop_w = required_w / global_scale
    crop_h = required_h / global_scale

    # Undo rotation + largest safe crop.
    pre_rotation_w, pre_rotation_h = required_pre_rotation_size_for_crop_at_least(
        crop_w,
        crop_h,
        rotation_degrees,
    )

    # Undo the operation that happened before rotation.
    if density_applicable:
        generated_w_float = pre_rotation_w / (1.0 + density_x)
        generated_h_float = pre_rotation_h / (1.0 + density_y)

    else:
        generated_w_float = pre_rotation_w / scale_x
        generated_h_float = pre_rotation_h / scale_y

    generated_w = max(1, int(math.ceil(generated_w_float)))
    generated_h = max(1, int(math.ceil(generated_h_float)))

    return generated_w, generated_h
