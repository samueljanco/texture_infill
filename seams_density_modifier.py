import argparse
import cv2
import numpy as np
from numba import njit
import math
import matplotlib.pyplot as plt
from helpers import rotate_image, global_resize, get_otsu_threshold



@njit(cache=True)
def directional_gap_energy_numba(gray, threshold):
    """
    Creates the directional distance energy from a grayscale image.

    First binarizes:
        white/object pixels = 1
        black/gap pixels    = 0

    Raw directional mask:
        white stroke pixels = 0
        black gaps          = higher values

    Since seam carving removes LOW-energy seams, invert_energy=True is
    usually preferred:
        white strokes = high energy
        black gaps    = lower energy
    """

    h, w = gray.shape

    binary = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            binary[y, x] = 1 if gray[y, x] > threshold else 0


    #binary = 1.0 - binary

    #cv2.imshow("binary", binary)
    #cv2.waitKey(0)

    left = np.zeros((h, w), dtype=np.float64)
    right = np.zeros((h, w), dtype=np.float64)
    up = np.zeros((h, w), dtype=np.float64)
    down = np.zeros((h, w), dtype=np.float64)

    # left -> right
    for y in range(h):
        d = 0.0
        for x in range(w):
            if binary[y, x] == 1:
                d = 0.0
            else:
                d += 1.0
            left[y, x] = d

    # right -> left
    for y in range(h):
        d = 0.0
        for x in range(w - 1, -1, -1):
            if binary[y, x] == 1:
                d = 0.0
            else:
                d += 1.0
            right[y, x] = d


    # top -> bottom
    for x in range(w):
        d = 0.0
        for y in range(h):
            if binary[y, x] == 1:
                d = 0.0
            else:
                d += 1.0
            up[y, x] = d

    # bottom -> top
    for x in range(w):
        d = 0.0
        for y in range(h - 1, -1, -1):
            if binary[y, x] == 1:
                d = 0.0
            else:
                d += 1.0
            down[y, x] = d


    raw_average = np.zeros((h, w), dtype=np.float64)

    max_val = 0.0

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 1:
                raw_average[y, x] = 0.0
            else:
                raw_average[y, x] = (
                    left[y, x] + right[y, x] + up[y, x] + down[y, x]
                )

            if raw_average[y, x] > max_val:
                max_val = raw_average[y, x]

    #energy = np.zeros((h, w), dtype=np.float64)


    #for y in range(h):
    #    for x in range(w):
    #        energy[y, x] = raw_average[y, x]

    return raw_average, left, right, up, down, raw_average, binary



@njit(cache=True)
def add_black_seam_grayscale_fast(im, seam_idx):
    """
    Insert one vertical seam.

    Inserted pixels are black, value 0.
    """

    h, w = im.shape
    output = np.zeros((h, w + 1), dtype=im.dtype)

    for y in range(h):
        seam_x = seam_idx[y]
        out_x = 0

        for x in range(w):
            output[y, out_x] = im[y, x]
            out_x += 1

            if x == seam_x:
                output[y, out_x] = 0.0
                out_x += 1

    return output


@njit(cache=True)
def apply_mask_to_energy_fast(energy, mask):
    h, w = energy.shape

    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                energy[y, x] = -800

    return energy


@njit(cache=True)
def remove_seam_grayscale_fast(im, boolmask):
    h, w = im.shape
    output = np.zeros((h, w - 1), dtype=im.dtype)

    for y in range(h):
        out_x = 0
        for x in range(w):
            if boolmask[y, x]:
                output[y, out_x] = im[y, x]
                out_x += 1

    return output


@njit(cache=True)
def remove_mask_grayscale_fast(mask, boolmask):
    h, w = mask.shape
    output = np.zeros((h, w - 1), dtype=mask.dtype)

    for y in range(h):
        out_x = 0
        for x in range(w):
            if boolmask[y, x]:
                output[y, out_x] = mask[y, x]
                out_x += 1

    return output



def seams_removal_grayscale(
    im,
    num_remove
):
    """
    Remove num_remove vertical seams.
    """

    energy, left, right, u, d, _, binary = directional_gap_energy_numba(
        im,
        20
    )

    im = apply_binary_mask_to_gray(im , binary)

    mask_for_numba = np.ascontiguousarray(binary.astype(np.float64))


    #if has_mask:
    energy = apply_mask_to_energy_fast(energy, mask_for_numba)

    #plt.imshow(energy, cmap="inferno")
    #plt.colorbar()
    #plt.show()

    for i in range(num_remove):
        seam_idx, boolmask = get_maximum_unrestricted_seam_from_energy(energy, mask_for_numba, jump_penalty=1000)

        #seam_idx, boolmask = get_maximum_seam_from_energy(energy)

        im = remove_seam_grayscale_fast(im, boolmask)
        energy = remove_seam_grayscale_fast(energy, boolmask)

        #if has_mask:
        mask_for_numba = remove_mask_grayscale_fast(mask_for_numba, boolmask)

        if (i + 1) % 10 == 0 or i == 0 or i == num_remove - 1:
            print(f"removed seam {i + 1}/{num_remove}, shape={im.shape}")


    return im


@njit(cache=True)
def get_maximum_seam_from_energy(energy_source):
    """
    Finds one maximum-energy vertical seam from a precomputed energy matrix.

    The input energy matrix is not modified; dynamic-programming costs are
    accumulated into a temporary copy.
    """

    h, w = energy_source.shape

    energy = energy_source.copy()
    backtrack = np.zeros((h, w), dtype=np.int64)

    for y in range(1, h):
        for x in range(w):
            best_x = x
            best_val = energy[y - 1, x]

            if x > 0:
                val = energy[y - 1, x - 1]
                if val > best_val:
                    best_val = val
                    best_x = x - 1

            if x < w - 1:
                val = energy[y - 1, x + 1]
                if val > best_val:
                    best_val = val
                    best_x = x + 1

            energy[y, x] += best_val
            backtrack[y, x] = best_x

    seam_idx = np.zeros(h, dtype=np.int64)

    best_end_x = 0
    best_end_val = energy[h - 1, 0]

    for x in range(1, w):
        if energy[h - 1, x] > best_end_val:
            best_end_val = energy[h - 1, x]
            best_end_x = x

    seam_idx[h - 1] = best_end_x

    for y in range(h - 2, -1, -1):
        seam_idx[y] = backtrack[y + 1, seam_idx[y + 1]]

    boolmask = np.ones((h, w), dtype=np.bool_)

    for y in range(h):
        boolmask[y, seam_idx[y]] = False

    return seam_idx, boolmask


@njit(cache=True)
def get_maximum_unrestricted_seam_from_energy(
    energy_source,
    forbidden_mask,
    jump_penalty=10.0,
    return_none=False,
):
    h, w = energy_source.shape

    prev_cost = np.empty(w, dtype=np.float64)
    curr_cost = np.empty(w, dtype=np.float64)

    left_cost = np.empty(w, dtype=np.float64)
    left_arg = np.empty(w, dtype=np.int64)

    backtrack = np.empty((h, w), dtype=np.int64)

    neg_inf = -np.inf

    for y in range(h):
        for x in range(w):
            backtrack[y, x] = -1

    first_has_valid = False

    for x in range(w):
        if forbidden_mask[0, x] or not np.isfinite(energy_source[0, x]):
            prev_cost[x] = neg_inf
        else:
            prev_cost[x] = float(energy_source[0, x])
            first_has_valid = True

    if not first_has_valid:
        if return_none:
            return None, None

        for x in range(w):
            prev_cost[x] = neg_inf

        start_x = w // 2
        prev_cost[start_x] = 0.0
        backtrack[0, start_x] = start_x

    for y in range(1, h):
        row_has_valid = False

        for x in range(w):
            if not forbidden_mask[y, x] and np.isfinite(energy_source[y, x]):
                row_has_valid = True
                break

        if not row_has_valid:
            if return_none:
                return None, None

            for x in range(w):
                curr_cost[x] = prev_cost[x]
                backtrack[y, x] = x

        else:
            # Left-to-right pass:
            # max prev_cost[p] - jump_penalty * (x - p), p <= x
            best_cost = prev_cost[0]
            best_x = 0

            left_cost[0] = best_cost
            left_arg[0] = best_x

            for x in range(1, w):
                continued = best_cost - jump_penalty

                if prev_cost[x] > continued:
                    best_cost = prev_cost[x]
                    best_x = x
                else:
                    best_cost = continued

                left_cost[x] = best_cost
                left_arg[x] = best_x

            # Right-to-left pass:
            # max prev_cost[p] - jump_penalty * (p - x), p >= x
            best_right_cost = prev_cost[w - 1]
            best_right_x = w - 1

            row_reachable = False

            for x in range(w - 1, -1, -1):
                if x < w - 1:
                    continued = best_right_cost - jump_penalty

                    if prev_cost[x] > continued:
                        best_right_cost = prev_cost[x]
                        best_right_x = x
                    else:
                        best_right_cost = continued

                if forbidden_mask[y, x] or not np.isfinite(energy_source[y, x]):
                    curr_cost[x] = neg_inf
                    backtrack[y, x] = -1
                    continue

                if left_cost[x] >= best_right_cost:
                    best_prev_cost = left_cost[x]
                    best_prev_x = left_arg[x]
                else:
                    best_prev_cost = best_right_cost
                    best_prev_x = best_right_x

                if np.isneginf(best_prev_cost):
                    curr_cost[x] = neg_inf
                    backtrack[y, x] = -1
                else:
                    curr_cost[x] = float(energy_source[y, x]) + best_prev_cost
                    backtrack[y, x] = best_prev_x
                    row_reachable = True

            if not row_reachable:
                if return_none:
                    return None, None

                for x in range(w):
                    curr_cost[x] = prev_cost[x]
                    backtrack[y, x] = x

        tmp = prev_cost
        prev_cost = curr_cost
        curr_cost = tmp

    seam_idx = np.empty(h, dtype=np.int64)

    best_end_x = 0
    best_end_cost = prev_cost[0]

    for x in range(1, w):
        if prev_cost[x] > best_end_cost:
            best_end_cost = prev_cost[x]
            best_end_x = x

    if np.isneginf(best_end_cost):
        if return_none:
            return None, None

        best_end_x = w // 2

    seam_idx[h - 1] = best_end_x

    for y in range(h - 2, -1, -1):
        prev_x = backtrack[y + 1, seam_idx[y + 1]]

        if prev_x < 0:
            prev_x = seam_idx[y + 1]

        seam_idx[y] = prev_x

    boolmask = np.ones((h, w), dtype=np.bool_)

    for y in range(h):
        boolmask[y, seam_idx[y]] = False

    return seam_idx, boolmask


def _expand_recorded_seams_evenly(seams_record, target_count):
    found_count = len(seams_record)

    if found_count == 0:
        return []

    expanded = []

    for i in range(found_count):
        start = (i * target_count) // found_count
        end = ((i + 1) * target_count) // found_count
        repeat_count = end - start

        for _ in range(repeat_count):
            expanded.append(seams_record[i].copy())

    return expanded


def apply_binary_mask_to_gray(image, binary_mask):
    image = np.asarray(image)
    binary_mask = np.asarray(binary_mask)

    # Make sure mask is boolean.
    # If your mask is 0/255 or 0/1, this works for both.
    mask_bool = binary_mask > 0

    output = np.zeros_like(image)
    output[mask_bool] = image[mask_bool]

    return output

def seams_insertion_grayscale(im, num_add):
    """
    Insert num_add vertical seams.

    Recorded seams are mapped back to original image coordinates before final
    insertion. This prevents seams found on the progressively shrunken image
    from being inserted at wrong columns.

    All inserted seam pixels are black.
    """

    seams_record = []

    temp_im = im.copy()
    h, w = temp_im.shape

    temp_energy, _, _, _, _, _, binary = directional_gap_energy_numba(
        temp_im,
        20
    )

    im = apply_binary_mask_to_gray(im , binary)
    temp_im = im.copy()




    # IMPORTANT:
    # If binary == 1 means background/gap and binary == 0 means white object,
    # then forbidden_mask must be True on white/object pixels.
    forbidden_mask = np.ascontiguousarray((binary == 1))

    # Optional, but useful if apply_mask_to_energy_fast expects large mask values.
    energy_mask = np.ascontiguousarray(forbidden_mask.astype(np.float64) * 255.0)
    temp_energy = apply_mask_to_energy_fast(temp_energy, energy_mask)

    temp_energy = np.ascontiguousarray(temp_energy, dtype=np.float64)

    #plt.imshow(temp_energy, cmap="inferno")
    #plt.colorbar()
    #lt.show()

    # Per-row map from current temporary-image x coordinate to original-image x.
    column_map = np.empty((h, w), dtype=np.int64)

    for y in range(h):
        for x in range(w):
            column_map[y, x] = x

    for i in range(num_add):
        seam_idx, boolmask = get_maximum_unrestricted_seam_from_energy(
            temp_energy,
            forbidden_mask,
            jump_penalty=10000.0,
            return_none=True
        )

        if seam_idx is None:
            print(f"no more valid insertion seams after {len(seams_record)}/{num_add}")
            break

        # Convert seam from current temporary coordinates to original coordinates.
        original_seam = np.empty_like(seam_idx)

        for y in range(seam_idx.shape[0]):
            original_seam[y] = column_map[y, seam_idx[y]]

        seams_record.append(original_seam.copy())

        temp_im = remove_seam_grayscale_fast(temp_im, boolmask)
        temp_energy = remove_seam_grayscale_fast(temp_energy, boolmask)
        forbidden_mask = remove_mask_grayscale_fast(forbidden_mask, boolmask)
        column_map = remove_seam_grayscale_fast(column_map, boolmask)

        if (i + 1) % 10 == 0 or i == 0 or i == num_add - 1:
            print(f"recorded insertion seam {i + 1}/{num_add}")

    if len(seams_record) == 0:
        print("no valid insertion seams found; returning image unchanged")
        return im

    seams_to_insert = _expand_recorded_seams_evenly(seams_record, num_add)

    if len(seams_record) < num_add:
        print(
            f"found {len(seams_record)} valid seams; "
            f"duplicating them to insert {num_add} seams"
        )

    # Insert in original-coordinate order. After each insertion, update all
    # remaining seam coordinates to account for the new column.
    for i in range(num_add):
        seam = seams_to_insert.pop(0)

        im = add_black_seam_grayscale_fast(im, seam)

        for remaining_seam in seams_to_insert:
            for y in range(remaining_seam.shape[0]):
                if remaining_seam[y] >= seam[y]:
                    remaining_seam[y] += 1

        if (i + 1) % 10 == 0 or i == 0 or i == num_add - 1:
            print(f"inserted black seam {i + 1}/{num_add}, shape={im.shape}")

    return im


def seams_insertion_grayscale1(
    im,
    num_add
):
    """
    Insert num_add vertical seams.

    If not enough valid seams can be found, already found seams are duplicated
    and distributed evenly until num_add seams are inserted.

    All inserted seam pixels are black.
    """

    seams_record = []

    temp_im = im.copy()

    temp_energy, _, _, _, _, _, binary = directional_gap_energy_numba(
        temp_im,
        30
    )

    mask_for_numba = np.ascontiguousarray(binary.astype(np.float64))

    temp_energy = apply_mask_to_energy_fast(temp_energy, mask_for_numba)

    #plt.imshow(temp_energy, cmap="inferno")
    #plt.colorbar()
    #plt.show()

    for i in range(num_add):
        seam_idx, boolmask = get_maximum_unrestricted_seam_from_energy(temp_energy, mask_for_numba, jump_penalty=100, return_none=True)

        if seam_idx is None:
            print(f"no more valid insertion seams after {len(seams_record)}/{num_add}")
            break

        seams_record.append(seam_idx.copy())

        temp_im = remove_seam_grayscale_fast(temp_im, boolmask)
        temp_energy = remove_seam_grayscale_fast(temp_energy, boolmask)
        mask_for_numba = remove_mask_grayscale_fast(mask_for_numba, boolmask)

        if (i + 1) % 10 == 0 or i == 0 or i == num_add - 1:
            print(f"recorded insertion seam {i + 1}/{num_add}")

    if len(seams_record) == 0:
        print("no valid insertion seams found; returning image unchanged")

        return im

    seams_to_insert = _expand_recorded_seams_evenly(seams_record, num_add)

    if len(seams_record) < num_add:
        print(
            f"found {len(seams_record)} valid seams; "
            f"duplicating them to insert {num_add} seams"
        )

    seams_to_insert.reverse()

    ## TODO: fix insertion of the seams

    for i in range(num_add):
        seam = seams_to_insert.pop()

        im = add_black_seam_grayscale_fast(im, seam)

        for remaining_seam in seams_to_insert:
            for y in range(remaining_seam.shape[0]):
                if remaining_seam[y] >= seam[y]:
                    remaining_seam[y] += 1

        if (i + 1) % 10 == 0 or i == 0 or i == num_add - 1:
            print(f"inserted black seam {i + 1}/{num_add}, shape={im.shape}")

    return im

def seam_carve_grayscale_by_delta(
    im,
    dx,
    dy
):
    """
    Apply seam carving/insertion by explicit seam-count deltas.

    dx < 0:
        remove vertical seams, image gets narrower.

    dx > 0:
        insert black vertical seams, image gets wider.

    dy < 0:
        remove horizontal seams, image gets shorter.

    dy > 0:
        insert black horizontal seams, image gets taller.
    """


    output = im #np.ascontiguousarray(im.astype(np.float64))

    print(f"Starting seam operation from shape: {output.shape}")
    print(f"Total seam delta: dx={dx}, dy={dy}")

    # Width operation
    if dx != 0:
        if dx < 0:
            output = seams_removal_grayscale(
                output,
                -dx,
            )



        elif dx > 0:
            output = seams_insertion_grayscale(
                output,
                dx
            )


    # Height operation via rotation
    if dy != 0:
        output = rotate_image(output, clockwise=True)
        if dy < 0:
            output = seams_removal_grayscale(
                output,
                -dy
            )


        elif dy > 0:
            output = seams_insertion_grayscale(
                output,
                dy
            )

        output = rotate_image(output, clockwise=False)

    return output



def modify_density(
    gray,
    scale_x=1.0,
    scale_y=1.0,
    density_x=0.0,
    density_y=0.0
):
    """
    Combined pipeline:

    1. Start with original image H x W.
    2. Globally resize by scale_x / scale_y.
    3. Compute balancing seams needed to return to original dimensions.
    4. Compute extra density seams from density_x / density_y.
    5. Add both seam counts together.
    6. Apply one combined seam-carving/insertion operation.

    density_x:
        > 0 inserts black vertical seams  -> more horizontal gaps
        < 0 removes vertical seams        -> fewer horizontal gaps

    density_y:
        > 0 inserts black horizontal seams -> more vertical gaps
        < 0 removes horizontal seams       -> fewer vertical gaps

    If force_original_size=True:
        after combined seam operations, the image is resized back to the
        original dimensions as a final safety step.

    If force_original_size=False:
        the final dimensions reflect the density seam changes.
    """

    original_h, original_w = gray.shape[:2]

    stretched = global_resize(
        gray,
        scale_x=scale_x,
        scale_y=scale_y,
        interpolation=cv2.INTER_LINEAR
    )

    stretched_h, stretched_w = stretched.shape[:2]

    print(f"Original shape:       {(original_h, original_w)}")
    print(f"After global resize:  {(stretched_h, stretched_w)}")


    balance_dx = original_w - stretched_w
    balance_dy = original_h - stretched_h

    density_dx = int(round(original_w * density_x))
    density_dy = int(round(original_h * density_y))

    total_dx = balance_dx + density_dx
    total_dy = balance_dy + density_dy

    print("Seam plan:")
    print(f"  balance_dx = {balance_dx}")
    print(f"  density_dx = {density_dx}")
    print(f"  total_dx   = {total_dx}")
    print(f"  balance_dy = {balance_dy}")
    print(f"  density_dy = {density_dy}")
    print(f"  total_dy   = {total_dy}")


    balanced = seam_carve_grayscale_by_delta(
        stretched,
        dx=total_dx,
        dy=total_dy,
    )

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    print(f"Final output shape: {balanced.shape}")

    return balanced
