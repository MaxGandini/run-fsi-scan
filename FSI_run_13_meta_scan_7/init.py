
import requests 
import json
import os
import cv2
import numpy as np
from pathlib import Path


from utilities import get_stack_around_from_list, read_order_img
from plots import plot_projection_results, plot_selected_crossings_on_image

from parameters import (  # noqa: E402
    NONE_STANDARD,
    TOP_INTENSITY,
    BOTTOM_INTENSITY,
    FROM_IMAGE,
    TO_IMAGE,
    PROJECT_FOLDER,
    PROJECTION_FOLDER,
    TEST_FOLDER,
    CROSSINGS_FOLDER,

    STRICTNESS,

    ADAPTIVE_VALUE,
    
    CHARACTERISTIC_LENGTH,
    CHARACTERISTIC_THRESH,
    EPSILON
)


#SCANNER imports-----------------------------

def filter_monitor_corr(image_list,index,n_image: int,top_intensity=BOTTOM_INTENSITY,bottom_intensity=BOTTOM_INTENSITY,*args, **kwargs):

        stack = get_stack_around_from_list(image_list, index, n_image).astype(np.float32)
        product_projection = np.prod(stack, axis=0)
        product_projection_norm = 255 * (product_projection / np.max(product_projection))
        product_projection_clipped = np.clip(product_projection_norm, 0, 255).astype(np.uint8)

        height, width = product_projection_clipped.shape[:2]

        product_projection_clipped[product_projection_clipped> top_intensity] = 255
        product_projection_clipped[product_projection_clipped< bottom_intensity] = 0

        return product_projection_clipped

def reader(from_image=FROM_IMAGE, to_image=TO_IMAGE, input_dir=PROJECT_FOLDER,
           prefix="Corte ", image_list=None, json_path="config.json"):

    def is_cloud_env():
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
            return True
        if os.path.exists("/.dockerenv"):
            return True
        return False

    if image_list is not None:
        file_list = [f"provided_image_{i}" for i in range(len(image_list))]
        return image_list, file_list, [{}] * len(image_list)

    if is_cloud_env() and os.path.exists(json_path):
        with open(json_path, "r") as f:
            cfg = json.load(f)

        if isinstance(cfg, dict) and "urls" in cfg:
            entries = [{"url": url, "filename": f"image_{i}.png"} for i, url in enumerate(cfg["urls"])]
            from_image = cfg.get("from_image", from_image)
            to_image = cfg.get("to_image", to_image)
        elif isinstance(cfg, list):
            entries = cfg
        else:
            raise ValueError("Invalid JSON structure in config.json")

        entries = entries[from_image - 1 : to_image]  # slice range

        image_list, file_list, metadata_list = [], [], []

        for entry in entries:
            url = entry.get("url")
            filename = entry.get("filename", os.path.basename(url or "unknown"))
            metadata = {k: v for k, v in entry.items() if k not in ("url", "filename")}

            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    image_list.append(img)
                    file_list.append(filename)
                    metadata_list.append(metadata)
                else:
                    print(f"⚠️ Could not decode {filename} ({url})")
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")

        return image_list, file_list, metadata_list

    file_list = read_order_img(from_image, to_image, input_dir, prefix)
    image_list = [cv2.imread(image) for image in file_list]
    metadata_list = [{} for _ in file_list]

    return image_list, file_list, metadata_list

def multiply_correlated_stack(
    image_list,
    correlation_window=1,
    bottom_intensity=BOTTOM_INTENSITY,
    top_intensity=TOP_INTENSITY,
):

    correlated_stack_list = []

    for idx, image in enumerate(image_list):
        if image is None:
            continue

        correlated = filter_monitor_corr(
            image_list,
            idx,
            correlation_window,
            bottom_intensity=bottom_intensity,
            top_intensity=top_intensity
        )

        correlated_stack_list.append(correlated)

    return correlated_stack_list

from parameters import (    
    TEST_FOLDER
)

def estimator(image, idx, depth):
    print("running estimator")
    threshold_h,threshold_v,h_derivative,v_derivative = compute_directional_projection_gray(image,CHARACTERISTIC_LENGTH,CHARACTERISTIC_LENGTH)

    x_h = np.arange(len(h_derivative))
    x_v = np.arange(len(v_derivative))

    width_h, widths_h,cutoffs_h = mean_zero_crossing_width(x_h, h_derivative)
    width_v, widths_v, cutoffs_v = mean_zero_crossing_width(x_v, v_derivative)

    diffs = np.concatenate([
        compute_cutoff_differences(cutoffs_v),
        compute_cutoff_differences(cutoffs_h)
    ], axis=0)

    darkest_h = select_darkest_cutoffs(image, cutoffs_h, "horizontal", percentage=STRICTNESS,characteristic_length=CHARACTERISTIC_THRESH)
    darkest_v = select_darkest_cutoffs(image, cutoffs_v, "vertical", percentage=STRICTNESS,characteristic_length=CHARACTERISTIC_THRESH)

    plot_projection_results(
        threshold_h, threshold_v, h_derivative, v_derivative,
        CHARACTERISTIC_LENGTH, CHARACTERISTIC_LENGTH,
        PROJECTION_FOLDER , image_name=f"Projected_{idx}",
        h_cutoffs=darkest_h, v_cutoffs=darkest_v,
    )

    selected_crossings, median_int= measure_crossings(image,darkest_h,darkest_v)
    plot_selected_crossings_on_image(image,selected_crossings,projection_folder=CROSSINGS_FOLDER,idx=idx)

    print("ending estimator")
    return width_h , width_v, diffs, cutoffs_v,cutoffs_h,median_int, selected_crossings


# result = {"threshold": black_thresholds, "cuts_h": cuts_h ,"cuts_v": cuts_v, "most_targets": most_targets_path, "target_width_h":target_width_h, "target_width_v":target_width_v,"correlation_window":windows}

def measure_crossings(image, darkest_h, darkest_v, percentage=STRICTNESS):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    h_coords = [
        entry[1][0] if isinstance(entry, (tuple, list)) and len(entry) == 3 else entry
        for entry in darkest_h
    ]
    v_coords = [
        entry[1][0] if isinstance(entry, (tuple, list)) and len(entry) == 3 else entry
        for entry in darkest_v
    ]

    h_valid = [y for y in h_coords if 0 <= y < gray.shape[0]]
    v_valid = [x for x in v_coords if 0 <= x < gray.shape[1]]

    crossings = []
    for y in h_valid:
        for x in v_valid:
            intensity = float(gray[int(y), int(x)])
            crossings.append((int(x), int(y), intensity))

    if not crossings:
        return [], None

    intensities = np.array([z for _, _, z in crossings])
    threshold = np.percentile(intensities, percentage * 100)
    selected = [(x, y, z) for x, y, z in crossings if z <= threshold]

    median_intensity = float(np.mean([z for _, _, z in selected])) if selected else None

    return selected, median_intensity

def measure_cross_intensity(image, darkest_h, darkest_v, percentage=STRICTNESS):

    selected_crossings = measure_crossings(image, darkest_h, darkest_v, percentage)
    if not selected_crossings:
        return None, None, []

    intensities = np.array([c[2] for c in selected_crossings])
    mean_intensity = np.mean(intensities)
    median_intensity = np.median(intensities)

    return mean_intensity, median_intensity, selected_crossings

def mean_zero_crossing_width(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of equal length")

    # slope (first derivative sign)
    dy = np.gradient(y, x)
    slope_sign = np.sign(dy)

    # zero crossings
    sign_y = np.sign(y)
    crossings = np.where(sign_y[:-1] * sign_y[1:] < 0)[0]

    widths = []
    cutoffs = []

    for idx in crossings:
        # interpolate zero crossing
        x0 = x[idx] - y[idx] * (x[idx+1] - x[idx]) / (y[idx+1] - y[idx])
        y0 = 0.0  # by definition

        # search left until slope changes
        i = idx
        base_sign = slope_sign[i]
        while i > 0 and slope_sign[i] == base_sign:
            i -= 1
        x_left = x[i]
        # interpolate y at x_left
        if i+1 < len(x):
            y_left = y[i] + (y[i+1]-y[i])*(x_left - x[i])/(x[i+1]-x[i])
        else:
            y_left = y[i]

        # search right until slope changes
        j = idx + 1
        base_sign = slope_sign[j]
        while j < len(slope_sign)-1 and slope_sign[j] == base_sign:
            j += 1
        x_right = x[j]
        # interpolate y at x_right
        if j+1 < len(x):
            y_right = y[j] + (y[j+1]-y[j])*(x_right - x[j])/(x[j+1]-x[j])
        else:
            y_right = y[j]

        width = (x_right - x0) + (x0 - x_left)
        widths.append(width)
        cutoffs.append(((x_left, y_left), (x0, y0), (x_right, y_right)))

    if not widths:
        return None, [], []

    return np.median(widths), widths, cutoffs

def measure_target_widths(result_list, max_radius=200):
    all_widths = []
    save_dir = Path("/home/Xilian/recursive/aws_run_meta/FSI_run_11_meta_scan_4/test_folder/boxes_length")
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, (depth, dark_median, selected_crossings, image) in enumerate(result_list):
        if depth is None or dark_median is None or selected_crossings is None or image is None:
            all_widths.append([])
            continue

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis_image = image.copy()
        else:
            gray = image.copy()
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        binary = (gray > dark_median).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        h, w = binary.shape
        widths = []

        for p in selected_crossings:
            if len(p) < 2:
                continue
            x, y = int(p[0]), int(p[1])
            if not (0 <= x < w and 0 <= y < h):
                continue

            # --- Horizontal expansion ---
            left = x
            for i in range(1, max_radius + 1):
                if x - i < 0 or binary[y, x - i] == 255:
                    break
                left = x - i

            right = x
            for i in range(1, max_radius + 1):
                if x + i >= w or binary[y, x + i] == 255:
                    break
                right = x + i

            horiz_width = right - left

            # --- Vertical expansion ---
            up = y
            for i in range(1, max_radius + 1):
                if y - i < 0 or binary[y - i, x] == 255:
                    break
                up = y - i

            down = y
            for i in range(1, max_radius + 1):
                if y + i >= h or binary[y + i, x] == 255:
                    break
                down = y + i

            vert_width = down - up

            # ✅ Measure only the smallest side
            min_width = float(min(horiz_width, vert_width))

            # ✅ Ignore very small widths (< 3 px)
            if min_width < 3:
                continue

            widths.append(min_width)

            # --- Draw box (rectangle) ---
            color = (0, 255, 0)
            cv2.rectangle(vis_image, (left, up), (right, down), color, 1)
            cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)

        # Save the annotated image
        out_path = save_dir / f"boxes_{idx:03d}.png"
        cv2.imwrite(str(out_path), vis_image)

        all_widths.append(widths)

    return all_widths

from itertools import combinations
from typing import List, Tuple, Union

def select_darkest_cutoffs(
    image,
    cutoffs,
    direction="horizontal",
    percentage=None,
    characteristic_length=CHARACTERISTIC_THRESH,
):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    items = []  # (entry, pos, intensity)
    for entry in (cutoffs or []):
        zero = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 3 else entry

        if isinstance(zero, (list, tuple, np.ndarray)):
            if len(zero) == 0:
                continue
            try:
                pos = int(zero[0])
            except Exception:
                continue
        else:
            try:
                pos = int(zero)
            except Exception:
                continue

        if direction == "horizontal":
            if not (0 <= pos < gray.shape[0]):
                continue
            intensity = float(np.mean(gray[pos, :]))
        elif direction == "vertical":
            if not (0 <= pos < gray.shape[1]):
                continue
            intensity = float(np.mean(gray[:, pos]))
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")

        items.append((entry, pos, intensity))

    if not items:
        return []

    # sort by intensity (darkest first)
    items.sort(key=lambda t: t[2])

    # if percentage requested, compute target count
    total = len(items)
    target_n = None
    if percentage is not None:
        target_n = max(1, int(total * percentage))

    # select while enforcing minimum distance (characteristic_length)
    char_len = int(round(characteristic_length))
    selected = []
    selected_positions = []

    for entry, pos, inten in items:
        conflict = any(abs(pos - p) < char_len for p in selected_positions)
        if conflict:
            continue
        selected.append((entry, pos, inten))
        selected_positions.append(pos)
        if target_n is not None and len(selected) >= target_n:
            break

    # return selected cutoffs sorted by position (ascending)
    selected.sort(key=lambda t: t[1])
    return [t[0] for t in selected]

Cutoff = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
CutoffList = List[Cutoff]

def _is_cutoff_triple(obj) -> bool:
    return (
        isinstance(obj, (list, tuple))
        and len(obj) == 3
        and isinstance(obj[0], (list, tuple))
        and isinstance(obj[1], (list, tuple))
        and isinstance(obj[2], (list, tuple))
        and len(obj[0]) == 2
        and len(obj[1]) == 2
        and len(obj[2]) == 2
        and isinstance(obj[0][0], (int, float))
    )


def _flatten_cutoffs(cutoffs: Union[CutoffList, List[CutoffList]]) -> CutoffList:
    """
    Accept either:
      - a flat list of cutoffs: [cutoff, cutoff, ...]
      - or a nested list: [ [cutoff..], [cutoff..], ... ]
    Returns a flat list of cutoffs.
    """
    if cutoffs is None:
        return []
    # already flat?
    if isinstance(cutoffs, (list, tuple)) and len(cutoffs) > 0 and _is_cutoff_triple(cutoffs[0]):
        return list(cutoffs)
    flat = []
    for item in cutoffs:
        if item is None:
            continue
        if _is_cutoff_triple(item):
            flat.append(item)
            continue
        if isinstance(item, (list, tuple)):
            for sub in item:
                if _is_cutoff_triple(sub):
                    flat.append(sub)
    return flat


def calculate_cutoff_width_vectors(cutoffs: Union[CutoffList, List[CutoffList]]) -> np.ndarray:
    """
    From cutoffs list returns width vectors for every cutoff:
        width_vector = (dx, dy) = (x_right - x_left, y_right - y_left)
    Input cutoffs format: [ ((x_left,y_left),(x0,y0),(x_right,y_right)), ... ]
    Returns: ndarray shape (N,2) dtype float
    """
    flat = _flatten_cutoffs(cutoffs)
    if not flat:
        return np.empty((0, 2), dtype=float)
    w = []
    for left, zero, right in flat:
        x_left, y_left = float(left[0]), float(left[1])
        x_right, y_right = float(right[0]), float(right[1])
        w.append((x_right - x_left, y_right - y_left))
    return np.asarray(w, dtype=float)

def compute_cutoff_differences(
    cutoffs: Union[CutoffList, List[CutoffList]],
    n_pairs: int = 4,
    method: str = "consecutive",
) -> np.ndarray:
    """
    Compute differences (Δx, Δy) using the left/right cutoff coordinates.

    Methods:
      - "consecutive" : compute diffs between consecutive width vectors:
            diffs[i] = width_vector[i+1] - width_vector[i], up to n_pairs items
      - "pairwise"    : compute all pairwise differences among the first (n_pairs+1)
                        width vectors (combinations), returning each (dx, dy)

    Returns:
        diffs: ndarray shape (M,2) where each row is (Δx, Δy)
    """
    w = calculate_cutoff_width_vectors(cutoffs)  # shape (N,2)
    N = len(w)
    if N < 2:
        return np.empty((0, 2), dtype=float)

    if method == "consecutive":
        max_pairs = min(n_pairs, N - 1)
        diffs = np.zeros((max_pairs, 2), dtype=float)
        for i in range(max_pairs):
            diffs[i] = w[i + 1] - w[i]
        return diffs

    if method == "pairwise":
        num_points = min(n_pairs + 1, N)
        subset = w[:num_points]
        combs = list(combinations(range(num_points), 2))
        diffs = []
        for i, j in combs:
            diffs.append(subset[j] - subset[i])
        return np.asarray(diffs, dtype=float)

    raise ValueError("Unknown method. Use 'consecutive' or 'pairwise'.")

def compute_directional_projection_gray(
    image,
    x_line: int,
    y_line: int,
    zero_threshold: int = int(CHARACTERISTIC_THRESH),
    characteristic_length: int = int(CHARACTERISTIC_LENGTH)
):
    """
    Compute horizontal and vertical projection profiles and derivatives.
    Derivative values are smoothed using a Savitzky–Golay filter and
    zeroed out within ±zero_threshold points around their strongest extrema.

    Parameters:
        image: np.ndarray
        x_line, y_line: reference coordinates
        zero_threshold: number of points around each extreme to zero out
        characteristic_length: window length for Savitzky–Golay smoothing

    Returns:
        h_profile, v_profile, h_derivative, v_derivative
    """
    import numpy as np
    from scipy.signal import savgol_filter

    if image is None:
        raise ValueError("Empty image provided")

    gray_image = image.astype(np.float64)

    h_ref_line = gray_image[y_line, :]
    v_ref_line = gray_image[:, x_line]

    h_profile = np.array([
        np.sum(gray_image[r, :] * h_ref_line) for r in range(gray_image.shape[0])
    ])
    v_profile = np.array([
        np.sum(gray_image[:, c] * v_ref_line) for c in range(gray_image.shape[1])
    ])

    # raw derivatives
    h_derivative_ = np.diff(h_profile)
    v_derivative_ = np.diff(v_profile)

    # ensure valid window for savgol_filter
    def safe_window(length):
        win = min(characteristic_length, length - 1)
        if win % 2 == 0:
            win = max(3, win - 1)
        return max(3, win)

    def smooth_and_zero(arr):
        if arr.size == 0:
            return arr
        # apply Savitzky–Golay smoothing
        win = safe_window(len(arr))
        arr_smooth = savgol_filter(arr, window_length=win, polyorder=2, mode='mirror')

        # normalize
        max_abs = np.max(np.abs(arr_smooth))
        arr_norm = arr_smooth / max_abs if max_abs != 0 else arr_smooth

        arr_copy = arr_norm.copy()
        max_idx = np.argmax(arr_copy)
        min_idx = np.argmin(arr_copy)

        # zero out ±zero_threshold points around extrema
        for idx in [max_idx, min_idx]:
            start = max(0, idx - zero_threshold)
            end = min(len(arr_copy), idx + zero_threshold + 1)
            arr_copy[start:end] = 0

        return arr_copy

    h_derivative = smooth_and_zero(h_derivative_)
    v_derivative = smooth_and_zero(v_derivative_)

    return h_profile, v_profile, h_derivative, v_derivative

def count_crossings_index(signal, threshold):
    s = np.array(signal)
    
    pos_crossing_indices = np.where((s[:-1] < threshold) & (s[1:] > threshold))[0] + 1
    pos_crossings = len(pos_crossing_indices)
    
    neg_crossing_indices = np.where((s[:-1] > -threshold) & (s[1:] < -threshold))[0] + 1
    neg_crossings = len(neg_crossing_indices)
    
    return pos_crossings, neg_crossings, pos_crossing_indices.tolist(), neg_crossing_indices.tolist()

def find_peak_threshold(signal, steps=200):

    max_val = np.max(np.abs(signal))*0.8
    thresholds = np.linspace(max_val, max_val / steps, steps)

    best_threshold = thresholds[0]
    best_diff = float('inf')

    for threshold in thresholds:
        pos_cross, neg_cross = count_crossings(signal, threshold)
        diff = abs(pos_cross - neg_cross)

        populated = (pos_cross > 10 and neg_cross > 10)
        if diff < best_diff and populated:
            best_diff = diff
            best_threshold = threshold

    return best_threshold

def count_crossings(signal, threshold):
    s = signal

    pos_crossings = np.sum((s[:-1] < threshold) & (s[1:] > threshold))
    neg_crossings = np.sum((s[:-1] > -threshold) & (s[1:] < -threshold))

    return pos_crossings, neg_crossings

def measure_targets(number_of_targets_pos, number_of_targets_neg, index_of_targets_pos_, index_of_targets_neg_, percentage_per_image= 0.33):

    target_width, index_of_targets_pos, index_of_targets_neg = coerce_indexes_and_measure_width(index_of_targets_pos_, index_of_targets_neg_)

    searched_targets = int(percentage_per_image * (number_of_targets_pos + number_of_targets_neg) / 2)

    if searched_targets == 0:
        searched_targets = 1  

    cuts = []
    count = 0

    for i, target in enumerate(index_of_targets_pos):
            count += 1
            if count == searched_targets and i < len(index_of_targets_pos) - 1:
                next_target = index_of_targets_pos[i + 1]
                midpoint = (target + next_target) / 2
                cuts.append(midpoint)
                count = 0

    return target_width, cuts

def coerce_indexes_and_measure_width(index_of_targets_pos, index_of_targets_neg):
    def safe_mean_diff(a, b):
        if len(a) < 1 or len(b) < 1:
            return float("inf")  # or np.nan if you want to signal failure
        return np.mean(np.array(a) - np.array(b))

    arrays_dont_exactly_coincide = (len(index_of_targets_pos) != len(index_of_targets_neg))

    if arrays_dont_exactly_coincide:
        positive1, negative1 = match_length_truncate(index_of_targets_pos, index_of_targets_neg)
        positive2, negative2 = match_length_truncate(index_of_targets_pos, index_of_targets_neg, shift=1)

        forward_diff_1 = safe_mean_diff(positive1[1:], negative1[:-1])
        backward_diff_1 = safe_mean_diff(positive1[:-1], negative1[1:])

        forward_diff_2 = safe_mean_diff(positive2[1:], negative2[:-1])
        backward_diff_2 = safe_mean_diff(positive2[:-1], negative2[1:])

        width_1 = min(forward_diff_1, backward_diff_1)
        width_2 = min(forward_diff_2, backward_diff_2)

        if width_1 < width_2:
            target_width = width_1
            index_of_targets_pos_ = positive1
            index_of_targets_neg_ = negative1
        else:
            target_width = width_2
            index_of_targets_pos_ = positive2
            index_of_targets_neg_ = negative2

    else:
        forward_diff = safe_mean_diff(index_of_targets_pos[1:], index_of_targets_neg[:-1])
        backward_diff = safe_mean_diff(index_of_targets_pos[:-1], index_of_targets_neg[1:])
        target_width = min(forward_diff, backward_diff)
        index_of_targets_pos_ = index_of_targets_pos
        index_of_targets_neg_ = index_of_targets_neg

    # If both forward/backward diffs are inf or nan, replace with fallback
    if not np.isfinite(target_width):
        target_width = 0  # or some sensible fallback value

    return target_width, index_of_targets_pos_, index_of_targets_neg_

def match_length_truncate(a, b, shift: int = 0):
    len_a, len_b = len(a), len(b)

    if len_a > len_b and shift != 0:
        if shift > 0:
            a = a[shift:]
        elif shift < 0:
            a = a[:shift]

    elif len_b > len_a and shift != 0:
        if shift > 0:
            b = b[shift:]
        elif shift < 0:
            b = b[:shift]

    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]

#binarizer imports-----------------------------


#binarizer imports ----- Filter --------

from skimage.morphology import skeletonize as sk_skeletonize

def skeletonize(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    inverted = cv2.bitwise_not(image)

    binary_bool = inverted.astype(bool)

    skeleton = sk_skeletonize(binary_bool)

    skeleton_img = np.where(skeleton, 0, 255).astype(np.uint8)

    return skeleton_img

import inspect 

def filter_monitor(image, idx, filters_parameters, *args, **kwargs):
    current = image
    for f_idx, filter_parameter in enumerate(filters_parameters):
        filter_func, f_args, f_kwargs = filter_parameter
        call_kwargs = dict(f_kwargs)

        # inject idx only if function supports it
        try:
            sig = inspect.signature(filter_func)
            if "idx" in sig.parameters:
                call_kwargs["idx"] = idx
        except (ValueError, TypeError):
            pass  # built-ins like OpenCV

        # run filter
        current = filter_func(current, *f_args, **call_kwargs)

        # directory handling
        filter_name = getattr(filter_func, "__name__", f"filter_{f_idx}")
        out_dir = TEST_FOLDER / filter_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"image_{idx}.png"
        cv2.imwrite(str(out_path), current)

    return current

def filter_monitor_manual(image, idx,inferred_parameters):

    current = image
    widths, image_intensity = inferred_parameters
    mean_width = np.median(widths)

    if mean_width is not None and not np.isnan(mean_width):
        KSIZE = int( mean_width )
        if KSIZE<3: 
            KSIZE=3
    else:
        KSIZE = 3

    print("the inferred parameters are: " ,KSIZE,type(KSIZE),image_intensity)

    # current = perona_malik_filter(current, idx=idx)  # pass idx if supported
    # out_dir = TEST_FOLDER / "perona_malik_filter"
    # out_dir.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite(str(out_dir / f"image_{idx}.png"), current)

    current = cv2.adaptiveThreshold(
        current,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_VALUE,
        2
    )

    current = open_close_repeat(current,ksize=KSIZE,repeats=2)
    out_dir = TEST_FOLDER / "open_close_repeat"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"image_{idx}.png"), current)

    out_dir = TEST_FOLDER / "adaptiveThreshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"image_{idx}.png"), current)

    current = skeletonize(current)
    out_dir = TEST_FOLDER / "skeletonize"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"image_{idx}.png"), current)

    return current

def perona_malik_filter(
    img: np.ndarray,
    idx,
    n_iter: int = 20,
    delta_t: float = 0.125,
    kappa: float = 30,
    option: int = 1,
) -> np.ndarray:

    if img.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image")

    img = img.astype(np.float32)
    diff = img.copy()

    for _ in range(n_iter):
        # shift differences
        nablaN = np.roll(diff, -1, axis=0) - diff
        nablaS = np.roll(diff, 1, axis=0) - diff
        nablaE = np.roll(diff, -1, axis=1) - diff
        nablaW = np.roll(diff, 1, axis=1) - diff

        if option == 1:
            # exponential conduction
            cN = np.exp(-(nablaN / kappa) ** 2)
            cS = np.exp(-(nablaS / kappa) ** 2)
            cE = np.exp(-(nablaE / kappa) ** 2)
            cW = np.exp(-(nablaW / kappa) ** 2)
        elif option == 2:
            # inverse quadratic conduction
            cN = 1.0 / (1.0 + (nablaN / kappa) ** 2)
            cS = 1.0 / (1.0 + (nablaS / kappa) ** 2)
            cE = 1.0 / (1.0 + (nablaE / kappa) ** 2)
            cW = 1.0 / (1.0 + (nablaW / kappa) ** 2)
        else:
            raise ValueError("option must be 1 or 2")

        diff += delta_t * (cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW)

    out = np.clip(diff, 0, 255).astype(np.uint8)

    out_filt_path = Path("TEST_FOLDER") / "new_filter" / f"filtered_perona_malik{idx}.png"
    out_filt_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_filt_path), out)

    return out

def edge_preserving_smoothing(
    img: np.ndarray,
    idx,
    method: str = "bilateral",
    d: int = 9,
    sigmaColor: float = 75,
    sigmaSpace: float = 75,
    sigma_s: float = 70,
    sigma_r: float = 0.9,
    radius: int = 8,
    eps: float = 0.01*255*255
) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image")

    if method == "bilateral":
        out = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    elif method == "epf":
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out_color = cv2.edgePreservingFilter(color_img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        out = cv2.cvtColor(out_color, cv2.COLOR_BGR2GRAY)

    elif method == "guided":
        if not hasattr(cv2, "ximgproc"):
            raise ImportError("cv2.ximgproc not available, install opencv-contrib-python")
        out = cv2.ximgproc.guidedFilter(guide=img, src=img, radius=radius, eps=eps)

    else:
        raise ValueError(f"Unknown method: {method}")

    out_filt_path = TEST_FOLDER / "new_filter" / f"filtered_edge{idx}.png"
    out_filt_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_filt_path), out)

    return out

def open_close_repeat(img, ksize=3, repeats=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    out = img.copy()
    for _ in range(repeats):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out

FILTERS_PARAMETERS = [
    (
        open_close_repeat,
        (),
        {}
    ),
    # (
    #     perona_malik_filter,
    #     (),
    #     {}
    # ),
    (
        cv2.adaptiveThreshold,
        (255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_VALUE, 2),
        {}
    ),
    (
        skeletonize,
        (),
        {}
    ),
]

#binarizer imports ----- Pointer --------

from collections import defaultdict
from scipy.spatial import KDTree
from scipy.ndimage import convolve

def find_strong_points_in_stack(image_list,index, n_image=1,
                                tile_size=CHARACTERISTIC_LENGTH, min_occurrences=1,
                                max_cluster_distance=CHARACTERISTIC_LENGTH+1,
                                suffix="fixed_points"):
    """
    Finds strong points across a stack and clusters them by proximity.
    """
    print("finding points in stack")
    stack = get_stack_around_from_list(image_list, index, n_image).astype(np.float32)

    occurrence_counter_main = defaultdict(int)
    occurrence_counter_isolated = defaultdict(int)

    for binary_image in stack:
        buckets = find_extreme_points_by_neighbors(binary_image, tile_size=tile_size)
        classified = classify_points_by_neighbors(buckets)

        for key in ("quad", "group"):
            for tile_coord, points in classified.get(key, []):
                for x, y in points:
                    occurrence_counter_main[(x, y)] += 1

        for tile_coord, points in classified.get("isolated", []):
            for x, y in points:
                occurrence_counter_isolated[(x, y)] += 1

    main_points = [pt for pt, count in occurrence_counter_main.items() if count >= min_occurrences]
    # isolated_points = [pt for pt, count in occurrence_counter_isolated.items() if count >= 2]

    all_points = list(set(main_points))

    def cluster_points_and_get_centers(points, max_distance):
        if not points:
            return []
        points_arr = np.array(points)
        tree = KDTree(points_arr)
        visited = set()
        centers = []
        for i, p in enumerate(points_arr):
            if i in visited:
                continue
            idxs = tree.query_ball_point(p, r=max_distance)
            cluster = points_arr[idxs]
            visited.update(idxs)
            center = cluster.mean(axis=0)
            x, y = center
            centers.append((x,y,index))
        return centers

    clustered_centers = cluster_points_and_get_centers(all_points, max_cluster_distance)

    return clustered_centers

def find_extreme_points_by_neighbors(binary_image, tile_size=CHARACTERISTIC_LENGTH):
    image = (binary_image == 0).astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    neighbor_count = convolve(image, kernel, mode='constant', cval=0)

    points_and_counts = np.argwhere(image == 1)
    neighbor_counts = neighbor_count[points_and_counts[:,0], points_and_counts[:,1]]

    buckets = defaultdict(list)
    for (y, x), count in zip(points_and_counts, neighbor_counts):
        tile_x = x // tile_size
        tile_y = y // tile_size
        buckets[(tile_x, tile_y)].append(((x, y), count))

    return buckets

def classify_points_by_neighbors(buckets):
    classified = {
        'quad': [],      # points with 4 neighbors
        'group': [],     # points with 2 or 3 neighbors
        'isolated': []   # points with 1 neighbor
    }

    for tile_coord, points_and_counts in buckets.items():
        quad_points = [p for p, c in points_and_counts if c == 4]
        group_points = [p for p, c in points_and_counts if c == 3]
        isolated_points = [p for p, c in points_and_counts if c == 1]

        if quad_points:
            classified['quad'].append((tile_coord, quad_points))
        if group_points:
            classified['group'].append((tile_coord, group_points))
        if isolated_points:
            classified['isolated'].append((tile_coord, isolated_points))

    return classified


#Vectorizer imports -------------

import math

from PIL import Image
from shapely.geometry import LineString, MultiLineString, Point, mapping
from shapely.ops import unary_union


class Cross:
    def __init__(self, point: Point):
        self.point = point
        self.arms = {'up': False, 'down': False, 'left': False, 'right': False}
        self.connections = []  
        self.neighbors = {'up': None, 'down': None, 'left': None, 'right': None}

    def add_connection(self, direction: str, line: LineString):
        if not self.arms[direction]:
            self.arms[direction] = True
            self.connections.append((direction, line))

class Cursor:
    def __init__(self,intensity_threshold):
        self.intensity_thresh = intensity_threshold
        self.scanned_points = {}

    def scan(self, start_cross: Cross, end_cross: Cross, img_cv,max_deviation_deg: float = 20):

        dx = end_cross.point.x - start_cross.point.x
        dy = end_cross.point.y - start_cross.point.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return None

        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
            nominal_angle = 0 if dx > 0 else 180
        else:
            direction = 'up' if dy > 0 else 'down'
            nominal_angle = 90 if dy > 0 else -90

        reverse = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[direction]

        if start_cross.arms[direction] or end_cross.arms[reverse]:
            return None

        actual_angle = math.degrees(math.atan2(dy, dx))
        if abs(actual_angle - nominal_angle) > max_deviation_deg:
            return None

        img = Image.fromarray(img_cv)
        x0, y0 = int(start_cross.point.x), int(start_cross.point.y)
        x1, y1 = int(end_cross.point.x), int(end_cross.point.y)

        line_coords = list(zip(np.linspace(x0, x1, num=100, dtype=int),
                               np.linspace(y0, y1, num=100, dtype=int)))

        def kernel_intensity(x, y):
            vals = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    xi, yj = x + i, y + j
                    if 0 <= xi < img.width and 0 <= yj < img.height:
                        vals.append(img.getpixel((xi, yj)))
            return min(vals) if vals else 255

        intensities = np.array([kernel_intensity(x, y) for x, y in line_coords])

        if np.all(intensities <= self.intensity_thresh):
            line = LineString(line_coords)
            start_cross.add_connection(direction, line)
            end_cross.add_connection(reverse, line)
            self.scanned_points[(x0, y0, x1, y1)] = {'intensity_min': intensities.min()}
            return line, direction, reverse

        return None

class MainGrid:
    def __init__(self):
        self.lines = []
        self.crosses = []
        self._kdtree = None
        self._cross_points = None  

    def add_lines(self, lines):
        self.lines.extend(lines)

    def create_crosses_from_points(self, points):
        self.crosses = [Cross(Point(p)) for p in points]
        self._cross_points = np.array([(c.point.x, c.point.y) for c in self.crosses])
        if len(self._cross_points) > 0:
            self._kdtree = KDTree(self._cross_points)

    def neighborhood_intensity(img,cx, cy, size=5):
        offsets = range(-(size // 2), size // 2 + 1)
        vals = []
        for dx in offsets:
            for dy in offsets:
                x, y = cx + dx, cy + dy
                if 0 <= x < img.width and 0 <= y < img.height:
                    vals.append(img.getpixel((x, y)))
        return np.mean(vals) if vals else 0

    def compute_neighbors(self):
        """
        uses kdtree to compute nearest neighbors in four cardinal directions.
        """
        if self._kdtree is None:
            raise RuntimeError("kdtree not built. call create_crosses_from_points first.")

        for idx, c1 in enumerate(self.crosses):
            x0, y0 = c1.point.x, c1.point.y
            nearest = {'up': (None, float('inf')),
                       'down': (None, float('inf')),
                       'left': (None, float('inf')),
                       'right': (None, float('inf'))}

            distances, indexes = self._kdtree.query([x0, y0], k=len(self.crosses))
            if np.isscalar(indexes):
                indexes = [indexes]
                distances = [distances]

            for i, d in zip(indexes, distances):
                if i == idx or d == 0:
                    continue
                c2 = self.crosses[i]
                dx = c2.point.x - x0
                dy = c2.point.y - y0
                angle = math.degrees(math.atan2(dy, dx))

                if -45 <= angle <= 45 and dx > 0:
                    if d < nearest['right'][1]:
                        nearest['right'] = (c2, d)
                elif 45 < angle <= 135 and dy > 0:
                    if d < nearest['up'][1]:
                        nearest['up'] = (c2, d)
                elif -135 <= angle < -45 and dy < 0:
                    if d < nearest['down'][1]:
                        nearest['down'] = (c2, d)
                elif angle <= -135 or angle >= 135:
                    if dx < 0 and d < nearest['left'][1]:
                        nearest['left'] = (c2, d)

            for d in ('up','down','left','right'):
                c1.neighbors[d] = nearest[d][0]

    def generate_connections(self, cursor, image):
        for c in self.crosses:
            for direction, neighbor in c.neighbors.items():
                if neighbor:
                    res = cursor.scan(c, neighbor, image)
                    if res:
                        line, direction, reverse = res
                        self.add_lines([line])
                        neighbor.add_connection(reverse, line)

    def plot_grid(self, img_path: str | Path, output_folder: Path,
                  plot_crosses: bool = True):
        img = cv2.imread(str(img_path))
        for line in self.lines:
            pts = np.array(line.coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 0), 1)
        if plot_crosses:
            for cross in self.crosses:
                x, y = int(cross.point.x), int(cross.point.y)
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / Path(img_path).name
        cv2.imwrite(str(output_path), img)

class Star(Cross):
    def __init__(self, cross: Cross, z_coordinate=0):
        super().__init__(cross.point)
        self.connections = cross.connections.copy()
        self.arms = cross.arms.copy()

        self.neighbors = cross.neighbors.copy()
        self.neighbors["zup"] = None
        self.neighbors["zdown"] = None

        x, y = self.point.x, self.point.y
        self.point = Point(x, y, z_coordinate)

class Grid3D:
    def __init__(self, grids=None):
        self.grids = grids if grids is not None else []
        self.fuzz_masks = []       
        self.filtered_shapes= []       
        self.stars= []
        self.filtered_crosses= []
        self.filtered_structure= []
        self.inferred_structure = []

    def build_stars_and_clean(self,tolerance=CHARACTERISTIC_LENGTH):
        for idx, grid in enumerate(self.grids):

            for cross in grid.crosses:    
                if len(cross.connections)>0:
                    self.stars.append(Star(cross, z_coordinate=idx))

        self.inferred_structure=self.stars
        print("initialized inferred_structure",self.inferred_structure)
        distinct_types = {type(star) for star in self.inferred_structure}
        print(distinct_types)

    def add_grid(self, grid):
        self.grids.append(grid)

    def project_cross_layers(self, epsilon=EPSILON, min_length=CHARACTERISTIC_LENGTH, dz=1.0):
        layer_0 = MultiLineString()
        filtered_shapes = []
        filtered_crosses_per_layer = []

        self.stars = []
        self.filtered_crosses = []
        self.fuzz_masks = []

        for layer_idx, grid in enumerate(self.grids):
            multi_line = MultiLineString(grid.lines)
            mask_buffer = layer_0.buffer(epsilon)

            new_lines = multi_line.difference(mask_buffer)

            if not new_lines.is_empty and new_lines.length >= min_length:
                filtered_shapes.append(new_lines)
                layer_0 = unary_union([layer_0, new_lines])
            else:
                filtered_shapes.append(None)

            kept_crosses = []
            for cross in grid.crosses:
                kept_conns = []
                for direction, line in cross.connections:
                    uncovered = line.difference(mask_buffer)
                    if not uncovered.is_empty and line.length >= min_length:
                        kept_conns.append((direction, line))

                if kept_conns:
                    kept_crosses.append(cross)
                    star = Star(cross, z_coordinate=layer_idx)
                    star.connections = kept_conns
                    star.arms = {d: True for d, _ in kept_conns}
                    self.stars.append(star)

            filtered_crosses_per_layer.append(kept_crosses)
            self.filtered_crosses.extend(kept_crosses)
            self.fuzz_masks.append(layer_0)

        self.filtered_shapes = filtered_shapes
        self.filtered_structure = self.stars

        return

    def align_connection_endpoints_to_top_star(self, grid_size=CHARACTERISTIC_LENGTH, tolerance=CHARACTERISTIC_LENGTH):

        seen = set()
        for star in self.filtered_structure:
            new_connections = []
            for direction, line in star.connections:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                pt1, pt2 = coords[0], coords[-1]
                key = tuple(sorted([(pt1[0], pt1[1]), (pt2[0], pt2[1])]))
                if key not in seen:
                    seen.add(key)
                    new_connections.append((direction, line))
            star.connections = new_connections

        # --- simplify connections to first and last point ---
        for star in self.filtered_structure:
            new_connections = []
            for direction, line in star.connections:
                coords = list(line.coords)
                if len(coords) > 2:
                    coords = [coords[0], coords[-1]]
                new_line = LineString(coords)
                new_connections.append((direction, new_line))
            star.connections = new_connections

        # --- build spatial index ---
        star_positions = [(star.point.x, star.point.y, star) for star in self.filtered_structure]

        # --- shift connection endpoints to top star ---
        for star in self.filtered_structure:
            updated_connections = []
            for direction, line in star.connections:
                new_coords = []
                for x, y, *rest in line.coords:
                    # find all stars in the XY neighborhood of this endpoint
                    neighbors = [
                        s for sx, sy, s in star_positions
                        if abs(sx - x) <= grid_size / 2
                        and abs(sy - y) <= grid_size / 2
                    ]
                    if neighbors:
                        top_star = min(neighbors, key=lambda s: s.point.z)
                        new_coords.append((x, y, top_star.point.z))
                    else:
                        # no neighbor found, keep original z
                        if rest:
                            new_coords.append((x, y, rest[0]))
                        else:
                            new_coords.append((x, y, 0.0))
                updated_connections.append((direction, LineString(new_coords)))
            star.connections = updated_connections

    def get_connections_geojson(self):
        """
        Build filtered_structure connections as GeoJSON and return as dict.
        """
        # Build features
        features = []
        for star in self.filtered_structure:
            for direction, line in star.connections:
                if not line.is_empty:
                    features.append({
                        "type": "Feature",
                        "geometry": mapping(line),
                        "properties": {
                            "direction": direction,
                            "star_z": getattr(star.point, "z", None)
                        }
                    })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return geojson
