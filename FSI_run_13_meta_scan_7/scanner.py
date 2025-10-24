from init import multiply_correlated_stack, estimator, measure_target_widths, reader
from utilities import clean_images_from_folders, get_stack_around_from_list
from scorer import order_score
import binarizer
from parameters import TEST_FOLDER, MAX_DEPTH, STRICTNESS, PROJ_ON_IMAGE_FOLDER, FROM_IMAGE,TO_IMAGE
from plots import plot_dark_zero_lines_on_image, plot_image_intensity_vs_depth
import numpy as np
from pathlib import Path
import cv2

def main_scanner():
    print("Running depth sweep")

    depth_scores = []
    best_result = None
    best_depth = None
    best_score = -float("inf")

    for depth in range(1, 11):  # try depths 1 to 10
        print(f"\n--- Testing depth {depth} ---")
        result_list = init_scanner(depth)

        widths = measure_target_widths(result_list)
        intensities = [r[1] for r in result_list]
        inferred_parameters = list(zip(widths, intensities))
        corr_stack_list = [r[3] for r in result_list if r[3] is not None]

        all_fixed_points, filtered_images_list, inferred_parameters = binarizer.main_binarizer(
            corr_stack_list, inferred_parameters
        )

        flat_points = np.array([(x, y) for centers in all_fixed_points for x, y, _ in centers])

        if len(flat_points) > 3:
            result = order_score(flat_points, n=4)
            score = result["score"]
        else:
            score = 0.0

        depth_scores.append((depth, score))
        print(f"Depth {depth}: Order score = {score:.3f}")

        if score > best_score:
            best_score = score
            best_depth = depth
            best_result = (all_fixed_points, filtered_images_list, inferred_parameters, corr_stack_list)

    print("\nDepth sweep complete:")
    for d, s in depth_scores:
        print(f"  Depth {d:2d} → Score = {s:.4f}")

    print(f"\n✅ Best depth = {best_depth} (score = {best_score:.3f})")

    if best_result is not None:
        all_fixed_points, filtered_images_list, inferred_parameters, corr_stack_list = best_result
    else:
        all_fixed_points, filtered_images_list, inferred_parameters, corr_stack_list = [], [], [], []

    return all_fixed_points, filtered_images_list, inferred_parameters, corr_stack_list

def apply_scan_sequentially(correlated_stack_list,n_repetitions):
    median_int = init_scanner(correlated_stack_list) 

    for i in range(1,n_repetitions):

        bot_int = median_int - STRICTNESS*i
        top_int = median_int + STRICTNESS*i

        correlated_stack_list = multiply_correlated_stack(image_list=correlated_stack_list,top_intensity=top_int,bottom_intensity=bot_int)
        for i, image in enumerate(correlated_stack_list):
            widths_h, widths_v,diffs,cutoffs_v ,cutoffs_h,median_int, selected_crossings = estimator(image,i)

            plot_dark_zero_lines_on_image(image,h_cutoffs=cutoffs_h,v_cutoffs=cutoffs_v,projection_folder=PROJ_ON_IMAGE_FOLDER,idx=i,dark_percentile=STRICTNESS)

    return 

# def init_scanner( max_depth=MAX_DEPTH):
#
#     clean_images_from_folders([TEST_FOLDER])
#
#     stack_lists = [multiply_correlated_stack(correlation_window=depth) for depth in range(1, max_depth + 1)]
#
#     image_intensities = [[] for _ in stack_lists[0]]
#
#     for i, image in enumerate(stack_lists[0]):
#         for depth_idx, stack_at_depth in enumerate(stack_lists):
#             # pick the image corresponding to index i at this depth
#             image_at_depth = stack_at_depth[i]
#
#             widths_h, widths_v, diffs, cutoffs_v, cutoffs_h, median_int, selected_crossings = estimator(
#                 image_at_depth, i, depth_idx + 1
#             )
#             image_intensities[i].append(median_int)
#             plot_dark_zero_lines_on_image(
#                 image_at_depth,
#                 h_cutoffs=cutoffs_h,
#                 v_cutoffs=cutoffs_v,
#                 projection_folder=PROJ_ON_IMAGE_FOLDER,
#                 idx=i,
#                 dark_percentile=STRICTNESS,
#             )
#
#         intensity_path = TEST_FOLDER / "depths"  
#         plot_image_intensity_vs_depth(image_intensities[i], intensity_path,i)
#
#     return 

from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_image_stack(i, stack_lists, test_depth=None):

    dark_medians = []
    whole_medians = []

    if test_depth is None:
        depth_idx = len(stack_lists) - 1
    else:
        depth_idx = min(test_depth - 1, len(stack_lists) - 1)

    if depth_idx < 0 or depth_idx >= len(stack_lists):
        return (None, None, None, None)

    stack_at_depth = stack_lists[depth_idx]

    if i >= len(stack_at_depth):
        return (None, None, None, None)

    image_at_depth = stack_at_depth[i]

    widths_h, widths_v, diffs, cutoffs_v, cutoffs_h, dark_median, selected_crossings = estimator(
        image_at_depth, i, depth_idx + 1
    )

    whole_mean = np.mean(image_at_depth)
    dark_medians.append(dark_median)
    whole_medians.append(whole_mean)

    plot_dark_zero_lines_on_image(
        image_at_depth,
        h_cutoffs=cutoffs_h,
        v_cutoffs=cutoffs_v,
        projection_folder=PROJ_ON_IMAGE_FOLDER,
        idx=i,
        dark_percentile=STRICTNESS,
    )

    intensity_path = TEST_FOLDER / "depths"
    plot_image_intensity_vs_depth(dark_medians, whole_medians, intensity_path, i)

    return (depth_idx + 1, dark_median, selected_crossings, image_at_depth.copy())

import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def init_scanner(depth):
    print(f"Running a scan for depth={depth} ...")
    start_time = time.perf_counter()

    clean_images_from_folders([TEST_FOLDER])
    image_list, file_list, metadata_list = reader()

    stack_lists = [multiply_correlated_stack(image_list,correlation_window=d) for d in range(1, depth + 1)]

    num_images = len(stack_lists[0])
    print(f"Processing {num_images} images at depth={depth}...")

    with ProcessPoolExecutor() as executor:
        result_list = list(executor.map(
            partial(process_image_stack, stack_lists=stack_lists, test_depth=depth),
            range(num_images)
        ))

    print(f"Depth={depth} completed in {time.perf_counter() - start_time:.2f} sec")
    return result_list
