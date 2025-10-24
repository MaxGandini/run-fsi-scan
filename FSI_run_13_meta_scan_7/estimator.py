import matplotlib as plt

import numpy as np 
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, Polygon
from typing import List, Tuple, Union
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import imageio
from matplotlib import animation
import plotly.graph_objects as go
import webbrowser
import math

from ploter_writer import directional_projection_gray, plot_threshold_finder, scatter_points_from_xy_lists_combinations, scatter_points_and_track_intensity_stack 

from parameters import (    
    TEST_FOLDER
)

def estimator(image, idx):
    thresholds_h = []
    thresholds_v = []
    black_thresholds = []
    widths_h=[]
    widths_v=[]
    windows=[]

    PROJECTION_FOLDER = TEST_FOLDER / "projection_folder"

    threshold_h,threshold_v,h_derivative,v_derivative = directional_projection_gray(image,4,4,PROJECTION_FOLDER,image_name = f"Projected_{idx}")

    thresholds_h.append(threshold_h)
    thresholds_v.append(threshold_v)
    median_threshold_h = np.median(thresholds_h)
    median_threshold_v = np.median(thresholds_v)

    count_pos_h,count_neg_h, index_pos_h,index_neg_h = count_crossings_index(h_derivative,median_threshold_h)
    count_pos_v,count_neg_v, index_pos_v,index_neg_v = count_crossings_index(v_derivative,median_threshold_v)

    target_width_h, cuts_h = measure_targets(count_pos_h,count_neg_h,index_pos_h,index_neg_h)
    target_width_v, cuts_v = measure_targets(count_pos_v,count_neg_v,index_pos_v,index_neg_v)


    widths_h.append(target_width_h)
    widths_v.append(target_width_v)

    plot_threshold_finder(median_threshold_v,v_derivative,PROJECTION_FOLDER / "vertical_proj_gray_derivative" ,threshold_v)
    plot_threshold_finder(median_threshold_h,h_derivative,PROJECTION_FOLDER/ "horizontal_proj_gray_derivative",threshold_h)
    return


# result = {"threshold": black_thresholds, "cuts_h": cuts_h ,"cuts_v": cuts_v, "most_targets": most_targets_path, "target_width_h":target_width_h, "target_width_v":target_width_v,"correlation_window":windows}
    

def directional_projection_gray(image, x_line: int, y_line: int, projection_folder, image_name: str = "projection"):
    """
    Computes and saves horizontal and vertical projections and their derivatives
    using grayscale intensity.

    Parameters:
        image_path (str or Path): Path to the input image.
        x_line (int): Column index for vertical reference line.
        y_line (int): Row index for horizontal reference line.
        projection_folder (Path): Directory where results will be saved.
        image_name (str): Base name for saved plots.
    """
    if image is None:
        raise FileNotFoundError(f"Empty image")

    gray_image = image.astype(np.float64)

    h_ref_line = gray_image[y_line, :]
    v_ref_line = gray_image[:, x_line]

    h_profile = np.array([
        np.sum(gray_image[r, :] * h_ref_line) for r in range(gray_image.shape[0])
    ])
    v_profile = np.array([
        np.sum(gray_image[:, c] * v_ref_line) for c in range(gray_image.shape[1])
    ])

    # --- Plot Projections ---
    (projection_folder / "horizontal_proj_gray").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(h_profile)), h_profile, color='black')
    plt.title(f"Horizontal Projection (Y={y_line})")
    plt.xlabel("Y coordinate")
    plt.ylabel("Weighted Sum (Gray)")
    plt.tight_layout()
    plt.savefig(projection_folder / "horizontal_proj_gray" / f"{image_name}_horizontal.png")
    plt.close()

    (projection_folder / "vertical_proj_gray").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(v_profile)), v_profile, color='black')
    plt.title(f"Vertical Projection (X={x_line})")
    plt.xlabel("X coordinate")
    plt.ylabel("Weighted Sum (Gray)")
    plt.tight_layout()
    plt.savefig(projection_folder / "vertical_proj_gray" / f"{image_name}_vertical.png")
    plt.close()

    # --- Derivatives ---
    h_derivative_ = np.diff(h_profile)
    v_derivative_ = np.diff(v_profile) 

    h_derivative = h_derivative_ / np.max(np.abs(h_derivative_))
    v_derivative = v_derivative_ / np.max(np.abs(v_derivative_))

    threshold_h = find_peak_threshold(h_derivative)
    (projection_folder / "horizontal_proj_gray_derivative").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hlines(threshold_h,0,len(h_derivative))
    plt.hlines(-threshold_h,0,len(h_derivative))
    plt.plot(np.arange(len(h_derivative)), h_derivative, color='gray')
    plt.title(f"Horizontal Derivative Projection (Y={y_line})")
    plt.xlabel("Y coordinate")
    plt.ylabel("d/dy (Gray Projection)")
    plt.tight_layout()
    plt.savefig(projection_folder / "horizontal_proj_gray_derivative" / f"{image_name}_horizontal_deriv.png")
    plt.close()

    (projection_folder / "vertical_proj_gray_derivative").mkdir(parents=True, exist_ok=True)

    threshold_v = find_peak_threshold(v_derivative)
    plt.figure(figsize=(8, 4))
    plt.hlines(threshold_v,0,len(v_derivative))
    plt.hlines(threshold_v,0,len(v_derivative))
    plt.plot(np.arange(len(v_derivative)), v_derivative, color='gray')
    plt.title(f"Vertical Derivative Projection (X={x_line})")
    plt.xlabel("X coordinate")
    plt.ylabel("d/dx (Gray Projection)")
    plt.tight_layout()
    plt.savefig(projection_folder / "vertical_proj_gray_derivative" / f"{image_name}_vertical_deriv.png")
    plt.close()

    return threshold_h , threshold_v , h_derivative, v_derivative

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
