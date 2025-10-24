from parameters import TEST_FOLDER
import cv2
from init import FILTERS_PARAMETERS, filter_monitor, find_strong_points_in_stack, filter_monitor_manual

def main_binarizer(out_scanner,inferred_parameters):

    print("running binarizer")

    filtered_images_list = []
    for (idx ,image), image_params in zip(enumerate(out_scanner),inferred_parameters):
        filtered_images_list.append(filter_monitor_manual(image,idx, image_params))

    all_fixed_points = []

    for idx, image in enumerate(filtered_images_list):
        fixed_points = find_strong_points_in_stack(filtered_images_list, idx)
        all_fixed_points.append(fixed_points)   

    return all_fixed_points, filtered_images_list,inferred_parameters
