"""
Module with processing functionality
"""

import cv2
import numpy as np


def get_segmentation_overlay(image, segmentation, indices_to_colors_map):
    """
    Given image and segmentation map, create image with segmentation drawn over it
    """

    overlay_image = image.copy()

    for category_index, color in indices_to_colors_map.items():

        overlay_image[segmentation == category_index, :] = color

    return cv2.addWeighted(image, 0.6, overlay_image, 0.4, 0)


def pad_to_size(
        image: np.ndarray, target_width: int, target_height: int, color: tuple) -> np.ndarray:
    """
    Given an image center-pad with selecter color to given size in both dimensions.

    Args:
        image (np.ndarray): 3D numpy array
        size (int): size to which image should be padded in both directions
        target_width (int): width image should be padded
        target_width (int): height image should be padded
        color (tuple): color that should be used for padding

    Returns:
        np.array: padded images
    """

    # Compute paddings
    total_vertical_padding = target_height - image.shape[0]

    upper_padding = total_vertical_padding // 2
    lower_padding = total_vertical_padding - upper_padding

    total_horizontal_padding = target_width - image.shape[1]

    left_padding = total_horizontal_padding // 2
    right_padding = total_horizontal_padding - left_padding

    if len(image.shape) == 3:

        # Create canvas with desired shape and background image, paste image on top of it
        canvas = np.ones(shape=(target_height, target_width, 3)) * color
        canvas[upper_padding:target_height - lower_padding, left_padding:target_width - right_padding, :] = image

        # Return canvas
        return canvas

    if len(image.shape) == 2:

        # Create canvas with desired shape and background image, paste image on top of it
        canvas = np.ones(shape=(target_height, target_width)) * color
        canvas[upper_padding:target_height - lower_padding, left_padding:target_width - right_padding] = image

        # Return canvas
        return canvas

    raise ValueError("Invalid image shape")
