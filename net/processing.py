"""
Module with processing functionality
"""

import cv2


def get_segmentation_overlay(image, segmentation, indices_to_colors_map):
    """
    Given image and segmentation map, create image with segmentation drawn over it
    """

    overlay_image = image.copy()

    for category_index, color in indices_to_colors_map.items():

        overlay_image[segmentation == category_index, :] = color

    return cv2.addWeighted(image, 0.6, overlay_image, 0.4, 0)
