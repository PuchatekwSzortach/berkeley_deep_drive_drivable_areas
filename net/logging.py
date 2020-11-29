"""
Module with logging utilities
"""

import logging
import typing

import cv2
import numpy as np
import tensorflow as tf
import vlogging

import net.data
import net.processing


def log_predictions(
        logger: logging.Logger, prediction_model: tf.keras.Model,
        images: typing.List[np.ndarray], ground_truth_segmentations: typing.List[np.ndarray],
        categories_indices_to_colors_map: typing.Dict[int, typing.Tuple[int, int, int]]):
    """
    Log a batch of predictions, along with input images and ground truth segmentations

    Args:
        logger (logging.Logger): logger instance
        prediction_model (tf.keras.Model): prediction model
        images (typing.List[np.ndarray]): list of images to run predictions on
        ground_truth_segmentations (typing.List[np.ndarray]): list of ground truth segmentations
        categories_to_colors_map (typing.Dict[int, typing.Tuple[int, int, int]]):
        dictionary mapping categories indices to overlay colors to be used in visualizations
    """

    ground_truth_overlay_segmentations = [
        net.processing.get_segmentation_overlay(
            image=image,
            segmentation=segmentation,
            indices_to_colors_map=categories_indices_to_colors_map
        ) for image, segmentation in zip(images, ground_truth_segmentations)
    ]

    logger.info(
        vlogging.VisualRecord(
            title="ground truth segmentations",
            imgs=ground_truth_overlay_segmentations
        )
    )

    predicted_segmentations = np.array(
        [np.argmax(prediction, axis=-1) for prediction in prediction_model.predict(images)])

    predictions_overlays = [
        net.processing.get_segmentation_overlay(
            image=image,
            segmentation=segmentation,
            indices_to_colors_map=categories_indices_to_colors_map
        ) for image, segmentation in zip(images, predicted_segmentations)
    ]

    logger.info(
        vlogging.VisualRecord(
            title="predicted segmentations",
            imgs=predictions_overlays
        )
    )


def get_prediction_overlay(
        prediction_model: tf.keras.Model,
        prediction_resolution: dict,
        categories_indices_to_colors_map: typing.Dict[int, typing.Tuple[int, int, int]],
        image: np.ndarray) -> np.ndarray:
    """
    Given input image compute segmentation prediction for it and return image with segmentation prediction
    overlaid on it

    Args:
        prediction_model (tf.keras.Model): segmentation prediction model
        prediction_model (dict): dictionary specifying image width and height at which predictions are run
        categories_indices_to_colors_map (typing.Dict[int, typing.Tuple[int, int, int]]):
        dictionary mapping categories indices to overlay colors to be used in visualizations
        image (np.ndarray): input image

    Returns:
        np.ndarray: [description]
    """

    resized_image = image.copy()

    while \
        resized_image.shape[0] > prediction_resolution["height"] or \
            resized_image.shape[1] > prediction_resolution["width"]:

        resized_image = cv2.pyrDown(resized_image)

    resized_image = net.processing.pad_to_size(
        image=resized_image,
        target_width=prediction_resolution["width"],
        target_height=prediction_resolution["height"],
        color=(0, 0, 0)
    )

    predicted_segmentation = prediction_model.predict(np.array([resized_image]))[0]

    segmentation_overlay = net.processing.get_segmentation_overlay(
        image=resized_image,
        segmentation=np.argmax(predicted_segmentation, axis=-1),
        indices_to_colors_map=categories_indices_to_colors_map
    )

    return cv2.resize(segmentation_overlay.astype(image.dtype), (image.shape[1], image.shape[0]))
