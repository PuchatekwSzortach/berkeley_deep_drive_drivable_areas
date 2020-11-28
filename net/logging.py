"""
Module with logging utilities
"""

import logging
import typing

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
