"""
Module with data related code
"""

import json
import os
import random
import typing

import cv2
import imgaug
import numpy as np

import net.processing


class BDDSamplesDataLoader:
    """
    Class for loading Berkeley Deep Drive driveable areas samples
    """

    def __init__(self, images_directory: str, segmentations_directory: str, labels_path: str) -> None:
        """
        [summary]

        Args:
            images_directory (str): path to dictionary with images
            segmentations_directory (str): path to directory with driveable areas segmentations
            labels_path (str): path to json file with labels data
        """

        self.images_directory = images_directory
        self.segmentations_directory = segmentations_directory

        with open(labels_path) as file:

            all_samples = json.load(file)

        self.samples = [sample for sample in all_samples if self._is_target_sample(sample) is True]

    def _is_target_sample(self, sample: dict) -> bool:
        """
        Check if sample fulfills our criteria for target sample

        Args:
            sample (dict): sample to examine

        Returns:
            bool: True if sample is considered target sample, False otherwise
        """

        # if sample["attributes"]["scene"] == "highway":
        if sample["attributes"]["scene"] in ["residential", "city street"]:

            for label in sample["labels"]:

                if label["category"] == "drivable area":

                    return True

        return False

    def __len__(self):

        return len(self.samples)

    def __iter__(self):

        for index in range(len(self)):

            yield self[index]

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = os.path.join(
            self.images_directory, sample["name"]
        )

        segmentation_path = os.path.join(
            self.segmentations_directory, os.path.splitext(sample["name"])[0] + "_drivable_id.png"
        )

        # Only return first channel of segmentation image
        return \
            cv2.pyrDown(cv2.imread(image_path)), \
            cv2.pyrDown(cv2.imread(segmentation_path)[:, :, 0]).astype(np.int32)


class TrainingDataLoader:
    """
    Data loader that yields batches (images, segmentations) suitable for training segmentation model
    """

    def __init__(
            self, samples_data_loader: BDDSamplesDataLoader,
            batch_size: int,
            target_image_dimensions: dict,
            use_training_mode: bool,
            augmentations_pipeline: imgaug.augmenters.Augmenter) -> None:
        """
        Constructor

        Args:
            samples_data_loader (BDDSamplesDataLoader): samples data loader
            batch_size (int): number of samples each yield should contain
            target_image_dimensions (dict): dictionary specifying with and height images samples should have
            use_training_mode (bool): if True, then samples are shuffled
            augmentations_pipeline (imgaug.augmenters.Augmenter): augmentation pipeline, optional.
            Used only if use_training_mode is set to True
        """

        self.samples_data_loader = samples_data_loader
        self.batch_size = batch_size
        self.target_image_dimensions = target_image_dimensions
        self.use_training_mode = use_training_mode

        self.samples_indices = list(range(len(self.samples_data_loader)))

        if self.use_training_mode is True:
            random.shuffle(self.samples_indices)

        self.augmentations_pipeline = augmentations_pipeline

    def __len__(self) -> int:

        return len(self.samples_data_loader) // self.batch_size

    def __getitem__(self, index) -> typing.Tuple[np.ndarray, np.ndarray]:

        # Get batch size numberr of samples indices
        samples_batch_indices = self.samples_indices[index * self.batch_size:(index + 1) * self.batch_size]

        images: list = []
        segmentations: list = []

        for sample_index in samples_batch_indices:

            image, segmentation = self.samples_data_loader[sample_index]

            images.append(image)
            segmentations.append(segmentation)

        return self._process_batch(images, segmentations)

    def __iter__(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Iterator, yields tuples (images, segmentations)
        """

        while True:

            for batch_index in range(len(self)):

                yield self[batch_index]

            if self.use_training_mode is True:
                random.shuffle(self.samples_indices)

    def _process_batch(self, images: np.ndarray, segmentations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Process batch into format suitable for training

        Args:
            images (np.ndarray): batch of images
            segmentations (np.ndarray): batch of segmentations
        """

        # Pad images and segmentations to a fixed size
        processed_images = []
        processed_segmentations = []

        for image, segmentation in zip(images, segmentations):

            processed_images.append(
                net.processing.pad_to_size(
                    image=image,
                    target_width=self.target_image_dimensions["width"],
                    target_height=self.target_image_dimensions["height"],
                    color=(0, 0, 0)
                )
            )

            processed_segmentations.append(
                net.processing.pad_to_size(
                    image=segmentation,
                    target_width=self.target_image_dimensions["width"],
                    target_height=self.target_image_dimensions["height"],
                    color=(0,)
                )
            )

        processed_images_array = np.array(processed_images, np.uint8)
        processed_segmentations_array = np.array(processed_segmentations, np.int32)

        if self.use_training_mode is True:

            # imgaug expect slightly different format and data types than rest of code, so need to
            # transform data to expected format for augmentations, then back
            processed_images_array, processed_segmentations_array = self.augmentations_pipeline(
                images=processed_images_array.astype(np.int32),
                segmentation_maps=np.expand_dims(processed_segmentations_array, -1)
            )

            processed_images_array = processed_images_array.astype(np.uint8)
            processed_segmentations_array = np.squeeze(processed_segmentations_array)

        return processed_images_array, processed_segmentations_array
