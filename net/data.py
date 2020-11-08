"""
Module with data related code
"""

import json


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

        if sample["attributes"]["scene"] == "highway":

            for label in sample["labels"]:

                if label["category"] == "drivable area":

                    return True

        return False
