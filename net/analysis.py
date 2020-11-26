"""
Module with analysis code
"""

import collections


def get_samples_analysis(samples: list) -> dict:
    """
    Given a list of BDD samples, compute count of discriminative properties

    Args:
        samples (list): list of dictionaries with BDD samples

    Returns:
        dict: dictionary of dictionaries with count of samples attribute
    """

    # Create default dictionary of default dictionaries of integers
    # Outer keys will correspond to different types of attributes (weather, timeofday, etc), inner keys
    # will correspond to values of attributes, e.g. rainy, sunny, etc, with their keys representing samples counts
    attributes_statistics: dict = collections.defaultdict(lambda: collections.defaultdict(int))

    attributes = ["weather", "timeofday", "scene"]

    for sample in samples:

        for attribute in attributes:

            attributes_statistics[attribute][sample["attributes"][attribute]] += 1

    return attributes_statistics
