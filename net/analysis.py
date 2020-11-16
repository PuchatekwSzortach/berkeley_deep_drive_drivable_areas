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
        dict: dictionary with count of samples attribute
    """

    weather_types_counts: dict = collections.defaultdict(int)
    time_of_day_times_counts: dict = collections.defaultdict(int)

    for sample in samples:

        weather_types_counts[sample["attributes"]["weather"]] += 1
        time_of_day_times_counts[sample["attributes"]["timeofday"]] += 1

    return {
        "weather_types_counts": weather_types_counts,
        "times_of_day_times_counts": time_of_day_times_counts
    }
