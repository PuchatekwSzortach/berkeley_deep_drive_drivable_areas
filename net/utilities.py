"""
Module with utilities
"""

import yaml


def read_yaml(path: str):
    """Read content of yaml file from path

    :param path: path to yaml file
    :type path: str
    :return: yaml file content, usually a dictionary
    """

    with open(path) as file:

        return yaml.safe_load(file)
