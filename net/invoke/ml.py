"""
Module with machine learning tasks
"""

import invoke


@invoke.task
def train(_context, config_path):
    """
    Task to train drivable areas segmentation model

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["training_images_directory"],
        segmentations_directory=config["training_segmentations_directory"],
        labels_path=config["training_labels_directory"]
    )

    print(training_data_loader)
