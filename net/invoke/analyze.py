"""
Module with analysis tasks
"""

import invoke


@invoke.task
def analyze_data(_context, config_path):
    """
    Analyze data

    Args:
        _context (invoke.Context): context object
        config_path (str): path to configuration file
    """

    import icecream

    import net.analysis
    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["training_images_directory"],
        segmentations_directory=config["training_segmentations_directory"],
        labels_path=config["training_labels_directory"]
    )

    training_data_analysis = net.analysis.get_samples_analysis(training_data_loader.samples)

    icecream.ic(training_data_analysis)

    validatation_data_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["validation_images_directory"],
        segmentations_directory=config["validation_segmentations_directory"],
        labels_path=config["validation_labels_directory"]
    )

    validation_data_analysis = net.analysis.get_samples_analysis(validatation_data_loader.samples)

    icecream.ic(validation_data_analysis)
