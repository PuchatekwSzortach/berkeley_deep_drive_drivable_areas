"""
Module with visualization related tasks
"""

import invoke


@invoke.task
def visualize_data(_context, config_path):
    """
    [summary]

    Args:
        _context ([type]): [description]
        config_path ([type]): [description]
    """

    import vlogging

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["images_directory"],
        segmentations_directory=None,
        labels_path=config["validation_labels_directory"]
    )

    iterator = iter(data_loader)

    logger = net.utilities.get_logger(path="/tmp/log.html")

    for _ in range(3):

        image = next(iterator)

        logger.info(
            vlogging.VisualRecord("deep drive", [image])
        )
