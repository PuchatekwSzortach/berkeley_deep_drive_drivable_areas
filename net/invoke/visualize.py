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

    import random

    import cv2
    import tqdm
    import vlogging

    import net.data
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    data_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["validation_images_directory"],
        segmentations_directory=config["validation_segmentations_directory"],
        labels_path=config["validation_labels_directory"]
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    indices = random.choices(
        population=range(len(data_loader)),
        k=4
    )

    for index in tqdm.tqdm(indices):

        image, segmentation = data_loader[index]

        original_resolution = image.shape

        images = [
            image,
            net.processing.get_segmentation_overlay(
                image=image,
                segmentation=segmentation,
                indices_to_colors_map=config["drivable_areas_indices_to_colors_map"]
            )
        ]

        logger.info(
            vlogging.VisualRecord(
                title="deep drive",
                imgs=[cv2.pyrDown(image) for image in images],
                footnotes=str(original_resolution)
            )
        )
