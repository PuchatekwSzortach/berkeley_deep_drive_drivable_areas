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

    samples_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["training_images_directory"],
        segmentations_directory=config["training_segmentations_directory"],
        labels_path=config["training_labels_directory"]
    )

    data_loader = net.data.TrainingDataLoader(
        samples_data_loader=samples_loader,
        batch_size=4,
        use_training_mode=True
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    indices = random.choices(
        population=range(len(data_loader)),
        k=4
    )

    for index in tqdm.tqdm(indices):

        images, segmentations = data_loader[index]

        overlay_segmentations = [
            net.processing.get_segmentation_overlay(
                image=image,
                segmentation=segmentation,
                indices_to_colors_map=config["drivable_areas_indices_to_colors_map"]
            ) for image, segmentation in zip(images, segmentations)
        ]

        logger.info(
            vlogging.VisualRecord(
                title="images",
                imgs=[cv2.pyrDown(image) for image in images],
            )
        )

        logger.info(
            vlogging.VisualRecord(
                title="segmentations",
                imgs=[cv2.pyrDown(image) for image in overlay_segmentations],
            )
        )
