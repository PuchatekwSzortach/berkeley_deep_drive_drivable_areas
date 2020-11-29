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
        target_image_dimensions=config["training_image_dimensions"],
        use_training_mode=True,
        augmentations_pipeline=net.processing.get_augmentation_pipepline()
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


@invoke.task
def visualize_predictions(_context, config_path):
    """
    Visualize model predictions on a few images

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import random

    import tensorflow as tf
    import tqdm

    import net.data
    import net.logging
    import net.ml
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    samples_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["validation_images_directory"],
        segmentations_directory=config["validation_segmentations_directory"],
        labels_path=config["validation_labels_directory"]
    )

    data_loader = net.data.TrainingDataLoader(
        samples_data_loader=samples_loader,
        batch_size=4,
        target_image_dimensions=config["training_image_dimensions"],
        use_training_mode=False,
        augmentations_pipeline=None
    )

    logger = net.utilities.get_logger(path="/tmp/log.html")

    prediction_model = tf.keras.models.load_model(
        filepath=config["current_model_directory"],
        custom_objects={
            "get_temperature_scaled_sparse_softmax": net.ml.get_temperature_scaled_sparse_softmax
        }
    )

    batches_indices = random.choices(
        population=range(len(data_loader)),
        k=4
    )

    for batch_index in tqdm.tqdm(batches_indices):

        images, segmentations = data_loader[batch_index]

        net.logging.log_predictions(
            logger=logger,
            prediction_model=prediction_model,
            images=images,
            ground_truth_segmentations=segmentations,
            categories_indices_to_colors_map=config["drivable_areas_indices_to_colors_map"]
        )


@invoke.task
def visualize_predictions_on_movie(_context, config_path):
    """
    Visualize model predictions on a movie

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import cv2
    import functools

    import moviepy.editor
    import tensorflow as tf

    import net.logging
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    prediction_model = tf.keras.models.load_model(
        filepath=config["current_model_directory"],
        custom_objects={
            "get_temperature_scaled_sparse_softmax": net.ml.get_temperature_scaled_sparse_softmax
        }
    )

    input_clip = moviepy.editor.VideoFileClip(config["drive_recoder_input_movie_path"])

    prediction_overlay_partial = functools.partial(
        net.logging.get_prediction_overlay,
        prediction_model,
        config["training_image_dimensions"],
        config["drivable_areas_indices_to_colors_map"]
    )

    def image_transformer(image):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = prediction_overlay_partial(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_clip = input_clip.fl_image(image_transformer)
    output_clip.write_videofile(config["drive_recoder_output_movie_path"])
