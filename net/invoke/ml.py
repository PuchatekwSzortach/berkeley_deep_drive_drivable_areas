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

    import icecream
    import tensorflow as tf

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_samples_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["training_images_directory"],
        segmentations_directory=config["training_segmentations_directory"],
        labels_path=config["training_labels_directory"]
    )

    training_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=training_samples_loader,
        batch_size=config["batch_size"],
        use_training_mode=True
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(training_data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([
                None,
                config["training_image_dimensions"]["height"],
                config["training_image_dimensions"]["width"],
                3
            ]),
            tf.TensorShape([
                None,
                config["training_image_dimensions"]["height"],
                config["training_image_dimensions"]["width"]
            ])
        )
    ).prefetch(32)

    training_data = list(training_dataset.take(2))

    icecream.ic(training_data[0][0].shape)
    icecream.ic(training_data[0][1].shape)

    validatation_samples_loader = net.data.BDDSamplesDataLoader(
        images_directory=config["validation_images_directory"],
        segmentations_directory=config["validation_segmentations_directory"],
        labels_path=config["validation_labels_directory"]
    )

    validation_data_loader = net.data.TrainingDataLoader(
        samples_data_loader=validatation_samples_loader,
        batch_size=config["batch_size"],
        use_training_mode=False
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([
                None,
                config["training_image_dimensions"]["height"],
                config["training_image_dimensions"]["width"],
                3
            ]),
            tf.TensorShape([
                None,
                config["training_image_dimensions"]["height"],
                config["training_image_dimensions"]["width"]
            ])
        )
    ).prefetch(32)

    validation_data = list(validation_dataset.take(2))

    icecream.ic(validation_data[0][0].shape)
    icecream.ic(validation_data[0][1].shape)
