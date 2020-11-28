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


@invoke.task
def analyze_predictions(_context, config_path):
    """
    Analyze model's performance

    Args:
        _context (invoke.Context): context instance
        config_path (str): path configuration file
    """
    import tensorflow as tf

    import net.analysis
    import net.data
    import net.ml
    import net.logging
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

    prediction_model = tf.keras.models.load_model(
        filepath=config["current_model_directory"],
        custom_objects={
            "get_temperature_scaled_sparse_softmax": net.ml.get_temperature_scaled_sparse_softmax
        }
    )

    net.analysis.ModelAnalyzer(
        mlflow_tracking_uri=config["mlflow_tracking_uri"],
        prediction_model=prediction_model,
        data_loader=data_loader,
        categories=config["categories"]
    ).analyze_intersection_over_union()
