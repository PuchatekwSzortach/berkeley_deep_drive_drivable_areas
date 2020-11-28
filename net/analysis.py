"""
Module with analysis code
"""

import collections
import typing

import mlflow
import numpy as np
import tensorflow as tf
import tqdm

import net.data


def get_samples_analysis(samples: list) -> dict:
    """
    Given a list of BDD samples, compute count of discriminative properties

    Args:
        samples (list): list of dictionaries with BDD samples

    Returns:
        dict: dictionary of dictionaries with count of samples attribute
    """

    # Create default dictionary of default dictionaries of integers
    # Outer keys will correspond to different types of attributes (weather, timeofday, etc), inner keys
    # will correspond to values of attributes, e.g. rainy, sunny, etc, with their keys representing samples counts
    attributes_statistics: dict = collections.defaultdict(lambda: collections.defaultdict(int))

    attributes = ["weather", "timeofday", "scene"]

    for sample in samples:

        for attribute in attributes:

            attributes_statistics[attribute][sample["attributes"][attribute]] += 1

    return attributes_statistics


class ModelAnalyzer:
    """
    Class for performing model analysis
    """

    def __init__(
            self,
            mlflow_tracking_uri: str,
            prediction_model: tf.keras.Model,
            data_loader: net.data.TrainingDataLoader,
            categories: typing.List[str]) -> None:
        """
        Constructor

        Args:
            mlflow_tracking_uri (str): uri to mlflow server to which results should be sent
            prediction_model (tf.keras.Model): prediction model
            data_loader (net.data.TrainingDataLoader): data loader instance
            categories (typing.List[str]): list of categories
        """

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.prediction_model = prediction_model
        self.data_loader = data_loader
        self.categories = categories

    def analyze_intersection_over_union(self):
        """
        Analyze intersection over union
        """

        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self.data_loader),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([None, None, None, 3]),
                tf.TensorShape([None, None, None])
            )
        ).prefetch(10)

        iterator = iter(dataset)

        categories_intersections_counts = collections.defaultdict(int)
        categories_unions_counts = collections.defaultdict(int)

        for _ in tqdm.tqdm(range(len(self.data_loader))):

            batch_categories_intersections_counts, batch_categories_unions_counts = \
                self._get_iou_results_for_single_batch(iterator)

            for category in self.categories:

                categories_intersections_counts[category] += batch_categories_intersections_counts[category]
                categories_unions_counts[category] += batch_categories_unions_counts[category]

        self._report_iou_results(
            categories_intersections_counts=categories_intersections_counts,
            categories_unions_counts=categories_unions_counts
        )

    def _get_iou_results_for_single_batch(self, iterator):
        """
        Yield one batch from iterator and compute intersections and unions counts on that batch

        Args:
            iterator (Iterable): iterator yielding (images, ground truth segmentations) batches
        Returns:
            [Tuple]: two dictionaries, first with categories: intersections pixels count and second with
            categories: union pixels count data
        """

        images, ground_truth_segmentations = next(iterator)

        # Our iterator returns tensor objects, but we want plain numpy arrays
        ground_truth_segmentations = ground_truth_segmentations.numpy()

        predictions = self.prediction_model.predict(images)
        sparse_predictions = np.argmax(predictions, axis=-1)

        categories_intersections_counts = collections.defaultdict(int)
        categories_unions_counts = collections.defaultdict(int)

        for ground_truth_segmentation, sparse_prediction in zip(ground_truth_segmentations, sparse_predictions):

            for index, category in enumerate(self.categories):

                categories_intersections_counts[category] += \
                    np.sum(np.logical_and(ground_truth_segmentation == index, sparse_prediction == index))

                categories_unions_counts[category] += \
                    np.sum(np.logical_or(ground_truth_segmentation == index, sparse_prediction == index))

        return categories_intersections_counts, categories_unions_counts

    def _report_iou_results(
            self,
            categories_intersections_counts: typing.Dict[str, int],
            categories_unions_counts: typing.Dict[str, int]):
        """
        Format and report intersection over union results

        Args:
            categories_intersections_counts (typing.Dict[int]):
            dictionary mapping categories to intersection pixels counts
            categories_unions_counts (typing.Dict[int]):
            dictionary mapping categories to union pixels counts
        """

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("analysis")

        with mlflow.start_run(run_name="simple_run"):

            categories_intersections_over_unions = {
                category: categories_intersections_counts[category] / categories_unions_counts[category]
                for category in self.categories
            }

            mlflow.log_params(categories_intersections_over_unions)

            print("Intersection over union across categories")
            for category in self.categories:

                print(f"{category}: {categories_intersections_over_unions[category]:.4f}")

            mean_intersection_over_union = np.mean(list(categories_intersections_over_unions.values()))

            print(f"\nMean intersection over union: {mean_intersection_over_union:.4f}")
            mlflow.log_param("mean intersection over union", mean_intersection_over_union)
