"""
Module with machine learning code
"""

import tensorflow as tf


class DeepLabV3PlusBuilder:
    """
    Helper class for building DeepLabV3 model from
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" paper
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.activation = tf.nn.swish

    def get_model(self, categories_count: int) -> tf.keras.Model:
        """
        Model builder functions

        Args:
            categories_count (int): number of categories to predict, including background

        Returns:
            tf.keras.Model: DeepLabV3 model
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(None, None, 3)
        )

        input_op = base_model.input

        decoded_features = self._get_decoder(
            feature_8x=[layer for layer in base_model.layers if layer.name == "conv4_block6_out"][0].output,
            feature_2x=[layer for layer in base_model.layers if layer.name == "conv2_block1_out"][0].output,
            categories_count=categories_count)

        predictions_op = tf.keras.layers.Conv2D(
            filters=categories_count, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=tf.nn.softmax
        )(decoded_features)

        model = tf.keras.Model(
            inputs=input_op,
            outputs=[predictions_op]
        )

        return model

    def _get_decoder(self, feature_8x: tf.Tensor, feature_2x: tf.Tensor, categories_count: int) -> tf.Tensor:
        """
        Get decoder

        Args:
            feature_8x (tf.Tensor): input tensor for which input images is downsampled by 8x
            feature_2x (tf.Tensor): input tensor for which input images is downsampled by 4x
            categories_count (int): number of categories to predict

        Returns:
            tf.Tensor: output op
        """

        x = self._get_atrous_spatial_pooling_pyramid_output(feature_8x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        x = tf.keras.layers.SeparableConv2D(
            filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=self.activation)(x)

        low_level_features = tf.keras.layers.SeparableConv2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=self.activation)(feature_2x)

        low_level_features = tf.keras.layers.SeparableConv2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=self.activation)(low_level_features)

        # Concatenate atrous spacial pooling pyramid features and low level features together
        x = tf.concat([x, low_level_features], axis=-1)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=categories_count, kernel_size=(1, 1), strides=(1, 1), padding="same",
            dilation_rate=(1, 1), activation=None)(x)

        x = tf.image.resize(
            images=x,
            size=(4 * tf.shape(x)[1], 4 * tf.shape(x)[2]),
            method=tf.image.ResizeMethod.BILINEAR
        )

        return x

    def _get_atrous_spatial_pooling_pyramid_output(self, input_op: tf.Tensor) -> tf.Tensor:
        """
        Get atrous spatial pooling pyramid output

        Args:
            input_op (tf.Tensor): input tensor

        Returns:
            tf.Tensor: output op
        """

        dilation_rates = [1, 6, 12, 18, 24]
        outputs = []

        for dilation_rate in dilation_rates:

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                dilation_rate=(dilation_rate, dilation_rate), activation=self.activation)(input_op)

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                dilation_rate=(1, 1), activation=self.activation)(x)

            x = tf.keras.layers.SeparableConv2D(
                filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                dilation_rate=(1, 1), activation=self.activation)(x)

            outputs.append(x)

        x = tf.concat(outputs, axis=-1)

        return tf.keras.layers.BatchNormalization()(x)


def get_temperature_scaled_sparse_softmax(labels: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
    """
    Get a temperature scaled sparse softmax loss

    Args:
        labels (tf.Tensor): batch of labels
        predictions (tf.Tensor): batch of predictions

    Returns:
        tf.Tensor: loss tensor
    """

    temperature = 0.5

    # Compute temperature scaled logits
    logits = predictions.op.inputs[0]
    scaled_logits = logits / temperature

    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, scaled_logits, from_logits=True
    )
