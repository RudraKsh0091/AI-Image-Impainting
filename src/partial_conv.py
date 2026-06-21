import tensorflow as tf
from tensorflow import keras
from keras import layers

class PartialConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same', apply_batchnorm=True, activation='leaky_relu'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.apply_batchnorm = apply_batchnorm
        self.activation_type = activation

        self.feature_conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.mask_conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, 
                                       use_bias=False, kernel_initializer='ones', trainable=False)

        if apply_batchnorm:
            self.bn = layers.BatchNormalization()

        if activation == 'leaky_relu':
            self.act = layers.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.act = layers.ReLU()
        else:
            self.act = None

    def call(self, inputs):
        x, mask = inputs

        # Convolve the masked input
        masked_x = x * mask
        feature_output = self.feature_conv(masked_x)

        # Convolve the mask itself (counts valid pixels in each receptive field)
        mask_output = self.mask_conv(mask)

        # Avoid divide-by-zero: clip mask_output to a minimum of 1 where it's 0
        mask_ratio = self.kernel_size * self.kernel_size * mask.shape[-1] / (mask_output + 1e-8)
        mask_ratio = tf.where(mask_output > 0, mask_ratio, tf.zeros_like(mask_ratio))

        # Renormalize feature output
        feature_output = feature_output * mask_ratio

        # Update mask: valid (1) wherever at least one valid pixel contributed
        new_mask = tf.cast(mask_output > 0, tf.float32)

        if self.apply_batchnorm:
            feature_output = self.bn(feature_output)
        if self.act is not None:
            feature_output = self.act(feature_output)

        return feature_output, new_mask