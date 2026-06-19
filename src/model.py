import tensorflow as tf
from tensorflow import keras
from keras import layers

def encoder_block(x, filters, apply_batchnorm=True):
    x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x) if apply_batchnorm else x
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def decoder_block(x, skip, filters, apply_dropout=False):
    x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.5)(x) if apply_dropout else x
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip])
    return x

def build_unet(img_size=256):
    # Input: (256, 256, 4)   — masked_image (3 channels) + mask (1 channel) concatenated
    masked_image = layers.Input(shape = (img_size, img_size, 3))
    mask = layers.Input(shape = (img_size, img_size, 1))
    combined_input = layers.Concatenate()([masked_image, mask])
    
    # Encoder (save each output for skip connections):
    e1 = encoder_block(combined_input, 64, apply_batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)

    # Bottleneck:
    b  = encoder_block(e5, 512)

    # Decoder (skip connections go in reverse order — e5, e4, e3, e2, e1):
    d1 = decoder_block(b,  e5, 512, apply_dropout=True)
    d2 = decoder_block(d1, e4, 512, apply_dropout=True)
    d3 = decoder_block(d2, e3, 256)
    d4 = decoder_block(d3, e2, 128)
    d5 = decoder_block(d4, e1, 64)

    # Output:
    output = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(d5)
    
    return keras.Model(inputs=[masked_image, mask], outputs=output)