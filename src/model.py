import tensorflow as tf
from tensorflow import keras
from keras import layers
from src import losses

class InpaintingModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        (masked_image, mask), image = data
        with tf.GradientTape() as tape:
            predictions = self((masked_image, mask), training=True)
            loss = losses.hole_valid_loss(image, predictions, mask)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (masked_image, mask), image = data
        predictions = self((masked_image, mask), training=False)
        loss = losses.hole_valid_loss(image, predictions, mask)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

def encoder_block(x, filters, apply_batchnorm=True):
    x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x) if apply_batchnorm else x
    x = layers.LeakyReLU(negative_slope=0.2)(x)
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
    
    return InpaintingModel(inputs=[masked_image, mask], outputs=output)

from src.partial_conv import PartialConv2D

def pconv_decoder_block(x, mask, skip_x, skip_mask, filters, apply_dropout=False):
    x = layers.UpSampling2D(size=2)(x)
    mask = layers.UpSampling2D(size=2)(mask)
    x = layers.Concatenate()([x, skip_x])
    mask = layers.Concatenate()([mask, skip_mask])

    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if apply_dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return x, mask

def build_pconv_unet(img_size=256):
    masked_image = layers.Input(shape=(img_size, img_size, 3))
    mask_3ch = layers.Input(shape=(img_size, img_size, 1))
    mask_input = layers.Concatenate()([mask_3ch, mask_3ch, mask_3ch])  # broadcast to 3 channels for image conv

    pc1 = PartialConv2D(64, apply_batchnorm=False)
    e1, m1 = pc1([masked_image, mask_input])

    pc2 = PartialConv2D(128)
    e2, m2 = pc2([e1, m1])

    pc3 = PartialConv2D(256)
    e3, m3 = pc3([e2, m2])

    pc4 = PartialConv2D(512)
    e4, m4 = pc4([e3, m3])

    pc5 = PartialConv2D(512)
    e5, m5 = pc5([e4, m4])

    pc6 = PartialConv2D(512)
    b, mb = pc6([e5, m5])

    d1, dm1 = pconv_decoder_block(b, mb, e5, m5, 512, apply_dropout=True)
    d2, dm2 = pconv_decoder_block(d1, dm1, e4, m4, 512, apply_dropout=True)
    d3, dm3 = pconv_decoder_block(d2, dm2, e3, m3, 256)
    d4, dm4 = pconv_decoder_block(d3, dm3, e2, m2, 128)
    d5, dm5 = pconv_decoder_block(d4, dm4, e1, m1, 64)

    d6 = layers.UpSampling2D(size=2)(d5)
    output = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(d6)

    return InpaintingModel(inputs=[masked_image, mask_3ch], outputs=output)