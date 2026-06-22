import tensorflow as tf
from tensorflow import keras

def hole_valid_loss(y_true, y_pred, mask, hole_weight=6.0, valid_weight=1.0):
    hole_loss = tf.reduce_mean(tf.abs((y_true - y_pred) * mask))
    valid_loss = tf.reduce_mean(tf.abs((y_true - y_pred) * (1 - mask)))
    return hole_weight * hole_loss + valid_weight * valid_loss

def preprocess_for_vgg(image):
    image = (image + 1.0) * 127.5
    image = tf.clip_by_value(image, 0, 255)
    return keras.applications.vgg16.preprocess_input(image)

def build_vgg_feature_extractor():
    vgg = keras.applications.VGG16(weights='imagenet', include_top=False)
    vgg.trainable = False
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv2']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return keras.Model(inputs=vgg.input, outputs=outputs)

def perceptual_loss(y_true, y_pred, vgg_extractor):
    true_preprocessed = preprocess_for_vgg(y_true)
    pred_preprocessed = preprocess_for_vgg(y_pred)
    
    true_features = vgg_extractor(true_preprocessed, training=False)
    pred_features = vgg_extractor(pred_preprocessed, training=False)
    
    loss = 0.0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.abs(true_feat - pred_feat))
    
    return loss

def gram_matrix(features):
    # features shape: (batch, h, w, channels)
    batch = tf.shape(features)[0]
    h = tf.shape(features)[1]
    w = tf.shape(features)[2]
    c = tf.shape(features)[3]
    
    # reshape to (batch, h*w, channels)
    features = tf.reshape(features, [batch, h * w, c])
    # gram matrix: (batch, channels, channels)
    gram = tf.matmul(features, features, transpose_a=True)
    # normalize by spatial size
    gram = gram / tf.cast(h * w * c, tf.float32)
    return gram

def style_loss(y_true, y_pred, vgg_extractor):
    true_preprocessed = preprocess_for_vgg(y_true)
    pred_preprocessed = preprocess_for_vgg(y_pred)
    
    true_features = vgg_extractor(true_preprocessed, training=False)
    pred_features = vgg_extractor(pred_preprocessed, training=False)
    
    loss = 0.0
    for true_feat, pred_feat in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.abs(gram_matrix(true_feat) - gram_matrix(pred_feat)))
    
    return loss

def total_variation_loss(y_pred):
    return tf.reduce_mean(tf.image.total_variation(y_pred))

def combined_loss(y_true, y_pred, mask, vgg_extractor, hole_weight=6.0, valid_weight=1.0, perceptual_weight=0.05, style_weight=120.0, tv_weight=0.1):
    
    h_v_loss = hole_valid_loss(y_true, y_pred, mask, hole_weight, valid_weight)
    p_loss = perceptual_loss(y_true, y_pred, vgg_extractor)
    s_loss = style_loss(y_true, y_pred, vgg_extractor)
    tv_loss = total_variation_loss(y_pred)
    
    return h_v_loss + perceptual_weight * p_loss + style_weight * s_loss + tv_weight * tv_loss