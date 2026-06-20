import tensorflow as tf

def hole_valid_loss(y_true, y_pred, mask, hole_weight=6.0, valid_weight=1.0):
    hole_loss = tf.reduce_mean(tf.abs((y_true - y_pred) * mask))
    valid_loss = tf.reduce_mean(tf.abs((y_true - y_pred) * (1 - mask)))
    return hole_weight * hole_loss + valid_weight * valid_loss