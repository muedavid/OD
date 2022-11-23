import tensorflow as tf


def expand_1d(scalar):
    return tf.expand_dims(tf.expand_dims(tf.expand_dims(scalar, 1), 1), 1)


def coordinate_loss(y_true, y_prediction):
    y_true = tf.cast(y_true, tf.float32)
    y_prediction = tf.cast(y_prediction, tf.float32)
    loss_co = tf.pow(y_true[:, :, 0:2] - y_prediction[:, :, 0:2], 2) * y_true[:, :, 2:3]
    loss_co = tf.reduce_mean(loss_co)

    # one = tf.constant(1.0, dtype=tf.float32)
    # one_sig_out = tf.sigmoid(y_prediction[:, :, 2:3])
    # zero_sig_out = one - one_sig_out
    #
    # loss_lab = -y_true[:, :, 2:3]*tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (one-y_true) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
    #
    # loss_lab = tf.reduce_mean(loss_lab)

    return loss_co


def optical_flow_loss(y_true, y_prediction):
    large = tf.where(tf.abs(y_true - y_prediction) >= 2.0, 1.0, 0.0)
    large = tf.cast(large, tf.float32)

    loss = 2*large * tf.abs(y_true - y_prediction) + 0.5*(1 - large) * tf.math.pow(2 * (y_true - y_prediction), 2)

    return tf.reduce_sum(loss * tf.cast(tf.where(y_true > 0, 1, 0), tf.float32))


def label_loss(y_true, y_prediction):
    y_true = tf.cast(y_true, tf.float32)
    y_prediction = tf.cast(y_prediction, tf.float32)

    large = tf.where(tf.abs(y_true - y_prediction) >= 0.5, 1, 0)
    large = tf.cast(large, tf.float32)

    loss = (large * tf.abs(y_true - y_prediction) + (1 - large) * tf.math.pow(2 * (y_true - y_prediction), 2)) * (
            tf.where(y_true > 0.0, 1.0, 0.0) + tf.where(y_true == 0.0, 0.1, 0.0))

    # loss = 0.001*tf.math.pow(4*(y_true - y_prediction), 4) * (tf.where(y_true > 0.0, 1.0, 0.0) + tf.where(y_true == 0.0, 0.01, 0.0))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))


def focal_loss_binary(y_true, y_prediction):
    y_prediction = tf.cast(y_prediction, tf.float32)

    y_true = tf.where(y_true > 0, 1, 0)
    y_true = tf.cast(y_true, tf.float32)

    one = tf.constant(1.0, dtype=tf.float32)
    one_sig_out = tf.sigmoid(y_prediction)
    zero_sig_out = one - one_sig_out

    loss = - y_true * tf.math.pow(1 - one_sig_out, 2) * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (
            1 - y_true) * tf.math.pow(1 - zero_sig_out, 2) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))

    loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=-1), axis=[1, 2]))

    return loss









