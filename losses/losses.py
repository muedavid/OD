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


def weighted_multi_label_sigmoid_loss(y_true, y_prediction, beta_lower=0.005, beta_upper=0.995, class_weighted=False):
    dtype = tf.float32

    beta_lower = tf.cast(beta_lower, dtype=dtype)
    beta_upper = tf.cast(beta_upper, dtype=dtype)

    # Transform y_true to Tensor
    range_reshape = tf.reshape(tf.range(1, y_prediction.shape[-1] + 1), [1, 1, 1, y_prediction.shape[-1]])
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = tf.cast(range_reshape == y_true, dtype=tf.float32)

    # Compute Beta
    if class_weighted:
        num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
    else:
        num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
    num_edge_pixel = tf.cast(num_edge_pixel, dtype=dtype)
    num_pixel = y_true.shape[1] * y_true.shape[2]
    num_pixel = tf.cast(num_pixel, dtype=dtype)
    num_non_edge_pixel = num_pixel - num_edge_pixel
    num_non_edge_pixel = tf.cast(num_non_edge_pixel, dtype=dtype)
    beta_tensor = num_non_edge_pixel / num_pixel
    beta_tensor = tf.clip_by_value(beta_tensor, beta_lower, beta_upper)

    # Loss
    beta_tensor = tf.cast(beta_tensor, dtype)
    y_true = tf.cast(y_true, dtype)
    y_prediction = tf.cast(y_prediction, dtype)
    one = tf.constant(1.0, dtype=dtype)
    one_sig_out = tf.sigmoid(y_prediction)
    zero_sig_out = one - one_sig_out

    loss = -beta_tensor * y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - beta_tensor) * (
            1 - y_true) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
    return loss


def weighted_multi_label_sigmoid_loss_check(y_true, y_prediction):
    # Transform y_true to Tensor
    y_true = tf.cast(y_true, dtype=tf.int32)
    range_reshape = tf.reshape(tf.range(1, y_prediction.shape[-1] + 1), [1, 1, 1, y_prediction.shape[-1]])
    y_true = tf.cast(range_reshape == y_true, dtype=tf.float32)

    # compute weights
    num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
    num_pixel = y_true.shape[1] * y_true.shape[2]
    num_non_edge_pixel = num_pixel - num_edge_pixel
    weight = tf.clip_by_value(num_non_edge_pixel / num_edge_pixel, 0, num_pixel)

    # To get the same result as above: loss has been devided by pos and multiplied by tot:
    # normalization term, if numEdgePixel = 0: loss = 0
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_prediction, weight) * num_edge_pixel / num_pixel
    loss = tf.reduce_mean(loss)
    return loss


def focal_loss_edges(y_true, y_prediction, gamma=2, weighted_beta=True, beta_lower=0.005, beta_upper=0.995):
    dtype = tf.float32
    gamma = tf.cast(gamma, dtype=dtype)

    # Transform y_true to Tensor
    range_reshape = tf.reshape(tf.range(1, y_prediction.shape[-1] + 1), [1, 1, 1, y_prediction.shape[-1]])
    range_reshape = tf.cast(range_reshape, dtype=dtype)
    y_true = tf.cast(y_true, dtype=dtype)
    y_true = tf.cast(range_reshape == y_true, dtype=dtype)

    one = tf.constant(1.0, dtype=tf.float32)
    one_sig_out = tf.sigmoid(y_prediction)
    zero_sig_out = one - one_sig_out

    if weighted_beta:
        # Compute Beta
        num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
        num_pixel = y_true.shape[1] * y_true.shape[2]
        num_non_edge_pixel = num_pixel - num_edge_pixel
        beta_tensor = num_non_edge_pixel / num_pixel
        beta_tensor = tf.clip_by_value(beta_tensor, beta_lower, beta_upper)
        beta_tensor = tf.cast(beta_tensor, dtype)

        loss = - beta_tensor * y_true * tf.math.pow(1 - one_sig_out, gamma) * tf.math.log(
            tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - beta_tensor) * (1 - y_true) * tf.math.pow(
            1 - zero_sig_out, gamma) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
    else:
        loss = - y_true * tf.math.pow(1 - one_sig_out, gamma) * tf.math.log(
            tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - y_true) * tf.math.pow(1 - zero_sig_out,
                                                                                     gamma) * tf.math.log(
            tf.clip_by_value(zero_sig_out, 1e-10, 1000))

    loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=-1), axis=[1, 2]))

    return loss
