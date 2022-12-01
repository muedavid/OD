import tensorflow as tf


class WeightedMultiLabelSigmoidLoss(tf.keras.losses.Loss):
    def __init__(self, min_edge_loss_weighting=0.005, max_edge_loss_weighting=0.995, class_individually_weighted=False,
                 name='weighted_multi_label_sigmoid_loss'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.class_individually_weighted = class_individually_weighted
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        
        min_edge_loss_weighting = tf.cast(self.min_edge_loss_weighting, dtype=dtype)
        max_edge_loss_weighting = tf.cast(self.max_edge_loss_weighting, dtype=dtype)
        
        # Transform y_true to Tensor
        range_reshape = tf.reshape(tf.range(1, y_pred.shape[-1] + 1), [1, 1, 1, y_pred.shape[-1]])
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.cast(range_reshape == y_true, dtype=tf.float32)
        
        # Compute Beta
        if self.class_individually_weighted:
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
        else:
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
        num_edge_pixel = tf.cast(num_edge_pixel, dtype=dtype)
        num_pixel = y_true.shape[1] * y_true.shape[2]
        num_pixel = tf.cast(num_pixel, dtype=dtype)
        num_non_edge_pixel = num_pixel - num_edge_pixel
        num_non_edge_pixel = tf.cast(num_non_edge_pixel, dtype=dtype)
        edge_loss_weighting = num_non_edge_pixel / num_pixel
        edge_loss_weighting = tf.clip_by_value(edge_loss_weighting, min_edge_loss_weighting, max_edge_loss_weighting)
        
        # Loss
        edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
        y_true = tf.cast(y_true, dtype)
        y_prediction = tf.cast(y_pred, dtype)
        one = tf.constant(1.0, dtype=dtype)
        one_sig_out = tf.sigmoid(y_prediction)
        zero_sig_out = one - one_sig_out
        
        loss = -edge_loss_weighting * y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (
                    1 - edge_loss_weighting) * (
                       1 - y_true) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        return loss


class FocalLossEdges(tf.keras.losses.Loss):
    def __init__(self, power=2, edge_loss_weighting=True, min_edge_loss_weighting=0.005, max_edge_loss_weighting=0.995,
                 name='focal_loss_edges'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.edge_loss_weighting = edge_loss_weighting
        self.power = power
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        power = tf.cast(self.power, dtype=dtype)
        
        # Transform y_true to Tensor
        range_reshape = tf.reshape(tf.range(1, y_pred.shape[-1] + 1), [1, 1, 1, y_pred.shape[-1]])
        range_reshape = tf.cast(range_reshape, dtype=dtype)
        y_true = tf.cast(y_true, dtype=dtype)
        y_true = tf.cast(range_reshape == y_true, dtype=dtype)
        
        one = tf.constant(1.0, dtype=tf.float32)
        one_sig_out = tf.sigmoid(y_pred)
        zero_sig_out = one - one_sig_out
        
        if self.edge_loss_weighting:
            # Compute Beta
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
            num_pixel = y_true.shape[1] * y_true.shape[2]
            num_non_edge_pixel = num_pixel - num_edge_pixel
            edge_loss_weighting = num_non_edge_pixel / num_pixel
            edge_loss_weighting = tf.clip_by_value(edge_loss_weighting, self.min_edge_loss_weighting,
                                                   self.max_edge_loss_weighting)
            edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
            
            loss = - edge_loss_weighting * y_true * tf.math.pow(1 - one_sig_out, power) * tf.math.log(
                tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - edge_loss_weighting) * (1 - y_true) * tf.math.pow(
                1 - zero_sig_out, power) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        else:
            loss = - y_true * tf.math.pow(1 - one_sig_out, power) * \
                   tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - \
                   (1 - y_true) * tf.math.pow(1 - zero_sig_out, power) * \
                   tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=-1), axis=[1, 2]))
        
        return loss


def weighted_multi_label_sigmoid_loss_alternative_implementation(y_true, y_prediction):
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
