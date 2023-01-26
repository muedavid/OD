import tensorflow as tf
import numpy as np
from utils.tools import mask_pixels_for_computation


@tf.function
def get_edge_weighting_matrix(y_true, pixels_at_edge_without_loss, edge_loss_weighting):
    y_true = tf.cast(y_true, tf.float32)
    filter_dilation = tf.zeros(
        shape=(1 + 2 * (pixels_at_edge_without_loss + 1), 1 + 2 * (pixels_at_edge_without_loss + 1),
               y_true.shape[-1]))
    y_widen_max = tf.nn.dilation2d(y_true, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                                   dilations=[1, 1, 1, 1], data_format="NHWC")
    filter_dilation = tf.zeros(
        shape=(1 + 2 * pixels_at_edge_without_loss, 1 + 2 * pixels_at_edge_without_loss,
               y_true.shape[-1]))
    y_widen = tf.nn.dilation2d(y_true, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                               dilations=[1, 1, 1, 1], data_format="NHWC")
    y_edge_weight_weight = y_widen_max - y_widen + y_true
    y_low_weight = y_widen - y_true
    y_non_edge_weight = 1.0 - y_widen_max
    weight = y_edge_weight_weight * edge_loss_weighting + y_non_edge_weight * (
            1.0 - edge_loss_weighting) + y_low_weight * (1.0 - edge_loss_weighting) / 1.0
    return weight


@tf.function
def derivative_focal_loss(sig, gamma):
    return - (1.0 / sig) * tf.pow(1.0 - sig, gamma) + gamma * tf.pow(1.0 - sig, gamma - 1.0) * tf.math.log(
        sig) * sig * (1.0 - sig)


class WeightedMultiLabelSigmoidLoss(tf.keras.losses.Loss):
    def __init__(self, min_edge_loss_weighting=0.005, max_edge_loss_weighting=0.995, class_individually_weighted=False,
                 padding=0, pixels_at_edge_without_loss=0, name='weighted_multi_label_sigmoid_loss'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.class_individually_weighted = class_individually_weighted
        self.padding = padding
        self.pixels_at_edge_without_loss = pixels_at_edge_without_loss
        size = 13
        sig = 4
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        kernel = np.expand_dims(kernel, axis=[2, 3])
        self.kernel = tf.constant(kernel, tf.float32)
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        
        min_edge_loss_weighting = tf.cast(self.min_edge_loss_weighting, dtype=dtype)
        max_edge_loss_weighting = tf.cast(self.max_edge_loss_weighting, dtype=dtype)
        
        y_true = tf.cast(y_true, dtype=dtype)
        
        mask = mask_pixels_for_computation(y_true, padding=self.padding)
        
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
        edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
        
        weight = get_edge_weighting_matrix(y_true, self.pixels_at_edge_without_loss, edge_loss_weighting)
        
        # Loss
        y_true = tf.cast(y_true, dtype)
        y_prediction = tf.cast(y_pred, dtype)
        one = tf.constant(1.0, dtype=dtype)
        one_sig_out = y_prediction
        zero_sig_out = one - one_sig_out
        
        loss = - y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - y_true) * tf.math.log(
            tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        loss = loss * mask * weight
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['min_edge_loss_weighting'] = self.min_edge_loss_weighting
        base_config['max_edge_loss_weighting'] = self.max_edge_loss_weighting
        base_config['class_individually_weighted'] = self.class_individually_weighted
        base_config['padding'] = self.padding
        base_config['pixels_at_edge_without_loss'] = self.pixels_at_edge_without_loss
        return base_config


class FocalLossEdges(tf.keras.losses.Loss):
    def __init__(self, power=2.0, edge_loss_weighting=True, min_edge_loss_weighting=0.005,
                 max_edge_loss_weighting=0.995, padding=0, pixels_at_edge_without_loss=0,
                 decay=False, focal_loss_derivative_threshold=0.0,
                 name='focal_loss_edges'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.edge_loss_weighting = edge_loss_weighting
        self.power = tf.Variable(power, tf.float32)
        self.padding = padding
        self.pixels_at_edge_without_loss = pixels_at_edge_without_loss
        self.decay = decay
        self.focal_loss_derivative_threshold = focal_loss_derivative_threshold
        self.iterations = tf.Variable(0, tf.int32)
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        y_true = tf.cast(y_true, dtype=dtype)
        
        # if self.power.value() > 0.005:
        #     self.power.assign_sub(0.0005)
        # else:
        #     self.power.assign(0.0)
        mask = mask_pixels_for_computation(y_true, self.padding)
        
        power = tf.cast(self.power.value(), dtype=dtype)
        
        min_steps_to_check_derivative = 2000
        self.iterations.assign_add(1)
        
        if self.decay and self.iterations.value() % min_steps_to_check_derivative == 0 and self.power.value() > 0.0:
            confidence = tf.reduce_sum((y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)) * mask)
            confidence = confidence / tf.reduce_sum(mask)
            derivative_avg = derivative_focal_loss(confidence, self.power.value())
            tf.print("\nDerivative of Focal Loss is: ", derivative_avg)
            if -derivative_avg <= tf.constant(
                    self.focal_loss_derivative_threshold) or self.iterations.value() >= min_steps_to_check_derivative * 10:
                self.power.assign_sub(1.0)
                tf.print("\nAverage is reduced.\nNew Power: ", self.power.value(), "\n \n")
        
        # weight = tf.cast(tf.where(tf.reduce_sum(y_true, keepdims=True) > 0, 1.0, 0.0), tf.float32)
        # filter_dilation = tf.zeros(shape=(11, 7, 1))
        # weight = tf.nn.dilation2d(weight, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
        #                           dilations=[1, 1, 1, 1], data_format="NHWC")
        # weight = weight + 0.5
        
        one = tf.constant(1.0, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0.005, 0.995)
        one_sig_out = y_pred
        zero_sig_out = one - one_sig_out
        
        background_loss = - (1.0 - y_true) * tf.math.pow(one_sig_out, power) * tf.math.log(
            tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        # asymetric: removed * tf.math.pow(zero_sig_out, power)
        edge_loss = - y_true * tf.math.pow(zero_sig_out, power) * tf.math.log(
            tf.clip_by_value(one_sig_out, 1e-10, 1000))
        loss = background_loss + edge_loss
        
        if self.edge_loss_weighting:
            # Compute Beta
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2], keepdims=True)
            num_pixel = y_true.shape[1] * y_true.shape[2]
            num_non_edge_pixel = num_pixel - num_edge_pixel
            edge_loss_weighting = num_non_edge_pixel / num_pixel
            edge_loss_weighting = tf.clip_by_value(edge_loss_weighting, self.min_edge_loss_weighting,
                                                   self.max_edge_loss_weighting)
            edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
            
            weight = get_edge_weighting_matrix(y_true, self.pixels_at_edge_without_loss, edge_loss_weighting)
            loss = loss * weight
        
        loss = loss * mask
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['min_edge_loss_weighting'] = self.min_edge_loss_weighting
        base_config['max_edge_loss_weighting'] = self.max_edge_loss_weighting
        base_config['edge_loss_weighting'] = self.edge_loss_weighting
        base_config['padding'] = self.padding
        base_config['pixels_at_edge_without_loss'] = self.pixels_at_edge_without_loss
        base_config['decay'] = self.decay
        base_config['focal_loss_derivative_threshold'] = self.focal_loss_derivative_threshold
        return base_config


def weighted_multi_label_sigmoid_loss_alternative_implementation(y_true, y_prediction):
    # Transform y_true to Tensor
    y_true = tf.cast(y_true, dtype=tf.float32)
    
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
