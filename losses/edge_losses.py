import tensorflow as tf
import numpy as np


class WeightedMultiLabelSigmoidLoss(tf.keras.losses.Loss):
    def __init__(self, min_edge_loss_weighting=0.005, max_edge_loss_weighting=0.995, class_individually_weighted=False,
                 name='weighted_multi_label_sigmoid_loss'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.class_individually_weighted = class_individually_weighted
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
        
        # reduce y_true
        weight = tf.cast(tf.where(tf.reduce_sum(y_true, keepdims=True) > 0, 1.0, 0.0), tf.float32)
        filter_dilation = tf.zeros(shape=(3, 3, 1))
        weight = tf.nn.dilation2d(weight, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                                  dilations=[1, 1, 1, 1], data_format="NHWC")
        weight = tf.nn.conv2d(weight, self.kernel, strides=[1, 1, 1, 1], padding="SAME") + 0.1
        
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
        one_sig_out = y_prediction
        zero_sig_out = one - one_sig_out
        
        loss = -edge_loss_weighting * y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (
                1 - edge_loss_weighting) * (
                       1 - y_true) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        loss = loss * weight
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['min_edge_loss_weighting'] = self.min_edge_loss_weighting
        base_config['max_edge_loss_weighting'] = self.max_edge_loss_weighting
        base_config['class_individually_weighted'] = self.class_individually_weighted
        return base_config

@tf.function
def derivative_focal_loss(sig, gamma):
    return - (1.0 / sig) * tf.pow(1.0 - sig, gamma) + gamma * tf.pow(1.0 - sig, gamma - 1.0) * tf.math.log(
        sig) * sig * (1.0 - sig)


class FocalLossEdges(tf.keras.losses.Loss):
    def __init__(self, power=2.0, edge_loss_weighting=True, min_edge_loss_weighting=0.005,
                 max_edge_loss_weighting=0.995,
                 name='focal_loss_edges'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.edge_loss_weighting = edge_loss_weighting
        self.power = tf.Variable(power, tf.float32)
        self.iterations = tf.Variable(0, tf.int32)
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        y_true = tf.cast(y_true, dtype=dtype)
        
        # if self.power.value() > 0.005:
        #     self.power.assign_sub(0.0005)
        # else:
        #     self.power.assign(0.0)
        
        power = tf.cast(self.power.value(), dtype=dtype)
        self.iterations.assign_add(1)
        
        if self.iterations.value() % 2000 == 0 and self.power.value() > 0.0:
            confidence = tf.reduce_mean(y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred))
            derivative_avg = derivative_focal_loss(confidence, self.power.value())
            if -derivative_avg <= tf.constant(0.05):
                tf.print("average is reduced \n \n \n ", derivative_avg)
                tf.print(self.power.value())
                self.power.assign_sub(1.0)
        
        # dist = 1
        # y_true_list = [y_true]
        # for i in range(dist):
        #     filter_dilation = tf.zeros(shape=(1 + 2 * (i + 1), 1 + 2 * (i + 1), 1))
        #     y_true_widen = tf.nn.dilation2d(y_true, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
        #                                     dilations=[1, 1, 1, 1],
        #                                     data_format="NHWC")
        #     # y_true_list.append(y_true_widen * (1 - (i + 1) / (dist + 1)))
        #     y_true_list.append(y_true_widen * 0.3)
        # y_true = tf.keras.layers.Maximum()(y_true_list)
        
        # reduce y_true
        weight = tf.cast(tf.where(tf.reduce_sum(y_true, keepdims=True) > 0, 1.0, 0.0), tf.float32)
        filter_dilation = tf.zeros(shape=(11, 7, 1))
        weight = tf.nn.dilation2d(weight, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                                  dilations=[1, 1, 1, 1], data_format="NHWC")
        weight = weight + 0.5
        
        weight_edges = tf.cast(tf.where(tf.reduce_sum(y_true, keepdims=True) > 0, 1.0, 0.0), tf.float32)
        weight_edges_ones = tf.ones(shape=(1, y_true.shape[1], y_true.shape[2], 1))
        filter_dilation = tf.zeros(shape=(5, 5, 1))
        weight_edges_widen = tf.nn.dilation2d(weight_edges, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                                              dilations=[1, 1, 1, 1], data_format="NHWC")
        weight_edges = weight_edges_ones - weight_edges_widen + weight_edges
        
        # weight border:
        padding = 5
        weight_border = np.ones(shape=(1, y_true.shape[1], y_true.shape[2], 1))
        for row in range(weight_border.shape[1]):
            for col in range(weight_border.shape[2]):
                if row < padding or row > weight_border.shape[1] - padding - 1 or col < padding or col > \
                        weight_border.shape[2] - padding - 1:
                    weight_border[:, row, col, :] = 0.0
        weight_border = tf.constant(weight_border, tf.float32)
        
        one = tf.constant(1.0, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0.005, 0.995)
        one_sig_out = y_pred
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
            
            background_loss = - (1.0 - edge_loss_weighting) * (
                    1.0 - y_true) * tf.math.pow(
                one_sig_out, power) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
            # asymetric: removed * tf.math.pow(zero_sig_out, power)
            edge_loss = - edge_loss_weighting * y_true * tf.math.pow(zero_sig_out, power) * tf.math.log(
                tf.clip_by_value(one_sig_out, 1e-10, 1000))
            loss = background_loss + edge_loss
        else:
            background_loss = - (1.0 - y_true) * tf.math.pow(one_sig_out, power) * \
                              tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
            # asymetric: removed * tf.math.pow(zero_sig_out, power)
            edge_loss = - y_true * tf.math.pow(zero_sig_out, power) * \
                        tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000))
            loss = background_loss + edge_loss
        
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=-1), axis=[1, 2]))
        loss = loss * weight_border * weight_edges * weight
        
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['min_edge_loss_weighting'] = self.min_edge_loss_weighting
        base_config['max_edge_loss_weighting'] = self.max_edge_loss_weighting
        base_config['edge_loss_weighting'] = self.edge_loss_weighting
        # base_config['power'] = self.power.value()
        return base_config


class FlowLoss(tf.keras.losses.Loss):
    def __init__(self, large_loss_threshold=4.0, name='flow_loss'):
        super().__init__(name=name)
        self.large_loss_threshold = large_loss_threshold
    
    @tf.function
    def call(self, y_true, y_pred):
        mult = 2.0
        
        large_loss = tf.where(tf.abs(y_true - y_pred) >= self.large_loss_threshold, 1.0, 0.0)
        large_loss = tf.cast(large_loss, tf.float32)
        
        y_true_mask = tf.where(y_true != 0, 1, 0) == tf.where(tf.abs(y_true) <= 4, 1, 0)
        y_true_mask = tf.cast(y_true_mask, tf.float32)
        # smoothing_loss = tf.pow((y_pred-tf.reduce_mean(y_pred*y_true_mask, axis=[1, 2], keepdims=True), 2)*y_true_mask)
        loss = mult * large_loss * tf.abs(y_true - y_pred) + \
               mult / self.large_loss_threshold * (1 - large_loss) * tf.math.square(y_true - y_pred)
        loss = loss * y_true_mask
        return 10 * tf.reduce_sum(loss) / (tf.reduce_sum(y_true_mask) + 1)  # + tf.reduce_mean(smoothing_loss)
    
    def get_config(self):
        base_config = super().get_config()
        base_config['large_loss_threshold'] = self.large_loss_threshold
        return base_config


class SegmentationLoss(tf.keras.losses.Loss):
    def __init__(self, min_edge_loss_weighting=0.005, max_edge_loss_weighting=0.995, class_individually_weighted=False,
                 name='segmentation_loss'):
        super().__init__(name=name)
        self.min_edge_loss_weighting = min_edge_loss_weighting
        self.max_edge_loss_weighting = max_edge_loss_weighting
        self.class_individually_weighted = class_individually_weighted
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        
        min_edge_loss_weighting = tf.cast(self.min_edge_loss_weighting, dtype=dtype)
        max_edge_loss_weighting = tf.cast(self.max_edge_loss_weighting, dtype=dtype)
        
        y_true = tf.cast(y_true, dtype=dtype)
        
        # filter_dilation = tf.zeros(shape=(7, 7, 5))
        # weight_large = tf.nn.dilation2d(y_true, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
        #                                 dilations=[1, 1, 1, 1], data_format="NHWC")
        # weight_small = tf.nn.erosion2d(y_true, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
        #                                dilations=[1, 1, 1, 1], data_format="NHWC")
        # weight = (weight_large - weight_small)*0.5 + 1.0
        
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
        one_sig_out = y_prediction
        zero_sig_out = one - one_sig_out
        
        loss = -edge_loss_weighting * y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (
                1 - edge_loss_weighting) * (
                       1 - y_true) * tf.math.log(tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        # loss = loss * weight
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3])) / 5
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['min_edge_loss_weighting'] = self.min_edge_loss_weighting
        base_config['max_edge_loss_weighting'] = self.max_edge_loss_weighting
        base_config['class_individually_weighted'] = self.class_individually_weighted
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
