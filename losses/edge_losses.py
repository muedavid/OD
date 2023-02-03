import tensorflow as tf
from utils.tools import mask_pixels_for_computation


@tf.function
def get_edge_weighting_matrix(y_true,
                              num_pixels_region_of_attraction,
                              max_edge_loss_weighting_edge=0.5,
                              max_edge_loss_weighting_non_edge=0.5,
                              min_edge_loss_weighting_region_of_attraction=0.5,
                              min_edge_loss_weighting_non_edge=0.5):
    y_true = tf.cast(y_true, tf.float32)
    filter_max_non_edge = tf.zeros(
        shape=(1 + 2 * (num_pixels_region_of_attraction + 1), 1 + 2 * (num_pixels_region_of_attraction + 1),
               y_true.shape[-1]))
    y_max_non_edge_dilated = tf.nn.dilation2d(y_true,
                                              filter_max_non_edge,
                                              strides=[1, 1, 1, 1],
                                              padding="SAME",
                                              dilations=[1, 1, 1, 1],
                                              data_format="NHWC")
    filter_region_of_attraction = tf.zeros(
        shape=(1 + 2 * num_pixels_region_of_attraction, 1 + 2 * num_pixels_region_of_attraction,
               y_true.shape[-1]))
    y_region_of_attraction_dilated = tf.nn.dilation2d(y_true,
                                                      filter_region_of_attraction,
                                                      strides=[1, 1, 1, 1],
                                                      padding="SAME",
                                                      dilations=[1, 1, 1, 1],
                                                      data_format="NHWC")
    y_region_of_attraction = y_region_of_attraction_dilated - y_true
    y_max_non_edge = y_max_non_edge_dilated - y_region_of_attraction_dilated
    y_non_edge = 1.0 - y_max_non_edge_dilated
    weight = y_region_of_attraction * min_edge_loss_weighting_region_of_attraction + y_non_edge * min_edge_loss_weighting_non_edge + y_max_non_edge * max_edge_loss_weighting_non_edge + y_true * max_edge_loss_weighting_edge
    return weight


@tf.function
def derivative_focal_loss(sig, gamma):
    return - (1.0 / sig) * tf.pow(1.0 - sig, gamma) + gamma * tf.pow(1.0 - sig, gamma - 1.0) * tf.math.log(
        sig) * sig * (1.0 - sig)


class EdgeSigmoidLoss(tf.keras.losses.Loss):
    def __init__(self,
                 edge_loss_weighting,
                 max_edge_loss_weighting_edge=0.5,
                 max_edge_loss_weighting_non_edge=0.5,
                 min_edge_loss_weighting_region_of_attraction=0.5,
                 min_edge_loss_weighting_non_edge=0.5,
                 padding=0,
                 num_pixels_region_of_attraction=0,
                 name='edge_sigmoid_loss'):
        super().__init__(name=name)
        self.edge_loss_weighting = edge_loss_weighting
        self.max_edge_loss_weighting_edge = max_edge_loss_weighting_edge
        self.max_edge_loss_weighting_non_edge = max_edge_loss_weighting_non_edge
        self.min_edge_loss_weighting_region_of_attraction = min_edge_loss_weighting_region_of_attraction
        self.min_edge_loss_weighting_non_edge = min_edge_loss_weighting_non_edge
        self.padding = padding
        self.num_pixels_region_of_attraction = num_pixels_region_of_attraction
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        y_true = tf.cast(y_true, dtype=dtype)
        
        padding_mask = mask_pixels_for_computation(y_true, padding=self.padding)
        
        y_true = tf.cast(y_true, dtype)
        y_prediction = tf.cast(y_pred, dtype)
        one = tf.constant(1.0, dtype=dtype)
        one_sig_out = y_prediction
        zero_sig_out = one - one_sig_out
        
        if self.edge_loss_weighting:
            weight = get_edge_weighting_matrix(y_true,
                                               self.num_pixels_region_of_attraction,
                                               self.max_edge_loss_weighting_edge,
                                               self.max_edge_loss_weighting_non_edge,
                                               self.min_edge_loss_weighting_region_of_attraction,
                                               self.min_edge_loss_weighting_non_edge)
        
        else:
            # weighted by inverse sampling frequency
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
            num_edge_pixel = tf.cast(num_edge_pixel, dtype=dtype)
            num_pixel = y_true.shape[1] * y_true.shape[2]
            num_pixel = tf.cast(num_pixel, dtype=dtype)
            num_non_edge_pixel = num_pixel - num_edge_pixel
            num_non_edge_pixel = tf.cast(num_non_edge_pixel, dtype=dtype)
            edge_loss_weighting = num_non_edge_pixel / num_pixel
            edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
            weight = get_edge_weighting_matrix(y_true, 0.0, edge_loss_weighting, 1.0 - edge_loss_weighting,
                                               1.0 - edge_loss_weighting, 1.0 - edge_loss_weighting)
        
        loss = - y_true * tf.math.log(tf.clip_by_value(one_sig_out, 1e-10, 1000)) - (1 - y_true) * tf.math.log(
            tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        loss = loss * padding_mask * weight
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['edge_loss_weighting'] = self.edge_loss_weighting
        base_config['max_edge_loss_weighting_edge'] = self.max_edge_loss_weighting_edge
        base_config['max_edge_loss_weighting_non_edge'] = self.max_edge_loss_weighting_non_edge
        base_config['min_edge_loss_weighting_region_of_attraction'] = self.min_edge_loss_weighting_region_of_attraction
        base_config['min_edge_loss_weighting_non_edge'] = self.min_edge_loss_weighting_non_edge
        base_config['padding = padding'] = self.padding
        base_config['num_pixels_region_of_attraction'] = self.num_pixels_region_of_attraction
        return base_config


class FocalLossEdges(tf.keras.losses.Loss):
    def __init__(self,
                 edge_loss_weighting,
                 max_edge_loss_weighting_edge=0.5,
                 max_edge_loss_weighting_non_edge=0.5,
                 min_edge_loss_weighting_region_of_attraction=0.5,
                 min_edge_loss_weighting_non_edge=0.5,
                 padding=0,
                 num_pixels_region_of_attraction=0,
                 power=2.0,
                 decay=False,
                 focal_loss_derivative_threshold=0.0,
                 name='focal_loss_edges'):
        super().__init__(name=name)
        self.edge_loss_weighting = edge_loss_weighting
        self.max_edge_loss_weighting_edge = max_edge_loss_weighting_edge
        self.max_edge_loss_weighting_non_edge = max_edge_loss_weighting_non_edge
        self.min_edge_loss_weighting_region_of_attraction = min_edge_loss_weighting_region_of_attraction
        self.min_edge_loss_weighting_non_edge = min_edge_loss_weighting_non_edge
        self.padding = padding
        self.num_pixels_region_of_attraction = num_pixels_region_of_attraction
        self.power = tf.Variable(power, tf.float32)
        self.decay = decay
        self.focal_loss_derivative_threshold = focal_loss_derivative_threshold
        self.iterations = tf.Variable(0, tf.int32)
    
    @tf.function
    def call(self, y_true, y_pred):
        dtype = tf.float32
        y_true = tf.cast(y_true, dtype=dtype)
        
        mask_padding = mask_pixels_for_computation(y_true, self.padding)
        
        power = tf.cast(self.power.value(), dtype=dtype)
        min_steps_to_check_derivative = 1000
        self.iterations.assign_add(1)
        
        if self.decay and self.iterations.value() % min_steps_to_check_derivative == 0 and self.power.value() > 0.0:
            confidence = tf.reduce_sum((y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)) * mask_padding)
            confidence = confidence / tf.reduce_sum(mask_padding)
            derivative_avg = derivative_focal_loss(confidence, self.power.value())
            tf.print("\nDerivative of Focal Loss is: ", derivative_avg)
            if -derivative_avg <= tf.constant(
                    self.focal_loss_derivative_threshold) or self.iterations.value() >= min_steps_to_check_derivative * 10:
                self.power.assign_sub(1.0)
                tf.print("\nAverage is reduced.\nNew Power: ", self.power.value(), "\n \n")
        
        y_true = tf.cast(y_true, dtype)
        y_prediction = tf.cast(y_pred, dtype)
        one = tf.constant(1.0, dtype=dtype)
        one_sig_out = y_prediction
        zero_sig_out = one - one_sig_out
        
        if self.edge_loss_weighting:
            weight = get_edge_weighting_matrix(y_true,
                                               self.num_pixels_region_of_attraction,
                                               self.max_edge_loss_weighting_edge,
                                               self.max_edge_loss_weighting_non_edge,
                                               self.min_edge_loss_weighting_region_of_attraction,
                                               self.min_edge_loss_weighting_non_edge)
        
        else:
            # weighted by inverse sampling frequency
            num_edge_pixel = tf.reduce_sum(y_true, axis=[1, 2, 3], keepdims=True)
            num_edge_pixel = tf.cast(num_edge_pixel, dtype=dtype)
            num_pixel = y_true.shape[1] * y_true.shape[2]
            num_pixel = tf.cast(num_pixel, dtype=dtype)
            num_non_edge_pixel = num_pixel - num_edge_pixel
            num_non_edge_pixel = tf.cast(num_non_edge_pixel, dtype=dtype)
            edge_loss_weighting = num_non_edge_pixel / num_pixel
            edge_loss_weighting = tf.cast(edge_loss_weighting, dtype)
            weight = get_edge_weighting_matrix(y_true, 0.0, edge_loss_weighting, 1.0 - edge_loss_weighting,
                                               1.0 - edge_loss_weighting, 1.0 - edge_loss_weighting)
        
        background_loss = - (1.0 - y_true) * tf.math.pow(one_sig_out, power) * tf.math.log(
            tf.clip_by_value(zero_sig_out, 1e-10, 1000))
        edge_loss = - y_true * tf.math.pow(zero_sig_out, power) * tf.math.log(
            tf.clip_by_value(one_sig_out, 1e-10, 1000))
        loss = background_loss + edge_loss
        
        loss = loss * mask_padding * weight
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]))
        
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        base_config['edge_loss_weighting'] = self.edge_loss_weighting
        base_config['max_edge_loss_weighting_edge'] = self.max_edge_loss_weighting_edge
        base_config['max_edge_loss_weighting_non_edge'] = self.max_edge_loss_weighting_non_edge
        base_config['min_edge_loss_weighting_region_of_attraction'] = self.min_edge_loss_weighting_region_of_attraction
        base_config['min_edge_loss_weighting_non_edge'] = self.min_edge_loss_weighting_non_edge
        base_config['padding = padding'] = self.padding
        base_config['num_pixels_region_of_attraction'] = self.num_pixels_region_of_attraction
        base_config['padding'] = self.padding
        base_config['decay'] = self.decay
        base_config['focal_loss_derivative_threshold'] = self.focal_loss_derivative_threshold
        return base_config
