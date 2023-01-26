import tensorflow as tf
from utils.tools import mask_pixels_for_computation


@tf.function
def compute_f1_precision_recall(true_positive, false_positive, false_negative):
    precision = tf.where(true_positive + false_positive != 0, true_positive / (true_positive + false_positive), 0)
    recall = tf.where((true_positive + false_negative) != 0, true_positive / (true_positive + false_negative), 0)
    f1 = tf.where(precision + recall != 0, 2 * precision * recall / (precision + recall), 0)
    
    return f1, precision, recall


# @tf.function
# def number_true_false_positive_negative(y_true, y_prediction, padding, pixels_at_edge_without_loss):
#     mask = mask_pixels_for_computation(y_true, padding, 0)
#     mask = tf.cast(mask, tf.int32)
#
#     number_true_positive = tf.reduce_sum(tf.cast((y_true & y_prediction), tf.int32) * mask, axis=(0, 1, 2))
#     number_false_positive = tf.reduce_sum(y_prediction * mask, axis=(0, 1, 2)) - number_true_positive
#     number_true_negative = tf.reduce_sum(tf.cast(((1 - y_prediction) & (1 - y_true)), tf.int32) * mask, axis=(0, 1, 2))
#     number_false_negative = tf.reduce_sum(tf.cast((1 - y_prediction) * mask, tf.int32),
#                                           axis=(0, 1, 2)) - number_true_negative
#
#     return number_true_positive, number_false_positive, number_true_negative, number_false_negative


@tf.function
def number_true_false_positive_negative(y_true, y_prediction, threshold_edge_width=0, padding=0,
                                        pixels_at_edge_without_loss=0):
    mask = mask_pixels_for_computation(y_true, padding, pixels_at_edge_without_loss)
    mask = tf.cast(mask, tf.int32)
    
    # widen the edges for the calculation of the number of true positive
    y_prediction_widen = tf.cast(y_prediction, tf.float32)
    filter_dilation = tf.zeros(shape=(1 + 2 * threshold_edge_width, 1 + 2 * threshold_edge_width, y_prediction_widen.shape[-1]))
    y_prediction_widen = tf.nn.dilation2d(y_prediction_widen, filter_dilation, strides=[1, 1, 1, 1], padding="SAME",
                                          dilations=[1, 1, 1, 1], data_format="NHWC")
    y_prediction_widen = tf.cast(y_prediction_widen, tf.int32)
    
    number_true_positive = tf.reduce_sum(tf.cast((y_true & y_prediction_widen), tf.int32) * mask, axis=(0, 1, 2))
    # number_true_positive_1 = tf.reduce_sum(tf.cast((y_true & y_prediction_widen), tf.int32), axis=(0, 1, 2))
    # y_true_widen = tf.cast(y_true, tf.float32)
    # y_true_widen = tf.nn.depthwise_conv2d(y_true_widen, kernel, strides=[1, 1, 1, 1], padding="SAME")
    # y_true_widen = tf.cast(tf.clip_by_value(y_true_widen, 0, 1), tf.int32)
    # number_true_positive_2 = tf.reduce_sum(tf.cast((y_true_widen & y_prediction), tf.int32), axis=(0, 1, 2))
    # number_true_positive = tf.math.minimum(number_true_positive_1, number_true_positive_2)
    number_false_positive = tf.reduce_sum(y_prediction * mask, axis=(0, 1, 2)) - number_true_positive
    
    number_true_negative = tf.reduce_sum(tf.cast((1 - y_prediction) & (1 - y_true) * mask, tf.int32), axis=(0, 1, 2))
    number_false_negative = tf.reduce_sum(tf.cast((1 - y_prediction), tf.int32) * mask,
                                          axis=(0, 1, 2)) - number_true_negative
    
    return number_true_positive, number_false_positive, number_true_negative, number_false_negative


class BinaryAccuracyEdges(tf.keras.metrics.Metric):
    def __init__(self, num_classes, classes_individually=False, name="accuracy_edges",
                 threshold_prediction=0.5, print_name="edges", padding=0, pixels_at_edge_without_loss=0, **kwargs):
        super(BinaryAccuracyEdges, self).__init__(name=name, **kwargs)
        self.numberTruePredictedPixels = self.add_weight(name="numberTruePredictedPixels", initializer="zeros",
                                                         shape=(num_classes,))
        self.numberPixels = self.add_weight(name="numberPixels", initializer="zeros",
                                            shape=(num_classes,))
        self.thresholdPrediction = threshold_prediction
        self.num_classes = num_classes
        self.classes_individually = classes_individually
        self.print_name = print_name
        self.padding = padding
        self.pixels_at_edge_without_loss = pixels_at_edge_without_loss
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold_prediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold_prediction, tf.int32)
        y_true = tf.cast(y_true, dtype=tf.int32)
        
        mask = mask_pixels_for_computation(y_true, self.padding, self.pixels_at_edge_without_loss)
        mask = tf.cast(mask, tf.int32)
        
        number_true_predicted = tf.cast(tf.reduce_sum(tf.cast((y_true == y_pred), dtype=tf.int32) * mask,
                                                      axis=[0, 1, 2]), dtype=tf.float32)
        
        self.numberTruePredictedPixels.assign_add(number_true_predicted)
        self.numberPixels.assign_add(tf.cast(tf.reduce_sum(mask, axis=[0, 1, 2]), tf.float32))
    
    @tf.function
    def result(self):
        accuracy = self.numberTruePredictedPixels / self.numberPixels
        metric_dict = {'accuracy': tf.reduce_mean(accuracy)}
        if self.classes_individually:
            for i in range(1, self.num_classes + 1):
                metric_dict['accuracy_' + self.print_name + "_{}".format(i)] = accuracy[i - 1]
        
        return metric_dict
    
    @tf.function
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.numberTruePredictedPixels.assign(tf.zeros((self.num_classes,)))
        self.numberPixels.assign(tf.zeros((self.num_classes,)))
    
    def get_config(self):
        base_config = super().get_config()
        base_config['print_name'] = self.print_name
        base_config['threshold_prediction'] = self.thresholdPrediction
        base_config['num_classes'] = self.num_classes
        base_config['classes_individually'] = self.classes_individually
        base_config['padding'] = self.padding
        base_config['pixels_at_edge_without_loss'] = self.pixels_at_edge_without_loss
        return base_config


class F1Edges(tf.keras.metrics.Metric):
    def __init__(self, num_classes, classes_individually=False, threshold_prediction=0.5,
                 print_name="edges", padding=0, pixels_at_edge_without_loss=0, threshold_edge_width=0,
                 name="f1_edges",
                 **kwargs):
        super(F1Edges, self).__init__(name=name, **kwargs)
        
        self.thresholdPrediction = threshold_prediction
        self.classes_individually = classes_individually
        self.num_classes = num_classes
        self.print_name = print_name
        self.padding = padding
        self.pixels_at_edge_without_loss = pixels_at_edge_without_loss
        self.threshold_edge_width = threshold_edge_width
        
        self.numberTruePositive = self.add_weight(name="numberTruePositive", initializer="zeros", shape=(num_classes,))
        self.numberTrueNegative = self.add_weight(name="numberTrueNegative", initializer="zeros", shape=(num_classes,))
        self.numberFalsePositive = self.add_weight(name="numberFalsePositive", initializer="zeros",
                                                   shape=(num_classes,))
        self.numberFalseNegative = self.add_weight(name="numberFalseNegative", initializer="zeros",
                                                   shape=(num_classes,))
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold_prediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold_prediction, tf.int32)
        y_true = tf.cast(y_true, dtype=tf.int32)
        
        number_true_positive, number_false_positive, number_true_negative, number_false_negative = \
            number_true_false_positive_negative(y_true, y_pred, padding=self.padding,
                                                threshold_edge_width=self.threshold_edge_width,
                                                pixels_at_edge_without_loss=self.pixels_at_edge_without_loss)
        
        self.numberTruePositive.assign_add(tf.cast(number_true_positive, tf.float32))
        self.numberFalsePositive.assign_add(tf.cast(number_false_positive, tf.float32))
        self.numberTrueNegative.assign_add(tf.cast(number_true_negative, tf.float32))
        self.numberFalseNegative.assign_add(tf.cast(number_false_negative, tf.float32))
    
    @tf.function
    def result(self):
        mean_scores = compute_f1_precision_recall(tf.reduce_sum(self.numberTruePositive, keepdims=True),
                                                  tf.reduce_sum(self.numberFalsePositive, keepdims=True),
                                                  tf.reduce_sum(self.numberFalseNegative, keepdims=True))
        
        metric_dict = {"f1_" + self.print_name: mean_scores[0][0], "precision_" + self.print_name: mean_scores[1][0],
                       "recall_" + self.print_name: mean_scores[2][0]}
        
        if self.classes_individually:
            f1, precision, recall = compute_f1_precision_recall(self.numberTruePositive, self.numberFalsePositive,
                                                                self.numberFalseNegative)
            for i in range(1, self.num_classes + 1):
                metric_dict["f1_" + self.print_name + "_{}".format(i)] = f1[i - 1]
                metric_dict["precision_" + self.print_name + "_{}".format(i)] = precision[i - 1]
                metric_dict["recall_" + self.print_name + "_{}".format(i)] = recall[i - 1]
        
        return metric_dict
    
    @tf.function
    def reset_state(self):
        self.numberTruePositive.assign(tf.zeros((self.num_classes,)))
        self.numberFalsePositive.assign(tf.zeros((self.num_classes,)))
        self.numberTrueNegative.assign(tf.zeros((self.num_classes,)))
        self.numberFalseNegative.assign(tf.zeros((self.num_classes,)))
    
    def get_config(self):
        base_config = super().get_config()
        base_config['num_classes'] = self.num_classes
        base_config['print_name'] = self.print_name
        base_config['threshold_prediction'] = self.thresholdPrediction
        base_config['classes_individually'] = self.classes_individually
        base_config['padding'] = self.padding
        base_config['pixels_at_edge_without_loss'] = self.pixels_at_edge_without_loss
        base_config['threshold_edge_width'] = self.threshold_edge_width
        return base_config
