import tensorflow as tf


@tf.function
def number_true_false_positive_negative(y_true, y_prediction, threshold_edge_width):
    # widen the edges for the calculation of the number of true positive
    kernel = tf.ones([1 + 2 * threshold_edge_width, 1 + 2 * threshold_edge_width, y_prediction.shape[-1], 1],
                     tf.float32)
    y_prediction_widen = tf.cast(y_prediction, tf.float32)
    y_prediction_widen = tf.nn.depthwise_conv2d(y_prediction_widen, kernel, strides=[1, 1, 1, 1], padding="SAME")
    y_prediction_widen = tf.cast(tf.clip_by_value(y_prediction_widen, 0, 1), tf.int32)
    
    number_true_positive_1 = tf.reduce_sum(tf.cast((y_true & y_prediction_widen), tf.int32), axis=(0, 1, 2))
    y_true_widen = tf.cast(y_true, tf.float32)
    y_true_widen = tf.nn.depthwise_conv2d(y_true_widen, kernel, strides=[1, 1, 1, 1], padding="SAME")
    y_true_widen = tf.cast(tf.clip_by_value(y_true_widen, 0, 1), tf.int32)
    number_true_positive_2 = tf.reduce_sum(tf.cast((y_true_widen & y_prediction), tf.int32), axis=(0, 1, 2))
    number_true_positive = tf.math.minimum(number_true_positive_1, number_true_positive_2)
    number_false_positive = tf.reduce_sum(y_prediction, axis=(0, 1, 2)) - number_true_positive
    
    number_true_negative = tf.reduce_sum(tf.cast((1 - y_prediction) & (1 - y_true), tf.int32), axis=(0, 1, 2))
    number_false_negative = tf.reduce_sum(tf.cast((1 - y_prediction), tf.int32), axis=(0, 1, 2)) - number_true_negative
    
    return number_true_positive, number_false_positive, number_true_negative, number_false_negative


# thresholdPrediction = 0, as computed directly without taking sigmoid, else 0.5
class BinaryAccuracyEdges(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy_edges", threshold_prediction=0, **kwargs):
        super(BinaryAccuracyEdges, self).__init__(name=name, **kwargs)
        self.numberTruePredictedPixel = self.add_weight(name="true", initializer="zeros")
        self.numberPixel = self.add_weight(name="num", initializer="zeros")
        self.thresholdPrediction = threshold_prediction
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold_prediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold_prediction, tf.int32)
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        range_classes = tf.range(1, y_pred.shape[-1] + 1)
        range_classes_reshape = tf.reshape(range_classes, [1, 1, 1, y_pred.shape[-1]])
        y_true = tf.cast(range_classes_reshape == y_true, dtype=tf.int32)
        accuracy = tf.cast(tf.reduce_sum(tf.cast(y_true == y_pred, dtype=tf.int32), axis=-1) == y_pred.shape[-1],
                           dtype=tf.float32)
        self.numberTruePredictedPixel.assign_add(tf.reduce_sum(accuracy))
        shape = tf.cast(tf.shape(y_true), tf.float32)
        self.numberPixel.assign_add(shape[0] * shape[1] * shape[2])
    
    @tf.function
    def result(self):
        return self.numberTruePredictedPixel / self.numberPixel
    
    @tf.function
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.numberTruePredictedPixel.assign(0.0)
        self.numberPixel.assign(0.0)
    
    def get_config(self):
        base_config = super().get_config()
        base_config['threshold_prediction'] = self.thresholdPrediction
        return base_config


# class F1Edges(tf.keras.metrics.Metric):
#     def __init__(self, name="f1_edges", threshold_prediction=0, threshold_edge_width=0, **kwargs):
#         super(F1Edges, self).__init__(name=name, **kwargs)
#         self.numberTruePositive = self.add_weight(name="numberTruePositive", initializer="zeros")
#         self.numberFalsePositive = self.add_weight(name="numberFalsePositive", initializer="zeros")
#         self.numberTrueNegative = self.add_weight(name="numberTrueNegative", initializer="zeros")
#         self.numberFalseNegative = self.add_weight(name="numberFalseNegative", initializer="zeros")
#         self.thresholdPrediction = threshold_prediction
#         self.thresholdEdgeWidth = threshold_edge_width
#
#     @tf.function
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.thresholdPrediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
#         y_pred = tf.cast(y_pred > self.thresholdPrediction, tf.int32)
#
#         # reshape y_true: channels = number of classes and binary classification of edge and nonedge
#         y_true = tf.cast(y_true, dtype=tf.int32)
#         class_range = tf.range(1, y_pred.shape[-1] + 1)
#         class_range_reshape = tf.reshape(class_range, [1, 1, 1, y_pred.shape[-1]])
#         y_true = tf.cast(class_range_reshape == y_true, dtype=tf.int32)
#
#         number_true_positive, number_false_positive, number_true_negative, number_false_negative = number_true_false_positive_negative(
#             y_true, y_pred, self.thresholdEdgeWidth)
#
#         self.numberTruePositive.assign_add(tf.cast(tf.reduce_sum(number_true_positive), tf.float32))
#         self.numberFalsePositive.assign_add(tf.cast(tf.reduce_sum(number_false_positive), tf.float32))
#         self.numberTrueNegative.assign_add(tf.cast(tf.reduce_sum(number_true_negative), tf.float32))
#         self.numberFalseNegative.assign_add(tf.cast(tf.reduce_sum(number_false_negative), tf.float32))
#
#     @tf.function
#     def result(self):
#         precision = tf.where(self.numberTruePositive + self.numberFalsePositive != 0,
#                              self.numberTruePositive / (self.numberTruePositive + self.numberFalsePositive), 1)
#         recall = tf.where(self.numberTruePositive + self.numberFalseNegative != 0,
#                           self.numberTruePositive / (self.numberTruePositive + self.numberFalseNegative), 1)
#         f1 = tf.where(precision + recall != 0, 2 * precision * recall / (precision + recall), 0)
#         return {"f1": f1, "precision": precision, "recall": recall}
#
#     @tf.function
#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.numberTruePositive.assign(0.0)
#         self.numberFalsePositive.assign(0.0)
#         self.numberTrueNegative.assign(0.0)
#         self.numberFalseNegative.assign(0.0)

class F1Edges(tf.keras.metrics.Metric):
    def __init__(self, num_classes, classes_individually=False, threshold_prediction=0,
                 threshold_edge_width=0, name="f1_edges", **kwargs):
        super(F1Edges, self).__init__(name=name, **kwargs)
        if classes_individually:
            self.dimension = num_classes
        else:
            self.dimension = 1
        self.numberTruePositive = self.add_weight(name="numberTruePositive", initializer="zeros",
                                                  shape=(self.dimension,))
        self.numberFalsePositive = self.add_weight(name="numberFalsePositive", initializer="zeros",
                                                   shape=(self.dimension,))
        self.numberTrueNegative = self.add_weight(name="numberTrueNegative", initializer="zeros",
                                                  shape=(self.dimension,))
        self.numberFalseNegative = self.add_weight(name="numberFalseNegative", initializer="zeros",
                                                   shape=(self.dimension,))
        self.thresholdPrediction = threshold_prediction
        self.thresholdEdgeWidth = threshold_edge_width
        self.classes_individually = classes_individually
        self.num_classes = num_classes
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold_prediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold_prediction, tf.int32)
        
        # reshape y_true: channels = number of classes and binary classification of edge and nonedge
        y_true = tf.cast(y_true, dtype=tf.int32)
        class_range = tf.range(1, y_pred.shape[-1] + 1)
        class_range_reshape = tf.reshape(class_range, [1, 1, 1, y_pred.shape[-1]])
        y_true = tf.cast(class_range_reshape == y_true, dtype=tf.int32)
        
        number_true_positive, number_false_positive, number_true_negative, number_false_negative = number_true_false_positive_negative(
            y_true, y_pred, self.thresholdEdgeWidth)
        
        if self.dimension == 1:
            self.numberTruePositive.assign_add(tf.cast(tf.reduce_sum(number_true_positive, keepdims=True), tf.float32))
            self.numberFalsePositive.assign_add(
                tf.cast(tf.reduce_sum(number_false_positive, keepdims=True), tf.float32))
            self.numberTrueNegative.assign_add(tf.cast(tf.reduce_sum(number_true_negative, keepdims=True), tf.float32))
            self.numberFalseNegative.assign_add(
                tf.cast(tf.reduce_sum(number_false_negative, keepdims=True), tf.float32))
        else:
            self.numberTruePositive.assign_add(tf.cast(number_true_positive, tf.float32))
            self.numberFalsePositive.assign_add(tf.cast(number_false_positive, tf.float32))
            self.numberTrueNegative.assign_add(tf.cast(number_true_negative, tf.float32))
            self.numberFalseNegative.assign_add(tf.cast(number_false_negative, tf.float32))
    
    @tf.function
    def result(self):
        precision = tf.where(self.numberTruePositive + self.numberFalsePositive != 0,
                             self.numberTruePositive / (self.numberTruePositive + self.numberFalsePositive), 1)
        recall = tf.where(self.numberTruePositive + self.numberFalseNegative != 0,
                          self.numberTruePositive / (self.numberTruePositive + self.numberFalseNegative), 1)
        f1 = tf.where(precision + recall != 0, 2 * precision * recall / (precision + recall), 0)
        
        # metrics = ["f1", "precision", "recall"]
        metric_dict = dict()
        metric_dict["f1"] = tf.reduce_mean(f1)
        metric_dict["precision"] = tf.reduce_mean(precision)
        metric_dict["recall"] = tf.reduce_mean(recall)
        if self.dimension > 1:
            for i in range(1, self.dimension + 1):
                metric_dict["f1" + "_{}".format(i)] = f1[i - 1]
                metric_dict["precision" + "_{}".format(i)] = precision[i - 1]
                metric_dict["recall" + "_{}".format(i)] = recall[i - 1]
        return metric_dict
    
    @tf.function
    def reset_state(self):
        tensor_for_resetting = tf.zeros((self.dimension,))
        self.numberTruePositive.assign(tensor_for_resetting)
        self.numberFalsePositive.assign(tensor_for_resetting)
        self.numberTrueNegative.assign(tensor_for_resetting)
        self.numberFalseNegative.assign(tensor_for_resetting)
    
    def get_config(self):
        base_config = super().get_config()
        base_config['num_classes'] = self.num_classes
        base_config['threshold_prediction'] = self.thresholdPrediction
        base_config['threshold_edge_width'] = self.thresholdEdgeWidth
        base_config['classes_individually'] = self.classes_individually
        return base_config
        
