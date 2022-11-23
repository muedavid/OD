import tensorflow as tf


@tf.function
def compute_f1_precision_recall(true_positive, false_positive, false_negative):
    precision = tf.where(true_positive + false_positive != 0, true_positive / (true_positive + false_positive), 0)
    recall = tf.where((true_positive + false_negative) != 0, true_positive / (true_positive + false_negative), 0)
    f1 = tf.where(precision + recall != 0, 2 * precision * recall / (precision + recall), 0)
    
    return f1, precision, recall


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
    def __init__(self, num_classes, classes_individually=False, name="accuracy_edges", threshold_prediction=0, **kwargs):
        super(BinaryAccuracyEdges, self).__init__(name=name, **kwargs)
        self.numberTruePredictedPixels = self.add_weight(name="numberTruePredictedPixels", initializer="zeros",
                                                         shape=(num_classes,))
        self.numberPixels = self.add_weight(name="numberPixels", initializer="zeros")
        self.thresholdPrediction = threshold_prediction
        self.num_classes = num_classes
        self.classes_individually = classes_individually
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold_prediction = tf.cast(self.thresholdPrediction, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold_prediction, tf.int32)
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        range_classes = tf.range(1, y_pred.shape[-1] + 1)
        range_classes_reshape = tf.reshape(range_classes, [1, 1, 1, y_pred.shape[-1]])
        y_true = tf.cast(range_classes_reshape == y_true, dtype=tf.int32)
        
        number_true_predicted = tf.cast(tf.reduce_sum(tf.cast(y_true == y_pred, dtype=tf.int32),
                                                      axis=[0, 1, 2]), dtype=tf.float32)
        
        self.numberTruePredictedPixels.assign_add(number_true_predicted)
        shape = tf.cast(tf.shape(y_true), tf.float32)
        self.numberPixels.assign_add(shape[0] * shape[1] * shape[2])
    
    @tf.function
    def result(self):
        accuracy = self.numberTruePredictedPixels / self.numberPixels
        metric_dict = {'accuracy': tf.reduce_mean(accuracy)}
        if self.classes_individually:
            for i in range(1, self.num_classes + 1):
                metric_dict['accuracy' + "_{}".format(i)] = accuracy[i - 1]
    
        return metric_dict
    
    @tf.function
    def reset_state(self):
        
        # The state of the metric will be reset at the start of each epoch.
        self.numberTruePredictedPixels.assign(tf.zeros((self.num_classes,)))
        self.numberPixels.assign(0.0)
    
    def get_config(self):
        base_config = super().get_config()
        base_config['threshold_prediction'] = self.thresholdPrediction
        base_config['num_classes'] = self.num_classes
        base_config['classes_individually'] = self.classes_individually
        return base_config


class F1Edges(tf.keras.metrics.Metric):
    def __init__(self, num_classes, classes_individually=False, threshold_prediction=0,
                 threshold_edge_width=0, name="f1_edges", **kwargs):
        super(F1Edges, self).__init__(name=name, **kwargs)
        
        self.thresholdPrediction = threshold_prediction
        self.thresholdEdgeWidth = threshold_edge_width
        self.classes_individually = classes_individually
        self.num_classes = num_classes
        
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
        
        # reshape y_true: channels = number of classes and binary classification of edge and nonedge
        y_true = tf.cast(y_true, dtype=tf.int32)
        class_range = tf.range(1, y_pred.shape[-1] + 1)
        class_range_reshape = tf.reshape(class_range, [1, 1, 1, y_pred.shape[-1]])
        y_true = tf.cast(class_range_reshape == y_true, dtype=tf.int32)
        
        number_true_positive, number_false_positive, number_true_negative, number_false_negative = number_true_false_positive_negative(
            y_true, y_pred, self.thresholdEdgeWidth)
        
        self.numberTruePositive.assign_add(tf.cast(number_true_positive, tf.float32))
        self.numberFalsePositive.assign_add(tf.cast(number_false_positive, tf.float32))
        self.numberTrueNegative.assign_add(tf.cast(number_true_negative, tf.float32))
        self.numberFalseNegative.assign_add(tf.cast(number_false_negative, tf.float32))
    
    @tf.function
    def result(self):
        mean_scores = compute_f1_precision_recall(tf.reduce_sum(self.numberTruePositive, keepdims=True),
                                                  tf.reduce_sum(self.numberFalsePositive, keepdims=True),
                                                  tf.reduce_sum(self.numberFalseNegative, keepdims=True))
        
        metric_dict = {"f1": mean_scores[0][0], "precision": mean_scores[1][0], "recall": mean_scores[2][0]}
        
        if self.classes_individually:
            f1, precision, recall = compute_f1_precision_recall(self.numberTruePositive, self.numberFalsePositive,
                                                                self.numberFalseNegative)
            for i in range(1, self.num_classes + 1):
                metric_dict["f1" + "_{}".format(i)] = f1[i - 1]
                metric_dict["precision" + "_{}".format(i)] = precision[i - 1]
                metric_dict["recall" + "_{}".format(i)] = recall[i - 1]
        
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
        base_config['threshold_prediction'] = self.thresholdPrediction
        base_config['threshold_edge_width'] = self.thresholdEdgeWidth
        base_config['classes_individually'] = self.classes_individually
        return base_config
