import tensorflow as tf


class FlowLoss(tf.keras.losses.Loss):
    def __init__(self, large_loss_threshold=4.0, name='flow_loss'):
        super().__init__(name=name)
        self.large_loss_threshold = large_loss_threshold
    
    @tf.function
    def call(self, y_true, y_pred):
        mult = 2.0
        
        large_loss = tf.where(tf.abs(y_true - y_pred) >= self.large_loss_threshold, 1.0, 0.0)
        large_loss = tf.cast(large_loss, tf.float32)
        
        y_true_mask = tf.where(y_true != 0, 1, 0)
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
