import tensorflow as tf
import argparse
import yaml
import numpy as np

def predict_class_postprocessing(prediction, threshold=0.5):
    predictions = tf.math.sigmoid(prediction)
    if isinstance(threshold, list):
        threshold_dim = len(threshold)
        threshold = tf.cast(tf.convert_to_tensor(threshold), tf.float32)
        threshold_mult = 1 / tf.convert_to_tensor(threshold)
        threshold_mult = tf.reshape(threshold_mult, shape=[1, 1, 1, threshold_dim])
    else:
        threshold_mult = 1 / threshold

    value_larger_threshold = tf.where(predictions * threshold_mult >= 1.0, predictions, 0.0)
    max_idx = tf.cast(tf.argmax(value_larger_threshold, axis=-1) + 1, tf.int32)
    predictions = tf.where(tf.reduce_sum(value_larger_threshold, axis=-1) > 0.005, max_idx, 0)

    predictions = tf.expand_dims(predictions, axis=-1)

    return predictions

@tf.function
def mask_pixels_for_computation(y_true):
    # TODO (add those values to the config file)
    dilation_size = 1
    y_true = tf.cast(y_true, tf.float32)
    weight_edges = tf.cast(tf.where(tf.reduce_sum(y_true, keepdims=True, axis=-1) > 0, 1.0, 0.0), tf.float32)
    weight_edges_ones = tf.ones(shape=(1, y_true.shape[1], y_true.shape[2], 1))
    filter_dilation = tf.zeros(shape=(2*dilation_size+1, 2*dilation_size+1, 1))
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
    
    return weight_border * weight_edges

def squeeze_labels_to_single_dimension(mask):
    pad = tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]])
    mask = tf.pad(mask, pad, "CONSTANT")
    mask = tf.math.argmax(mask, axis=-1, output_type=tf.int32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask

def parser(cfg, cfg_data):

    file_name = None
    try:
        file_name = __file__
    except:
        print("Jupyter Notebook")

    if cfg['PARSER'] and file_name is not None:

        p = argparse.ArgumentParser()

        p.add_argument('--model', type=str, required=False, default=None)
        p.add_argument('--data', type=str, required=False, default=None)

        p.add_argument('--epoch', type=int, required=False, default=None)

        p.add_argument('--train_model', action='store_true', default=None)
        p.add_argument('--save', action='store_true', default=None)

        p.add_argument('--sigmoid', action='store_true', default=None)
        p.add_argument('--focal', action='store_true', default=None)
        args = p.parse_args()

        print(args.save)

        # Reset all configs
        cfg['NAME'] = cfg['NAME'] if args.model is None else args.model
        cfg_data['NAME'] = cfg_data['NAME'] if args.data is None else args.data
        cfg['EPOCHS'] = cfg['EPOCHS'] if args.epoch is None else args.epoch
        cfg['TRAIN_MODEL'] = cfg['TRAIN_MODEL'] if args.train_model is None else args.train_model
        cfg['SAVE'] = cfg['SAVE'] if args.save is None else args.save
        cfg['LOSS']['SIGMOID'] = cfg['LOSS']['FOCAL'] if args.sigmoid is None else args.sigmoid
        cfg['LOSS']['FOCAL'] = cfg['LOSS']['FOCAL'] if args.focal is None else args.focal

    return cfg


def config_loader(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    file.close()

    return cfg