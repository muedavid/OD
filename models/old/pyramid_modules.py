import tensorflow as tf
from models.network_elements import utils


def flow_image(image_layer_1, image_layer_2):
    image_layer_1 = tf.expand_dims(image_layer_1, axis=-1)
    dim = 3
    shift_up = tf.constant([0, 0, 1])
    shift_down = tf.constant([1, 0, 0])
    shift_left = tf.constant([0, 0, 1])
    shift_right = tf.constant([1, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [1, dim, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [1, dim, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, 1, dim, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, 1, dim, 1, 1]), tf.float32)
    shift_direction_to_filter = {"up": shift_up, "down": shift_down, "left": shift_left, "right": shift_right}
    shift_pattern = {"up": ["up", "left", "right"],
                     "down": ["down", "left", "right"],
                     "left": ["left"],
                     "right": ["right"]}
    
    shifted = [image_layer_1]
    for shift_direction in shift_pattern.keys():
        conv_filter = shift_direction_to_filter[shift_direction]
        image_shifted = tf.nn.conv3d(image_layer_1, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME")
        shifted.append(image_shifted)
        for shift_direction_second in shift_pattern[shift_direction]:
            conv_filter = shift_direction_to_filter[shift_direction_second]
            shifted.append(tf.nn.conv3d(image_shifted, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME"))
    o = tf.keras.layers.Concatenate(axis=-1)(shifted)
    out = []
    for shifted_image_1 in shifted:
        correlation = shifted_image_1[:, :, :, :, 0] * image_layer_2
        mixed = tf.keras.layers.Conv2D(filters=10, activation="relu", kernel_size=1)(correlation)
        mixed = tf.keras.layers.Conv2D(filters=10, activation="relu", kernel_size=3)(mixed)
        mixed = tf.keras.layers.Conv2D(filters=2, activation="relu", kernel_size=3)(mixed)
        correlation = tf.keras.layers.AvgPool2D(pool_size=(5, 5), strides=3)(mixed)
        out.append(correlation)
    
    out = tf.keras.layers.Concatenate(axis=-1)(out)
    out = utils.convolution_block(out, BN=False, kernel_size=1, num_filters=20, name="out_1")
    out = utils.convolution_block(out, BN=False, kernel_size=1, num_filters=20, name="out_2")
    out = utils.convolution_block(out, BN=False, kernel_size=1, num_filters=8, name="out_3")
    out = utils.convolution_block(out, BN=False, kernel_size=3, num_filters=8, name="out_4")
    out = utils.convolution_block(out, BN=False, kernel_size=5, num_filters=4, name="out_5")
    out = utils.convolution_block(out, BN=False, kernel_size=7, num_filters=4, name="out_6")
    out = tf.image.resize(out, (80, 45))
    out = tf.keras.layers.Conv2D(filters=2, kernel_size=1, name="out_flow")(out)
    
    return out


def edge_shifted_params(image_layer, edge_layer):
    small_filter_num = 10
    
    image_layer_f = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                            strides=2, separable=True,
                                            name="pyramid_module_image_down_sampling_1_1")
    image_layer_f = utils.convolution_block(image_layer_f, num_filters=small_filter_num, kernel_size=3,
                                            strides=1, separable=True,
                                            name="pyramid_module_image_down_sampling_1_2")
    image_layer_f = utils.convolution_block(image_layer_f, num_filters=small_filter_num, kernel_size=3,
                                            strides=1, separable=True,
                                            name="pyramid_module_image_down_sampling_1_3")
    
    image_edge = tf.keras.layers.Concatenate(axis=-1)([image_layer_f, edge_layer])
    image_edge = utils.convolution_block(image_edge, num_filters=5, kernel_size=1, name="image_edge_1")
    image_edge = utils.convolution_block(image_edge, num_filters=5, kernel_size=3, strides=2, name="image_edge_3")
    image_edge = utils.convolution_block(image_edge, num_filters=5, kernel_size=3, name="image_edge_4")
    image_edge = utils.convolution_block(image_edge, num_filters=5, kernel_size=3, strides=2, name="image_edge_5")
    image_edge = utils.convolution_block(image_edge, num_filters=5, kernel_size=3, name="image_edge_6")
    dense = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(image_edge)
    dense = tf.keras.layers.Flatten()(dense)
    dense = tf.keras.layers.Dense(50, activation="relu")(dense)
    dense = tf.keras.layers.Dense(10, activation="relu")(dense)
    dense_x = tf.keras.layers.Dense(1, activation="relu")(dense)
    dense_y = tf.keras.layers.Dense(1, activation="relu")(dense)
    dense_x = tf.keras.layers.Dense(21, activation="relu")(dense_x)
    dense_y = tf.keras.layers.Dense(21, activation="relu")(dense_y)
    filter_y = tf.reshape(dense_y, [1, 21, 1, 1])
    filter_x = tf.reshape(dense_x, [1, 21, 1, 1])
    output_edge = tf.nn.conv2d(edge_layer, filter_x, strides=[1, 1, 1, 1], padding="SAME")
    output_edge = tf.nn.conv2d(output_edge, filter_y, strides=[1, 1, 1, 1], padding="SAME")
    output_edge = tf.keras.layers.Activation(name="out_edge", activation="sigmoid")(output_edge)
    
    return output_edge, dense_x, dense_y


def edge_shifted(image_layer, edge_layer):
    filter_num = 10
    
    down_sampling = int(image_layer.shape[1] / edge_layer.shape[1])
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=filter_num, kernel_size=down_sampling + 1,
                                              strides=down_sampling,
                                              separable=True, name="pyramid_module_image_down_sampling")
    image_layer = utils.convolution_block(image_layer, num_filters=filter_num, kernel_size=3, name="image_processing_1")
    image_layer = utils.convolution_block(image_layer, num_filters=filter_num, kernel_size=1, name="image_processing_2")
    image_layer = utils.convolution_block(image_layer, num_filters=filter_num, kernel_size=1, name="image_mult",
                                          RELU=False, BN=False)
    
    mult = tf.keras.layers.Concatenate(axis=-1)([edge_layer, image_layer])
    mult = utils.convolution_block(mult, num_filters=25, kernel_size=1, name="mult_1", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=25, kernel_size=1, name="mult_2", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=25, kernel_size=1, name="mult_2_1", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=8, kernel_size=1, name="mult_2_3", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=8, kernel_size=3, name="mult_3", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=8, kernel_size=5, name="mult_4", RELU=True,
                                   BN=False)
    mult = utils.convolution_block(mult, num_filters=4, kernel_size=7, name="mult_5", RELU=True,
                                   BN=False, separable=True)
    mult = utils.convolution_block(mult, num_filters=4, kernel_size=14, name="mult_6", RELU=True,
                                   BN=False, separable=True)
    mult = utils.convolution_block(mult, num_filters=4, kernel_size=21, name="mult_7", RELU=True,
                                   BN=False, separable=True)
    out = tf.keras.layers.Conv2D(filters=2, kernel_size=1, name="out_flow", padding="SAME")(mult)
    
    return out
