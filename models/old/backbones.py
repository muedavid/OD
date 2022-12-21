import tensorflow as tf
from numpy import log2
from models.network_elements import utils


def edge_map_preprocessing_shifted(input_layer):
    edge = tf.expand_dims(input_layer, axis=-1)
    dim = 15
    shift_up = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_down = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_left = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_right = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [1, dim, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [1, dim, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, 1, dim, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, 1, dim, 1, 1]), tf.float32)
    shift_direction_to_filter = {"up": shift_up, "down": shift_down, "left": shift_left, "right": shift_right}
    shift_pattern = {"up": ["up", "left", "right"],
                     "down": ["down", "left", "right"],
                     "left": ["left"],
                     "right": ["right"]}
    
    shifted = [edge]
    for shift_direction_first, following_shift_direction in shift_pattern.items():
        conv_filter = shift_direction_to_filter[shift_direction_first]
        edge_shifted = tf.nn.conv3d(edge, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME")
        shifted.append(edge_shifted)
        for shift_direction_second in following_shift_direction:
            conv_filter = shift_direction_to_filter[shift_direction_second]
            shifted.append(tf.nn.conv3d(edge_shifted, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME"))
    
    edge_shifted = tf.keras.layers.Concatenate(axis=-1, name="concat_shifted_filter")(shifted)
    
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=1, padding="same", use_bias=True, activation="relu")(
        edge_shifted)
    edge_shifted = tf.keras.layers.BatchNormalization(name='edge_input_3_bn')(edge_shifted)
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=3, padding="same", use_bias=True,
                                          activation="relu")(
        edge_shifted)
    
    shape = tf.shape(edge_shifted)
    x = tf.reshape(edge_shifted, [shape[0], shape[1], shape[2], shape[3] * shape[4]])
    return x


def edge_map_preprocessing_3D(input_layer, image_layer, output_shape, num_classes):
    # for 3D convolution such that each channel has same filter
    edge = tf.expand_dims(input_layer, axis=-1)
    
    edge = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                  strides=1, use_bias=False, name='edge_input_1_conv')(edge)
    edge = tf.keras.layers.ReLU(name='edge_input_1_relu')(edge)
    edge = tf.keras.layers.BatchNormalization(name='edge_input_1_bn')(edge)
    edge = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                  strides=(2, 2, 1), use_bias=False, name='edge_input_2_conv')(edge)
    edge = tf.keras.layers.ReLU(name='edge_input_2_relu')(edge)
    edge = tf.keras.layers.Conv3D(3, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                                  strides=(2, 2, 1), use_bias=False, name='edge_input_3_conv')(edge)
    edge = tf.keras.layers.BatchNormalization(name='edge_input_2_bn')(edge)
    
    image_layer = utils.convolution_block(image_layer, num_filters=num_classes, strides=2, kernel_size=5,
                                          name='backbone_output_for_edge_1')
    image_layer = utils.convolution_block(image_layer, num_filters=num_classes, RELU=False,
                                          name='backbone_output_for_edge_2')
    image_layer = tf.expand_dims(image_layer, axis=-1)
    x = tf.keras.layers.Concatenate(axis=-1)([edge, image_layer])
    
    x = tf.keras.layers.Conv3D(1, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=False, name='edge_image_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_1_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_1_relu')(x)
    
    x = tf.keras.layers.Conv3D(1, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=True, name='edge_image_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_2_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_2_relu')(x)
    
    down_sampling = int(log2(x.shape[1] / output_shape[0]).tolist())
    if down_sampling % 1 != 0.0:
        raise ValueError("input shape of the edge map must be exact dividable by the output shape of the backbone")
    for i in range(down_sampling):
        x = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                   strides=(2, 2, 1), use_bias=True, name="edge_map_down_sampling_{}".format(i))(x)
    x = tf.squeeze(x, axis=-1)
    return x


def image_flow(img_shape):
    # TODO(DAVID) skip connection extremely important, as elsewhere vanishing gradients
    image_layer_1 = tf.keras.layers.Input(shape=img_shape, name="in_img")
    image_layer_2 = tf.keras.layers.Input(shape=img_shape, name="in_prior_img")
    image_concat = tf.keras.layers.Concatenate(axis=-1)([image_layer_1, image_layer_2])
    flow = utils.convolution_block(image_concat, kernel_size=11, num_filters=12, name="d_1",
                                   strides=2, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=9, num_filters=24, name="d_2",
                                   strides=2, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=7, num_filters=48, name="d_3",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=5, num_filters=96, name="d_4",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_5",
                                   strides=1, padding="VALID", BN=False)
    flow_x = utils.convolution_block(flow, kernel_size=1, num_filters=96, name="d_8",
                                     strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow_x, kernel_size=1, num_filters=96, name="d_9",
                                   strides=1, padding="VALID", BN=False)
    flow_avg = tf.keras.layers.AveragePooling2D((4, 4), strides=4)(flow)
    flow_avg = utils.convolution_block(flow_avg, kernel_size=3, num_filters=96, name="avg_1", BN=False)
    flow_avg = utils.convolution_block(flow_avg, kernel_size=3, num_filters=96, name="avg_2", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_6",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_7",
                                   strides=1, padding="VALID", BN=False)
    flow_avg = tf.image.resize(flow_avg, (flow.shape[1], flow.shape[2]))
    flow = tf.keras.layers.Concatenate(axis=-1)([flow, flow_avg])
    flow = utils.convolution_block(flow, kernel_size=7, num_filters=48, name="d_11",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=5, num_filters=24, name="d_12",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=24, name="d_13",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=1, num_filters=24, name="d_14",
                                   strides=1, padding="VALID", BN=False)
    flow_x = tf.image.resize(flow_x, (80, 45))
    out = tf.image.resize(flow, (80, 45))
    out = tf.keras.layers.Concatenate(axis=-1)([flow_x, out])
    out = utils.convolution_block(out, kernel_size=1, num_filters=24, separable=True, name="out_1", BN=False)
    out = utils.convolution_block(out, kernel_size=7, num_filters=8, separable=True, name="out_2", BN=False)
    out = tf.keras.layers.Conv2D(filters=2, kernel_size=1, name="out_flow")(out)
    return image_layer_1, image_layer_2, out


def edge_map_preprocessing(input_layer):
    num_filters = 5
    edge_map_1 = utils.convolution_block(input_layer, kernel_size=3, num_filters=num_filters, use_bias=True,
                                         name="edge_map_processing_1")
    edge_map_2 = utils.convolution_block(edge_map_1, kernel_size=3, num_filters=num_filters, use_bias=True,
                                         strides=1, name="edge_map_processing_2")
    edge_map_3 = utils.convolution_block(edge_map_2, kernel_size=5, num_filters=num_filters, use_bias=True,
                                         name="edge_map_processing_3")
    edge_map_4 = utils.convolution_block(edge_map_3, kernel_size=5, num_filters=num_filters, use_bias=True, strides=1,
                                         separable=True, name="edge_map_processing_4")
    edge_map_5 = tf.keras.layers.Concatenate(axis=-1)([edge_map_2, edge_map_4])
    edge_map_6 = utils.convolution_block(edge_map_5, kernel_size=1, num_filters=num_filters, use_bias=True, strides=1,
                                         separable=True, name="edge_map_processing_6")
    return edge_map_6


def edge_map_preprocessing_combined(input_layer):
    num_filters = 5
    edge = tf.expand_dims(input_layer, axis=-1)
    dim = 10
    shift_up = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_down = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_left = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_right = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [1, dim, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [1, dim, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, 1, dim, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, 1, dim, 1, 1]), tf.float32)
    shift_direction_to_filter = {"up": shift_up, "down": shift_down, "left": shift_left, "right": shift_right}
    shift_pattern = {"up": ["up", "left", "right"],
                     "down": ["down", "left", "right"],
                     "left": ["left"],
                     "right": ["right"]}
    
    shifted = [edge]
    for shift_direction_first, following_shift_direction in shift_pattern.items():
        conv_filter = shift_direction_to_filter[shift_direction_first]
        edge_shifted = tf.nn.conv3d(edge, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME")
        shifted.append(edge_shifted)
        for shift_direction_second in following_shift_direction:
            conv_filter = shift_direction_to_filter[shift_direction_second]
            shifted.append(tf.nn.conv3d(edge_shifted, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME"))
    
    edge_shifted = tf.keras.layers.Concatenate(axis=-1, name="concat_shifted_filter")(shifted)
    shape = tf.shape(edge_shifted)
    edge_shifted = tf.reshape(edge_shifted, [shape[0], shape[1], shape[2], shape[3] * shape[4]])
    
    edge_map_1 = utils.convolution_block(edge_shifted, kernel_size=1, num_filters=2 * num_filters, use_bias=True,
                                         separable=True, name="edge_map_processing_1")
    edge_map_2 = utils.convolution_block(edge_map_1, kernel_size=3, num_filters=num_filters, use_bias=True,
                                         separable=True, name="edge_map_processing_2")
    return edge_map_2
