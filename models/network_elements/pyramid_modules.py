import tensorflow as tf
from models.network_elements import utils


def viot_coarse_features_no_prior(input_layer, num_classes, num_filters_per_class, output_shape):
    num_filters = num_classes * num_filters_per_class
    
    image_1 = utils.convolution_block(input_layer, kernel_size=3, num_filters=16)
    image_1 = utils.convolution_block(image_1, kernel_size=3, num_filters=16)
    image_1_1 = tf.keras.layers.AveragePooling2D((5, 5), strides=1, padding="SAME")(image_1)
    image_1_2 = tf.keras.layers.MaxPool2D((5, 5), strides=1, padding="SAME")(image_1)
    image_1 = tf.keras.layers.Concatenate()([image_1_1, image_1_2, image_1])
    image_1 = utils.convolution_block(image_1, kernel_size=1, num_filters=32)
    image_1 = utils.convolution_block(image_1, kernel_size=3, num_filters=8)
    image_edge = tf.image.resize(image_1, (output_shape[0], output_shape[1]), method="bilinear")
    # image_edge = tf.keras.layers.AveragePooling2D((2, 2), strides=1, padding="SAME")(image_edge)
    return image_edge


def viot_coarse_features_prior(input_layer, input_edge, num_classes, num_filters_per_class, output_shape):
    num_filters = num_classes * num_filters_per_class
    
    edge_1 = utils.convolution_block(input_edge, kernel_size=1, num_filters=6)
    edge_1 = utils.convolution_block(edge_1, kernel_size=3, num_filters=6, separable=True)
    edge_1 = utils.convolution_block(edge_1, kernel_size=3, num_filters=6, separable=True)
    
    image_1 = utils.convolution_block(input_layer, kernel_size=3, num_filters=num_filters, separable=True)
    image_1 = utils.mobile_net_v2_inverted_residual(image_1, depth_multiplier=6)
    image_1 = tf.keras.layers.Concatenate()([image_1, edge_1])
    image_1 = utils.convolution_block(image_1, kernel_size=1, num_filters=2 * num_filters)
    image_1 = utils.convolution_block(image_1, separable=True, kernel_size=3, num_filters=3 * num_filters)
    image_1 = utils.convolution_block(image_1, separable=True, kernel_size=3, num_filters=3 * num_filters)
    image_1 = utils.convolution_block(image_1, separable=True, kernel_size=1, num_filters=3 * num_filters)
    image_1 = utils.convolution_block(image_1, separable=True, kernel_size=1, num_filters=num_filters)
    # image_1 = utils.convolution_block(image_1, separable=True, kernel_size=1, num_filters=num_filters)
    # image_1 = tf.keras.layers.Activation(activation="hard_sigmoid")(image_1)
    image_1 = tf.image.resize(image_1, (output_shape[0], output_shape[1]), method="bilinear")
    return image_1


def viot_coarse_features_no_prior_dilation(daspp_input, num_classes, num_filters_per_class, output_shape):
    num_filters = num_classes*num_filters_per_class
    
    dims = daspp_input.shape
    
    out_1 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                    RELU=True, name="daspp_1")
    out_2 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=2, BN=True,
                                    RELU=True, name="daspp_2")
    out_3 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=4, BN=True,
                                    RELU=True, name="daspp_3")
    out = tf.keras.layers.AveragePooling2D(pool_size=(int(dims[2] / 2), int(dims[2] / 2)), strides=int(dims[2] / 4))(
        daspp_input)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="daspp_avg")
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3])
    
    x = utils.convolution_block(x, kernel_size=1, name="daspp_out", num_filters=num_filters)
    
    x = tf.keras.layers.Activation(activation="hard_sigmoid")(x)
    x = tf.image.resize(x, (output_shape[0], output_shape[1]), method="bilinear")
    
    return x


def flow_edge(image_layer, edge_layer):
    edge_width_height = int(edge_layer.shape[1] / 20) + 1
    edge_width_width = int(edge_layer.shape[2] / 12) + 1
    ones = tf.ones(shape=(edge_width_height, edge_width_width, 1, 1))
    edge_down = tf.nn.conv2d(edge_layer, ones, strides=[1, 1, 1, 1], padding="SAME")
    edge_down = tf.where(edge_down >= 1, 1, 0)
    edge_down = tf.cast(tf.image.resize(edge_down, (20, 12), method="nearest"), tf.float32)
    
    image_layer = utils.convolution_block(image_layer, name="image_1", kernel_size=3, strides=1, num_filters=12)
    image_layer = utils.convolution_block(image_layer, name="image_2", kernel_size=3, strides=1, num_filters=6)
    image_layer_out = utils.convolution_block(image_layer, name="image_out_1", kernel_size=3, num_filters=6)
    image_layer_out = tf.keras.layers.Concatenate(axis=-1)([image_layer_out, edge_layer])
    image_layer_out = utils.convolution_block(image_layer_out, name="image_out_2", kernel_size=5, num_filters=6,
                                              separable=True)
    image_layer_out = utils.convolution_block(image_layer_out, name="image_out_3", kernel_size=3, num_filters=2,
                                              RELU=False, BN=False)
    image_layer = utils.convolution_block(image_layer, name="image_3", kernel_size=3, strides=1, num_filters=1,
                                          RELU=False, BN=False)
    image_layer_sig = tf.keras.layers.Activation(activation="sigmoid")(image_layer)
    
    dim = 3
    shift_up = tf.constant([0, 0, 1])
    shift_down = tf.constant([1, 0, 0])
    shift_left = tf.constant([0, 0, 1])
    shift_right = tf.constant([1, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [dim, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [dim, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, dim, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, dim, 1, 1]), tf.float32)
    shift_direction_to_filter = {"up": shift_up, "down": shift_down, "left": shift_left, "right": shift_right}
    shift_pattern = {"up": ["up", "left", "right"],
                     "down": ["down", "left", "right"],
                     "left": ["left"],
                     "right": ["right"]}
    
    shifted = [image_layer_sig]
    for shift_direction_1 in shift_pattern.keys():
        conv_filter = shift_direction_to_filter[shift_direction_1]
        image_shifted_1 = tf.nn.conv2d(image_layer_sig, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
        shifted.append(image_shifted_1)
        for shift_direction_2 in shift_pattern[shift_direction_1]:
            conv_filter = shift_direction_to_filter[shift_direction_2]
            image_shifted_2 = tf.nn.conv2d(image_shifted_1, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
            shifted.append(image_shifted_2)
            for shift_direction_3 in shift_pattern[shift_direction_2]:
                conv_filter = shift_direction_to_filter[shift_direction_3]
                image_shifted_3 = tf.nn.conv2d(image_shifted_2, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
                shifted.append(image_shifted_3)
                for shift_direction_4 in shift_pattern[shift_direction_3]:
                    conv_filter = shift_direction_to_filter[shift_direction_4]
                    image_shifted_4 = tf.nn.conv2d(image_shifted_2, conv_filter, strides=[1, 1, 1, 1], padding="SAME")
                    shifted.append(image_shifted_4)
    shifted_concat = tf.keras.layers.Concatenate(axis=-1)(shifted)
    correlation = shifted_concat * edge_layer
    flow_max = tf.keras.layers.MaxPool2D((4, 4), strides=2, padding="SAME")(correlation)
    flow_avg = tf.keras.layers.AveragePooling2D((4, 4), strides=2)(flow_max)
    flow_max = tf.keras.layers.MaxPool2D((2, 2), strides=1, padding="SAME")(flow_avg)
    flow_avg = tf.keras.layers.AveragePooling2D((2, 2), strides=1)(flow_max)
    flow_max_norm = tf.keras.layers.LayerNormalization(axis=-1)(flow_avg)
    flow_max = tf.keras.layers.Concatenate(axis=-1)([flow_avg, flow_max_norm])
    flow = utils.convolution_block(flow_max, num_filters=len(shifted), strides=1, kernel_size=3,
                                   name="flow_1", padding="VALID", BN=False)
    flow = utils.convolution_block(flow, num_filters=len(shifted), strides=1, kernel_size=3, depthwise=True,
                                   name="flow_2", BN=False, padding="VALID")
    flow = utils.convolution_block(flow, num_filters=len(shifted), strides=1, kernel_size=3, depthwise=True,
                                   name="flow_3", BN=False, padding="VALID")
    flow_o = tf.keras.layers.LayerNormalization(axis=-1)(flow)
    flow_o = tf.keras.layers.Concatenate(axis=-1)([flow, flow_o])
    flow = utils.convolution_block(flow_o, num_filters=len(shifted), strides=1, kernel_size=1,
                                   name="flow_4", BN=False, padding="VALID")
    flow = utils.convolution_block(flow, num_filters=len(shifted), strides=1, kernel_size=3, depthwise=True,
                                   name="flow_5", BN=False, padding="VALID")
    flow = utils.convolution_block(flow, num_filters=len(shifted), strides=1, kernel_size=1,
                                   name="flow_7", BN=False, padding="VALID")
    flow_o1 = utils.convolution_block(flow, num_filters=len(shifted), strides=1, kernel_size=3,
                                      name="flow_8", BN=False, padding="VALID")
    flow_x = utils.convolution_block(flow_o1, num_filters=8, strides=1, kernel_size=1,
                                     name="flow_9", BN=False, padding="VALID")
    flow_x = utils.convolution_block(flow_x, num_filters=1, strides=1, kernel_size=1, separable=True,
                                     name="flow_10", BN=False, padding="VALID", RELU=False)
    flow_y = utils.convolution_block(flow_o1, num_filters=8, strides=1, kernel_size=1, separable=True,
                                     name="flow_11", BN=False, padding="VALID")
    flow_y = utils.convolution_block(flow_y, num_filters=1, strides=1, kernel_size=1, separable=True,
                                     name="flow_12", BN=False, padding="VALID", RELU=False)
    flow = tf.keras.layers.Concatenate(axis=-1)([flow_x, flow_y])
    flow = tf.image.resize(flow, (20, 12), method="bilinear")
    flow_out = tf.keras.layers.Multiply(name="out_flow")([flow, edge_down])
    
    img_out = tf.keras.layers.Concatenate(axis=-1)([image_layer, image_layer_out])
    img_out = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(img_out)
    img_out = tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(img_out)
    return flow_out, img_out, flow_o, flow_o1


def daspp(daspp_input, num_filters=20):
    dims = daspp_input.shape
    
    out_1 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                    separable=True, RELU=True, name="daspp_1")
    out_3 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=3,
                                    separable=True, BN=True, RELU=True, name="daspp_3_dilated")
    out_3 = utils.convolution_block(out_3, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True,
                                    BN=True, RELU=True, name="daspp_3_conv")
    out_6 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=6,
                                    separable=True, BN=True, RELU=True, name="daspp_6_dilated")
    out_6 = utils.convolution_block(out_6, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True,
                                    BN=True, RELU=True, name="daspp_6_conv")
    out_9 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=9,
                                    separable=True, BN=True, RELU=True, name="daspp_9_dilated")
    out_9 = utils.convolution_block(out_9, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True,
                                    BN=True, RELU=True, name="daspp_9_conv")
    
    out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(daspp_input)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                  name="daspp_avg")
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    
    daspp_output = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_3, out_6, out_9, daspp_input])
    
    daspp_output = utils.convolution_block(daspp_output, kernel_size=1, name="daspp_out")
    
    return daspp_output

def JPU(input_1, input_2, input_3, output_shape):
    
    conv_1 = utils.convolution_block(input_1, num_filters=input_1.shape[-1])
    conv_2 = utils.convolution_block(input_2, num_filters=input_2.shape[-1])
    conv_3 = utils.convolution_block(input_3, num_filters=input_3.shape[-1])
    
    conv_1 = tf.image.resize(conv_1, output_shape)
    conv_2 = tf.image.resize(conv_2, output_shape)
    conv_3 = tf.image.resize(conv_3, output_shape)
    
    conv = tf.keras.layers.Concatenate()([conv_1, conv_2, conv_3])
    
    conv_d_1 = utils.convolution_block(conv, num_filters=int(conv_1.shape[-1]/4), dilation_rate=1, separable=True)
    conv_d_2 = utils.convolution_block(conv, num_filters=int(conv_1.shape[-1] / 4), dilation_rate=2, separable=True)
    conv_d_4 = utils.convolution_block(conv, num_filters=int(conv_1.shape[-1] / 4), dilation_rate=4, separable=True)
    conv_d_8 = utils.convolution_block(conv, num_filters=int(conv_1.shape[-1] / 4), dilation_rate=8, separable=True)
    
    output = tf.keras.layers.Concatenate()([conv_d_1, conv_d_2, conv_d_4, conv_d_8])
    
    output = utils.convolution_block(output, num_filters=1)
    
    return output
    
    
