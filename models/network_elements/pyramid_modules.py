import tensorflow as tf
from models.network_elements import utils


def pyramid_module_small_backbone(pyramid_input, num_classes, num_filters_per_class):
    num_filters = num_classes * num_filters_per_class
    x = utils.convolution_block(pyramid_input, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                separable=True, name="skip_1")
    x_1 = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                  separable=True, name="skip_2_1")
    x_1 = utils.convolution_block(x_1, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                  separable=True, name="skip_2_2")
    x_2 = utils.convolution_block(x, num_filters=2, kernel_size=1, strides=1,
                                  separable=True, name="skip_3_1")
    x = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2])
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, strides=1,
                                separable=True, name="skip_out")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                separable=True, name="img_downsampling")
    
    dims = x.shape
    
    out_1 = utils.convolution_block(x, num_filters=3, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_1")
    out_2 = utils.convolution_block(x, num_filters=3, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_2")
    out_3 = utils.convolution_block(x, num_filters=2, kernel_size=5, dilation_rate=1, BN=True, RELU=True,
                                    separable=True,
                                    name="pyramid_3")
    out_4 = utils.convolution_block(x, num_filters=1, kernel_size=7, dilation_rate=1, BN=True, RELU=True,
                                    separable=True,
                                    name="pyramid_4")
    out = tf.keras.layers.AveragePooling2D(pool_size=(14, 14), strides=14,
                                           padding='SAME')(x)
    out = utils.convolution_block(out, num_filters=2, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="pyramid_avg")
    out = tf.keras.activations.sigmoid(out)
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3, out_4])
    
    x = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=1, name="pyramid_out_1")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, name="pyramid_out_2")
    
    x = tf.image.resize(x, (pyramid_input.shape[1], pyramid_input.shape[2]))
    
    return x


def pyramid_module(pyramid_input, num_filters=12):
    x = utils.convolution_block(pyramid_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                separable=True, name="img_downsampling")
    
    dims = x.shape
    
    out_1 = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_1")
    out_2 = utils.convolution_block(x, num_filters=num_filters, kernel_size=5, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_2")
    out_3 = utils.convolution_block(x, num_filters=num_filters, kernel_size=7, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_3")
    out = tf.keras.layers.AveragePooling2D(pool_size=(int(dims[2] / 2), int(dims[2] / 2)), strides=int(dims[2] / 4),
                                           padding='SAME')(x)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=False,
                                  name="pyramid_avg")
    out = tf.keras.activations.sigmoid(out)
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3])
    
    x = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=3, name="pyramid_out")
    
    return x


def daspp_efficient(daspp_input, num_classes, num_filters_per_class):
    num_filters = num_classes * num_filters_per_class
    daspp_input = utils.convolution_block(daspp_input, num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                          separable=True)
    
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
    
    x = utils.convolution_block(x, kernel_size=1, name="daspp_out")
    
    return x


def concatenate_edge_and_image_mult(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = image_layer.shape[1] / edge_layer.shape[1]
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=small_filter_num,
                                              kernel_size=int(down_sampling + 1),
                                              strides=int(down_sampling),
                                              separable=True, name="image_down_sampling")
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True, name='pyramid_module_image_1_1')
    # image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3,
    #                                              separable=True,
    #                                              name='pyramid_module_image_1_2')
    # edge_layer_concat = utils.convolution_block(edge_layer, num_filters=small_filter_num, kernel_size=3, RELU=True,
    #                                             BN=True,
    #                                             name='pyramid_module_edge_1')
    
    image = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_1")
    image = utils.convolution_block(image, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_2")
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer])
    image_edge_concat = utils.convolution_block(x, num_filters=2 * small_filter_num, kernel_size=1,
                                                name="pyramid_module_pyramid_concat")
    
    pyramid = utils.convolution_block(image_edge_concat, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
                                      num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=2)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=2,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=2,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=2,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    pyramid_out_mult = utils.convolution_block(pyramid_out, name="pyramid_out_mult", num_filters=3, kernel_size=1)
    pyramid_out_concat = utils.convolution_block(pyramid_out, name="pyramid_out_concat", num_filters=3, kernel_size=1)
    image_mult = utils.convolution_block(image_layer, name="image_layer", num_filters=3, kernel_size=1)
    
    out = []
    for i in range(3):
        mult = tf.slice(pyramid_out_mult, begin=[0, 0, 0, i], size=[-1, -1, -1, 1],
                        name='pyramid_module_pyramid_out_{}'.format(i))
        out.append(tf.multiply(mult, image_mult, name='pyramid_module_multiplication_{}'.format(i)))
    
    after_mult = tf.keras.layers.Concatenate(axis=-1)(out)
    after_mult = utils.convolution_block(after_mult, num_filters=small_filter_num, BN=True, RELU=True,
                                         kernel_size=1,
                                         name="pyramid_module_after_mult_1")
    after_mult = utils.convolution_block(after_mult, num_filters=3, BN=True, RELU=True, kernel_size=3,
                                         separable=True, name="pyramid_module_after_mult_2")
    
    image = utils.convolution_block(image, kernel_size=3, num_filters=2,
                                    name="pyramid_module_image_2_3")
    out = tf.keras.layers.Concatenate(axis=-1)([after_mult, image, pyramid_out_concat])
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_2")
    
    return out, pyramid_out, edge_layer, image_layer_concat, after_mult


def concatenate_edge_and_image(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = int(image_layer.shape[1] / edge_layer.shape[1])
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=down_sampling + 1,
                                              strides=down_sampling,
                                              separable=True, name="pyramid_module_image_down_sampling")
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True,
                                                 name='pyramid_module_image_1_1')
    image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True,
                                                 name='pyramid_module_image_1_2')
    edge_layer_concat = utils.convolution_block(edge_layer, num_filters=small_filter_num, kernel_size=3, RELU=True,
                                                BN=True,
                                                name='pyramid_module_edge_1')
    
    edge = utils.convolution_block(edge_layer_concat, num_filters=small_filter_num, kernel_size=3, separable=True,
                                   name="pyramid_module_edge_2")
    image = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2")
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer_concat])
    pyramid = utils.convolution_block(x, num_filters=small_filter_num, kernel_size=1,
                                      name="pyramid_module_pyramid_concat")
    
    # pyramid = utils.convolution_block(pyramid, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
    #                                   num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=4)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=4,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=2,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=1,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    
    out = tf.keras.layers.Concatenate(axis=-1)([pyramid_out, image])
    out = utils.convolution_block(out, kernel_size=1, num_filters=2 * small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=3, num_filters=small_filter_num, name="pyramid_module_out_2")
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num,
                                  name="pyramid_module_out_3")
    
    return out, pyramid_out, edge_layer_concat, image_layer_concat


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


def viot_coarse_features(input_layer, num_classes, num_filters_per_class):
    num_filters = num_classes * num_filters_per_class
    
    image_1 = utils.convolution_block(input_layer, RELU=False, num_filters=num_filters, kernel_size=3)
    image_1 = utils.mobile_net_v2_inverted_residual(image_1, depth_multiplier=4, strides=2)
    
    image_2 = utils.convolution_block(image_1, kernel_size=1, strides=1, num_filters=6 * num_filters)
    image_3 = utils.convolution_block(image_2, kernel_size=5, depthwise=True)
    image_4 = utils.convolution_block(image_3, kernel_size=3, depthwise=True)
    image_5 = utils.convolution_block(image_4, kernel_size=1, num_filters=num_filters, RELU=False)
    
    # TODO(DAVID) Maybe nearest neighbour sampling
    out = tf.image.resize(image_5, (input_layer.shape[1], input_layer.shape[2]))
    
    return out