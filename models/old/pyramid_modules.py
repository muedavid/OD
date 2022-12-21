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