import tensorflow as tf
from models.network_elements import utils


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


def daspp_efficient(daspp_input, num_filters=12):
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
    
    # TODO: remove
    # d = []
    # for i in range(3):
    #     x = tf.keras.layers.Concatenate(axis=-1)(
    #         [out_pool[:, :, :, 4 * i:4 * i + 3], out_1[:, :, :, 4 * i:4 * i + 3], out_2[:, :, :, 4 * i:4 * i + 3],
    #          out_3[:, :, :, 4 * i:4 * i + 3]])
    #     x = utils.convolution_block(x, num_filters=4, kernel_size=1, name="daspp_out_{}".format(i))
    #     d.append(x)
    #
    # return d
    
    return x


def daspp(daspp_input, num_filters=12):
    dims = daspp_input.shape
    
    out_1 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                    RELU=True, name="daspp_1")
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
    out_18 = utils.convolution_block(daspp_input, kernel_size=3, dilation_rate=18, separable=True, BN=True, RELU=True,
                                     name="daspp_18_dilated")
    out_18 = utils.convolution_block(out_18, kernel_size=3, dilation_rate=1, separable=True, BN=True, RELU=True,
                                     name="daspp_18_conv")
    
    out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(daspp_input)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="daspp_avg")
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_3, out_6, out_9, out_18, daspp_input])
    
    x = utils.convolution_block(x, kernel_size=1, name="daspp_out")
    
    return x


def concatenate_edge_and_image(image_layer, edge_layer, num_classes, filter_mult):
    num_mult_filter = num_classes
    num_filters = num_classes * filter_mult
    
    img_down = utils.convolution_block(image_layer, num_filters=num_filters, kernel_size=5, dilation_rate=1, strides=3,
                                       separable=True, name='pyramid_downsampling')
    
    pyramid_1 = utils.convolution_block(img_down, num_filters=num_filters, kernel_size=5, dilation_rate=1, BN=True,
                                        RELU=True,
                                        name="pyramid_1_1")
    pyramid_1 = utils.convolution_block(pyramid_1, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                        RELU=True, name="pyramid_1_2")
    pyramid_2 = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), strides=4,
                                                 padding='SAME')(img_down)
    pyramid_2 = utils.convolution_block(pyramid_2, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                        RELU=False, name="pyramid_avg")
    pyramid_1 = tf.image.resize(pyramid_1, size=(image_layer.shape[1], image_layer.shape[2]))
    pyramid_2 = tf.image.resize(pyramid_2, size=(image_layer.shape[1], image_layer.shape[2]))
    image_layer_concat = utils.convolution_block(image_layer, num_filters=num_filters, kernel_size=3,
                                                 name='image_preprocessing_before_concatenation_1')
    image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=num_filters, kernel_size=3,
                                                 name='image_preprocessing_before_concatenation_2')
    edge_layer_concat = utils.convolution_block(edge_layer, num_filters=num_filters, kernel_size=3,
                                                name="edge_before_concatenated_1")
    edge_layer_concat = utils.convolution_block(edge_layer_concat, num_filters=num_filters, kernel_size=3,
                                                name="edge_before_concatenated_2")
    image_edge_concat = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer_concat])
    image_edge_concat = utils.convolution_block(image_edge_concat, num_filters=num_filters, kernel_size=5,
                                                name="image_edge_concatenated_1")
    image_edge_concat = utils.convolution_block(image_edge_concat, num_filters=num_filters, kernel_size=3, RELU=True,
                                                separable=True, name="image_edge_concatenated_2")
    
    image_layer_mult_1 = utils.convolution_block(image_layer, num_filters=num_filters, kernel_size=3, separable=True,
                                                 name='image_preprocessing_before_multiplication_1')
    image_layer_mult_2 = utils.convolution_block(image_layer_mult_1, num_filters=num_filters, kernel_size=3, RELU=False,
                                                 separable=True, use_bias=True,
                                                 name='image_preprocessing_before_multiplication_2')
    image_layer_mult_2 = tf.keras.layers.Activation(activation='sigmoid')(image_layer_mult_2)
    
    edge_layer_mult_1 = utils.convolution_block(edge_layer, num_filters=num_mult_filter, kernel_size=5,
                                                separable=True, name='edge_layer_mult_1', RELU=False)
    # edge_layer_mult_2 = utils.convolution_block(edge_layer_mult_1, num_filters=num_mult_filter, kernel_size=5, RELU=False,
    #                                             separable=True, name='edge_layer_mult_2')
    edge_layer_mult_1 = tf.keras.layers.Activation(activation='sigmoid', name='sigmoid_activation')(edge_layer_mult_1)
    
    out = []
    for i in range(num_mult_filter):
        edge = tf.slice(edge_layer_mult_1, begin=[0, 0, 0, i], size=[-1, -1, -1, 1],
                        name='edge_layer_{}'.format(i))
        edge = tf.multiply(image_layer_mult_2, edge, name='multiplication_{}'.format(i))
        out.append(edge)
    mult_1 = tf.keras.layers.Concatenate(axis=-1)(out)
    mult_1 = utils.convolution_block(mult_1, num_filters=num_filters, BN=True, RELU=True,
                                     name="edge_multiplication_1")
    # mult_2 = utils.convolution_block(mult_1, num_filters=num_filters, BN=True, RELU=True, separable=True,
    #                                  name="edge_multiplication_2")
    
    concat = tf.keras.layers.Concatenate(axis=-1)([mult_1, image_edge_concat, pyramid_1, pyramid_2])
    x = utils.convolution_block(concat, num_filters=num_filters, kernel_size=3, name="concat_1")
    x = utils.convolution_block(x, depthwise=True, kernel_size=3, name="concat_2")
    
    return x, image_edge_concat


def concatenate_edge_and_image_no_mult_only_edge(image_layer, edge_layer, num_classes, filter_mult):
    num_filters = num_classes * filter_mult
    
    img_down = utils.convolution_block(image_layer, num_filters=num_filters, kernel_size=5, dilation_rate=1, strides=2,
                                       separable=True, name='pyramid_downsampling')
    img_down = tf.keras.layers.Concatenate(axis=-1)([edge_layer, img_down])
    
    pyramid_1 = utils.convolution_block(img_down, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                        RELU=True, separable=True, name="pyramid_1_1")
    pyramid_1 = utils.convolution_block(pyramid_1, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                        RELU=True, name="pyramid_1_2")
    pyramid_2 = utils.convolution_block(img_down, num_filters=num_filters, kernel_size=5, dilation_rate=1, BN=True,
                                        RELU=True, name="pyramid_2_1")
    pyramid_3 = utils.convolution_block(img_down, num_filters=num_filters, kernel_size=7, dilation_rate=1, BN=True,
                                        RELU=True, name="pyramid_3_1")
    
    pyramid_4 = tf.keras.layers.AveragePooling2D(pool_size=(14, 14), strides=7,
                                                 padding='SAME')(img_down)
    pyramid_4 = utils.convolution_block(pyramid_4, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                        RELU=True, name="pyramid_avg")

    pyramid_4 = tf.image.resize(pyramid_4, size=(edge_layer.shape[1], edge_layer.shape[2]))

    pyramid = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid = utils.convolution_block(pyramid, num_filters=num_filters, kernel_size=3, name="pyramid_out")
    
    out = tf.keras.layers.Concatenate(axis=-1)([edge_layer, pyramid])
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, name="pyramid_module_out")
    
    return out
