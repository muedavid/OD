import tensorflow as tf
from models.network_elements import utils


def pyramid_module(pyramid_input, num_filters=12):
    x = utils.convolution_block(pyramid_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                seperable=True, name="pyramid_input")
    
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
    daspp_input = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                          seperable=True, name="daspp_input")
    
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
                                    seperable=True, BN=True, RELU=True, name="daspp_3_dilated")
    out_3 = utils.convolution_block(out_3, num_filters=num_filters, kernel_size=3, dilation_rate=1, seperable=True,
                                    BN=True, RELU=True, name="daspp_3_conv")
    out_6 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=6,
                                    seperable=True, BN=True, RELU=True, name="daspp_6_dilated")
    out_6 = utils.convolution_block(out_6, num_filters=num_filters, kernel_size=3, dilation_rate=1, seperable=True,
                                    BN=True, RELU=True, name="daspp_6_conv")
    out_9 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=9,
                                    seperable=True, BN=True, RELU=True, name="daspp_9_dilated")
    out_9 = utils.convolution_block(out_9, num_filters=num_filters, kernel_size=3, dilation_rate=1, seperable=True,
                                    BN=True, RELU=True, name="daspp_9_conv")
    out_18 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=18, seperable=True, BN=True, RELU=True,
                                     name="daspp_18_dilated")
    out_18 = utils.convolution_block(out_18, num_filters=num_filters, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True,
                                     name="daspp_18_conv")
    
    out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(daspp_input)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="daspp_avg")
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_3, out_6, out_9, out_18, daspp_input])
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, name="daspp_out")
    
    return x


def concatenate_edge_and_image(image_layer, edge_layer, num_classes, filter_mult):
    num_filters = num_classes * filter_mult
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=num_filters, kernel_size=3, seperable=True,
                                                 name='image_preprocessing_before_concatenation')
    edge_layer_concat = utils.convolution_block(edge_layer, num_filters=num_filters, kernel_size=3, RELU=True, BN=True, name='edge_preprocessing_before_concatenation')
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer_concat])
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, name="image_edge_concatenated_1")
    image_edge_concat = utils.convolution_block(x, num_filters=num_filters, kernel_size=3,
                                                name="image_edge_concatenated_2")
    
    pyramid = utils.convolution_block(image_edge_concat, name="pyramid_down", strides=2, kernel_size=3,
                                      num_filters=num_filters, seperable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_1", kernel_size=1, num_filters=num_filters)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_2", kernel_size=3, num_filters=num_filters,
                                        seperable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_3", kernel_size=5, num_filters=num_filters,
                                        seperable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_out", num_filters=num_filters, kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]))
    out = tf.keras.layers.Concatenate(axis=-1)([pyramid_out, image_edge_concat])
    out = utils.convolution_block(out, kernel_size=1, num_filters=num_filters, name="out_pyramid_module")
    
    return out
