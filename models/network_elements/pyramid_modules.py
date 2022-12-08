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


def concatenate_edge_and_image_mult(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = image_layer.shape[1] / edge_layer.shape[1]
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=down_sampling + 1,
                                              strides=down_sampling,
                                              separable=True, name="image_down_sampling")
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True, name='pyramid_module_image_1_1')
    image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True,
                                                 name='pyramid_module_image_1_2')
    edge_layer_concat = utils.convolution_block(edge_layer, num_filters=small_filter_num, kernel_size=3, RELU=True,
                                                BN=True,
                                                name='pyramid_module_edge_1')
    
    image = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_1")
    image = utils.convolution_block(image, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_2")
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer_concat])
    image_edge_concat = utils.convolution_block(x, num_filters=small_filter_num, kernel_size=1,
                                                name="pyramid_module_pyramid_concat")
    
    pyramid = utils.convolution_block(image_edge_concat, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
                                      num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=4)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=3,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=2,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=1,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    
    out = []
    for i in range(small_filter_num):
        mult = tf.slice(pyramid_out, begin=[0, 0, 0, i], size=[-1, -1, -1, 1],
                        name='pyramid_module_pyramid_out_{}'.format(i))
        out.append(tf.multiply(mult, image, name='pyramid_module_multiplication_{}'.format(i)))
    
    after_mult = tf.keras.layers.Concatenate(axis=-1)(out)
    after_mult = utils.convolution_block(after_mult, num_filters=3 * small_filter_num, BN=True, RELU=True,
                                         kernel_size=1,
                                         name="pyramid_module_after_mult_1")
    after_mult = utils.convolution_block(after_mult, num_filters=small_filter_num, BN=True, RELU=True, kernel_size=3,
                                         separable=True, name="pyramid_module_after_mult_2")
    
    image = utils.convolution_block(image, kernel_size=3, num_filters=2,
                                    name="pyramid_module_image_2_3")
    out = tf.keras.layers.Concatenate(axis=-1)([after_mult, image])
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_2")
    
    return out, pyramid_out, edge_layer_concat, image_layer_concat, after_mult


def concatenate_edge_and_image(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = image_layer.shape[1] / edge_layer.shape[1]
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
    image_edge_concat = utils.convolution_block(x, num_filters=small_filter_num, kernel_size=1,
                                                name="pyramid_module_pyramid_concat")
    
    pyramid = utils.convolution_block(image_edge_concat, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
                                      num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=5)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=4,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=3,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=3,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    
    out = tf.keras.layers.Concatenate(axis=-1)([pyramid_out, edge, image])
    out = utils.convolution_block(out, kernel_size=1, num_filters=2 * small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=3, num_filters=2 * small_filter_num, name="pyramid_module_out_2")
    out = utils.convolution_block(out, kernel_size=3, separable=True, num_filters=small_filter_num,
                                  name="pyramid_module_out_3")
    
    return out, pyramid_out, edge_layer_concat, image_layer_concat
