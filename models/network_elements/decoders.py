import tensorflow as tf
from models.network_elements import utils


def sged_decoder(decoder_input, side, output_dims, num_filters=15):
    x = tf.image.resize(decoder_input, (side.shape[1], side.shape[2]))
    
    side = utils.convolution_block(side, num_filters=int(num_filters / 4), kernel_size=1, dilation_rate=1,
                                   separable=False, BN=True, RELU=True, name="decoder_1")
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, side])
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True, BN=True,
                                RELU=True, name="decoder_2")
    
    x = tf.image.resize(x, size=(output_dims[0], output_dims[1]))
    
    x_1 = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True, BN=True,
                                  RELU=True, name="decoder_3")
    
    x_2 = utils.convolution_block(x_1, num_filters=num_filters, kernel_size=1, dilation_rate=1, separable=True,
                                  BN=True, RELU=True, name="decoder_4")
    
    return x_2


def sged_side_feature(x, output_dims, num_filters, method="bilinear", name=None):
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_1_conv3x3")
    x = utils.convolution_block(x, num_filters=int(num_filters/2), kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_2_conv3x3")
    x = utils.convolution_block(x, num_filters=3, kernel_size=1, dilation_rate=1, BN=False, RELU=False,
                                separable=True, name=name + "_conv1x1")
    
    if x.shape[1] != output_dims[0]:
        x = tf.image.resize(x, (output_dims[0], output_dims[1]), method=method)
    return x


def side_feature_edge_prior(x, output_dims, num_filters, method="bilinear", name=None):
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_1_conv3x3")
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_2_conv3x3")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_3_conv1x1")
    x = tf.image.resize(x, (int(output_dims[0]/2), int(output_dims[1]/2)), method=method)
    
    x = utils.convolution_block(x, num_filters=3, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_4_conv1x1")
    
    if x.shape[1] != output_dims[0]:
        x = tf.image.resize(x, (output_dims[0], output_dims[1]), method=method)
    return x


def casenet_side_feature(x, channels, kernel_size_transpose, stride_transpose, output_padding=None, padding='same',
                         name=None):
    x = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=(1, 1), padding='same')(x)
    return tf.keras.layers.Conv2DTranspose(channels, kernel_size=kernel_size_transpose,
                                           strides=(stride_transpose, stride_transpose), padding=padding,
                                           output_padding=output_padding, use_bias=False, name=name)(x)
