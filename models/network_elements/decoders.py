import tensorflow as tf
from models.network_elements import utils


def sged_decoder(decoder_input, side, output_dims, num_classes, num_filters_per_class):
    num_filters = num_filters_per_class * num_classes
    x = tf.image.resize(decoder_input, (side.shape[1], side.shape[2]))
    
    side = utils.convolution_block(side, num_filters=int(num_filters / 4), kernel_size=1, dilation_rate=1,
                                   separable=False, BN=True, RELU=True, name="decoder_1")
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, side])
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True, BN=True,
                                RELU=True, name="decoder_2")
    
    x = tf.image.resize(x, size=(output_dims[0], output_dims[1]))
    
    x_1 = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True, BN=True,
                                  RELU=True, name="decoder_3")
    
    return x_1


def decoder_small(decoder_input, output_dims, num_classes, num_filters_per_class):
    num_filters = num_filters_per_class*num_classes
    x = tf.image.resize(decoder_input, size=(output_dims[0], output_dims[1]), method="bilinear")
    x_1 = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, separable=True, BN=True,
                                  RELU=True, name="decoder_1")
    return x_1


def lite_edge_decoder(daspp_input, side_input):
    name = "LiteEdge_Decoder_"
    daspp = utils.convolution_block(daspp_input, kernel_size=1, name=name+"1", num_filters=20)
    
    daspp = tf.image.resize(daspp, (side_input.shape[1], side_input.shape[2]))
    
    decoder = tf.keras.layers.Concatenate()([daspp, side_input])
    decoder = utils.convolution_block(decoder, kernel_size=3, separable=True, name=name+"2", num_filters=20)
    decoder = utils.convolution_block(decoder, kernel_size=3, separable=True, name=name + "2", num_filters=20)
    decoder = utils.convolution_block(decoder, kernel_size=3, separable=True, name=name + "2", num_filters=1)
    
    return decoder
