import tensorflow as tf
from models.network_elements import utils


def viot_fusion_module(pyramid_module_input, side_feature_input, num_classes, num_filters_per_class, output_shape,
                       output_name="out_edge"):
    
    num_filters = num_filters_per_class * num_classes
    
    pyramid_1 = tf.image.resize(pyramid_module_input, (side_feature_input.shape[1], side_feature_input.shape[2]), method="bilinear")
    pyramid_2 = utils.convolution_block(pyramid_1, kernel_size=3, num_filters=2 * num_filters, separable=True)
    
    fusion_module_1 = tf.keras.layers.Concatenate()([pyramid_2, side_feature_input])
    fusion_module_2 = utils.convolution_block(fusion_module_1, kernel_size=1, num_filters=4 * num_filters)
    fusion_module_3 = utils.convolution_block(fusion_module_2, kernel_size=1, num_filters=4 * num_filters)
    fusion_module_4 = utils.convolution_block(fusion_module_3, kernel_size=1, num_filters=2 * num_filters)
    fusion_module_5 = tf.image.resize(fusion_module_4, output_shape, method="bilinear")
    fusion_module_6 = utils.convolution_block(fusion_module_5, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_7 = utils.convolution_block(fusion_module_6, BN=False, RELU=False, num_filters=num_classes,
                                              kernel_size=3)
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(fusion_module_7)


def viot_fusion_module_prior_segmentation(pyramid_module_input, side_feature_input, num_classes, num_filters_per_class,
                                          output_shape,
                                          output_name="out_segmentation"):
    num_filters = num_filters_per_class * num_classes

    pyramid_1 = tf.image.resize(pyramid_module_input, (side_feature_input.shape[1], side_feature_input.shape[2]),
                                method="bilinear")
    pyramid_2 = utils.convolution_block(pyramid_1, kernel_size=3, num_filters=num_filters, separable=True)
    
    fusion_module_1 = tf.keras.layers.Concatenate()([pyramid_2, side_feature_input])
    fusion_module_2 = utils.convolution_block(fusion_module_1, kernel_size=1, num_filters=2 * num_filters, BN=False,
                                              RELU=False)
    fusion_module_3 = utils.convolution_block(fusion_module_2, kernel_size=3, num_filters=2 * num_filters)
    fusion_module_4 = utils.convolution_block(fusion_module_3, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_5 = tf.image.resize(fusion_module_4, output_shape, method="bilinear")
    fusion_module_6 = utils.convolution_block(fusion_module_5, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_7 = tf.keras.layers.Conv2D(kernel_size=3, filters=num_classes, name=output_name, padding="SAME")(
        fusion_module_6)
    return fusion_module_7

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


def lite_edge_decoder(daspp_output, side_output):
    decoder_1 = utils.convolution_block(daspp_output, kernel_size=1, num_filters=96)

    decoder_2 = tf.image.resize(decoder_1, (side_output.shape[1], side_output.shape[2]))
    
    decoder_3 = tf.keras.layers.Concatenate()([decoder_2, side_output])
    decoder_4 = utils.convolution_block(decoder_3, kernel_size=3, separable=True, num_filters=96)
    decoder_5 = utils.convolution_block(decoder_4, kernel_size=3, separable=True, num_filters=96)
    decoder_6 = utils.convolution_block(decoder_5, kernel_size=3, separable=True, num_filters=1)
    
    return decoder_6
