import tensorflow as tf
from models.network_elements import utils


def viot_fusion_module(dec_edge, side_1, side_2, num_classes, num_filters_per_class, output_name="out_edge"):
    num_filters = num_filters_per_class * num_classes
    fusion_1 = tf.keras.layers.Concatenate()([dec_edge, side_1, side_2])
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=1, num_filters=4 * num_filters)
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=3, num_filters=4 * num_filters, separable=True)
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=1, num_filters=num_filters, BN=False)
    # fusion_1 = utils.convolution_block(fusion_1, kernel_size=3, num_filters=num_filters, separable=True)
    output = utils.convolution_block(fusion_1, kernel_size=3, BN=False, RELU=False, num_filters=1)
    # adaptive_weight = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(fusion_1)
    # adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, num_filters=4, BN=False)
    # adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, num_filters=4, BN=False, RELU=False)
    # adaptive_weight = tf.keras.layers.Activation(activation="hard_sigmoid")(adaptive_weight)
    # fusion_1 = fusion_1 * adaptive_weight * dec_avg
    
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(output)


def viot_fusion_module_prior(pyramid_module_input, side_feature_input, num_classes, num_filters_per_class, output_shape,
                             output_name="out_edge"):
    num_filters = num_filters_per_class * num_classes
    fusion_module_1 = tf.keras.layers.Concatenate()([pyramid_module_input, side_feature_input])
    fusion_module_2 = utils.convolution_block(fusion_module_1, kernel_size=1, num_filters=2 * num_filters, BN=False,
                                              RELU=False)
    fusion_module_3 = utils.convolution_block(fusion_module_2, kernel_size=1, num_filters=2 * num_filters)
    fusion_module_4 = utils.convolution_block(fusion_module_3, kernel_size=1, num_filters=num_filters, separable=True)
    fusion_module_5 = tf.image.resize(fusion_module_4, output_shape, method="bilinear")
    fusion_module_6 = utils.convolution_block(fusion_module_5, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_7 = utils.convolution_block(fusion_module_6, BN=False, RELU=False, num_filters=num_classes,
                                              kernel_size=3)
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(fusion_module_7)


def viot_fusion_module_prior_segmentation(pyramid_module_input, side_feature_input, num_classes, num_filters_per_class,
                                          output_shape,
                                          output_name="out_edge"):
    num_filters = num_filters_per_class * num_classes
    fusion_module_1 = tf.keras.layers.Concatenate()([pyramid_module_input, side_feature_input])
    fusion_module_2 = utils.convolution_block(fusion_module_1, kernel_size=1, num_filters=2 * num_filters, BN=False,
                                              RELU=False)
    fusion_module_3 = utils.convolution_block(fusion_module_2, kernel_size=3, num_filters=2 * num_filters)
    fusion_module_4 = utils.convolution_block(fusion_module_3, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_5 = tf.image.resize(fusion_module_4, output_shape, method="bilinear")
    fusion_module_6 = utils.convolution_block(fusion_module_5, kernel_size=3, num_filters=num_filters, separable=True)
    fusion_module_7 = tf.keras.layers.Conv2D(kernel_size=3, filters=num_classes, name=output_name, padding="SAME")(fusion_module_6)
    return fusion_module_7


# Elements from Paper
def lite_edge_output(decoder_output, sides, num_classes, output_name="out_edge", output_shape=(320, 180)):
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        decoder_per_class = decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)]
        
        decoder = tf.keras.layers.Concatenate()(sides + [decoder_per_class])
        decoder = utils.convolution_block(decoder, kernel_size=3, num_filters=decoder.shape[-1])
        decoder = utils.convolution_block(decoder, kernel_size=1, num_filters=1, RELU=False, BN=False)
        
        out.append(decoder)
    
    if num_classes == 1:
        output = out[0]
    else:
        output = tf.keras.layers.Concatenate()(out)
    output = tf.image.resize(output, output_shape)
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(output)


def FENet(decoder_output, sides, num_classes, output_shape=(320, 160)):
    decoder = tf.image.resize(decoder_output, output_shape)
    
    dec = tf.keras.layers.Concatenate()([decoder] * 4)
    
    concat = tf.keras.layers.Concatenate()(sides * num_classes + [decoder])
    
    adaptive_weight = utils.convolution_block(concat, kernel_size=1)
    adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, RELU=False)
    
    output = dec * adaptive_weight
    
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2], 4, num_classes])
    
    output = tf.reduce_sum(output, axis=3)
    
    return tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(output)
