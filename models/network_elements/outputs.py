import tensorflow as tf
from models.network_elements import utils


def viot_fusion_module(dec_edge, side_1, side_2, num_classes, num_filters_per_class, output_name="out_edge"):
    num_filters = num_filters_per_class*num_classes
    fusion_1 = tf.keras.layers.Concatenate()([dec_edge, side_1, side_2])
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=1, num_filters=4*num_filters)
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=3, num_filters=4*num_filters, separable=True)
    fusion_1 = utils.convolution_block(fusion_1, kernel_size=1, num_filters=num_filters, BN=False)
    # fusion_1 = utils.convolution_block(fusion_1, kernel_size=3, num_filters=num_filters, separable=True)
    output = utils.convolution_block(fusion_1, kernel_size=3, BN=False, RELU=False, num_filters=1)
    # adaptive_weight = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(fusion_1)
    # adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, num_filters=4, BN=False)
    # adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, num_filters=4, BN=False, RELU=False)
    # adaptive_weight = tf.keras.layers.Activation(activation="hard_sigmoid")(adaptive_weight)
    # fusion_1 = fusion_1 * adaptive_weight * dec_avg

    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(output)


def viot_fusion_module_prior (dec_1, side_1, num_classes, num_filters_per_class, output_name="out_edge"):
    # Note: Avoided indexing (memory consumption, avoided BN as values should start being meaningful as what they really are)
    num_filters = num_filters_per_class * num_classes
    side = tf.keras.layers.Concatenate()([side_1, dec_1])
    side = utils.convolution_block(side, kernel_size=1, num_filters=2*num_filters)
    side = utils.convolution_block(side, kernel_size=3, num_filters=2*num_filters, separable=True, BN=False)
    side = utils.convolution_block(side, kernel_size=3, num_filters=num_filters, separable=True, BN=False)
    output = utils.convolution_block(side, BN=False, RELU=False, num_filters=1, kernel_size=1)
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(output)


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




def shared_concatenation_and_classification_old(decoder_output, side_outputs, num_classes, num_filters_per_class,
                                                name):
    num_filters = num_filters_per_class*num_classes
    sides = tf.keras.layers.Concatenate(axis=-1)(side_outputs)
    sides = utils.convolution_block(sides, num_filters=3, kernel_size=1, name="sides_concatenation")
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        concatenated_layers_1 = tf.keras.layers.Concatenate(axis=-1)(
            [decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)], sides])
        x = utils.convolution_block(concatenated_layers_1, num_filters=num_filters, kernel_size=1,
                                    name='out_concat_1_{}'.format(i))
        x = utils.convolution_block(x, num_filters=idx_mult, kernel_size=3, separable=True,
                                    name='out_concat_2_{}'.format(i))
        
        if num_classes == 1:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1, name=name)(x)
            return convolved_layers
        else:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
            out.append(convolved_layers)
    
    output = tf.keras.layers.Concatenate(axis=-1, name='output_concatenation')(out)
    return tf.keras.layers.Activation(activation='sigmoid', name=name)(output)


def FENet(decoder_output, sides, num_classes, output_shape=(320, 160)):
    decoder = tf.image.resize(decoder_output, output_shape)
    
    dec = tf.keras.layers.Concatenate()([decoder]*4)
    
    concat = tf.keras.layers.Concatenate()(sides*num_classes+[decoder])
    
    adaptive_weight = utils.convolution_block(concat, kernel_size=1)
    adaptive_weight = utils.convolution_block(adaptive_weight, kernel_size=1, RELU=False)
    
    output = dec*adaptive_weight
    
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2], 4, num_classes])
    
    output = tf.reduce_sum(output, axis=3)
    
    return tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(output)
    
    
    
    