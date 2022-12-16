import tensorflow as tf
from models.network_elements import utils


def viot_fusion_module(decoder_output, side, num_classes, num_filters_per_class, output_name="out_edge"):
    # Note: Avoided indexing (memory consumption, avoided BN as values should start being meaningful as what they really are)
    num_filters = num_filters_per_class * num_classes
    side_single = utils.convolution_block(side, num_filters=1, kernel_size=3, RELU=False)
    # TODO(Try only multiplication, only adding layers, ...)
    fusion_1 = tf.keras.layers.Concatenate()([side, side_single*decoder_output])
    fusion_2 = utils.mobile_net_v2_inverted_residual(fusion_1, depth_multiplier=6)
    fusion_3 = utils.convolution_block(fusion_2, kernel_size=3, num_filters=num_filters, RELU=False)
    output = utils.convolution_block(fusion_3, BN=False, RELU=False, num_filters=1, kernel_size=1)
    return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(output)


def shared_concatenation_and_classification(decoder_output, side, num_classes, num_filters_per_class, output_name):
    num_filters = num_classes * num_filters_per_class
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        decoder_per_class = decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)]
        
        decoder_mult = utils.convolution_block(decoder_per_class, kernel_size=1,
                                               name="output_decoder_mult_{}".format(i),
                                               num_filters=2,
                                               RELU=False)
        decoder_mult = tf.keras.layers.Activation(activation="sigmoid")(decoder_mult)
        side_mult = utils.convolution_block(side, name="output_side_mult_{}".format(i), num_filters=2)
        out = []
        for j in range(2):
            out.append(decoder_mult[:, :, :, j:j + 1] * side_mult)
        mult = tf.keras.layers.Concatenate(axis=-1)(out)
        decoder_concat = utils.convolution_block(decoder_per_class, kernel_size=1,
                                                 name="output_decoder_concat_{}".format(i),
                                                 num_filters=2)
        side_concat = utils.convolution_block(side, name="output_side_concat_{}".format(i), num_filters=2)
        
        concatenated_layers_1 = tf.keras.layers.Concatenate(axis=-1)([mult, decoder_concat, side_concat])
        
        x = utils.convolution_block(concatenated_layers_1, num_filters=num_filters, kernel_size=1,
                                    name='out_concat_1_{}'.format(i))
        x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1,
                                    name='out_concat_2_{}'.format(i))
        convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
        if num_classes == 1:
            return tf.keras.layers.Activation(activation='sigmoid', name=output_name)(convolved_layers)
        else:
            out.append(convolved_layers)
    
    output = tf.keras.layers.Concatenate(axis=-1, name='output_concatenation')(out)
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


