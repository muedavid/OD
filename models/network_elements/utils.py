import tensorflow as tf


def convolution_block(block_input, name, num_filters=1, kernel_size=3, dilation_rate=1, strides=1, padding="same",
                      use_bias=True, separable=False, depthwise=False, depth_multiplier=1, BN=True, RELU=True):
    if separable:
        x = tf.keras.layers.SeparableConv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, strides=strides, use_bias=use_bias,
                                            name=name + '_separable_conv')(block_input)
    elif depthwise:
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, strides=strides, use_bias=use_bias,
                                            depth_multiplier=depth_multiplier,
                                            name=name + 'depthwise_conv')(block_input)
    
    else:
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                   strides=strides, use_bias=use_bias, name=name + '_conv')(block_input)
    if BN:
        x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    if RELU:
        x = tf.keras.layers.ReLU(name=name + '_relu')(x)
    return x


def shared_concatenation_and_classification(decoder_output, side, num_classes, num_filters, name):
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        decoder_per_class = decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)]
        
        decoder_mult = convolution_block(decoder_per_class, kernel_size=1, name="output_decoder_mult_{}".format(i), num_filters=2,
                                         RELU=False)
        decoder_mult = tf.keras.layers.Activation(activation="sigmoid")(decoder_mult)
        side_mult = convolution_block(side, name="output_side_mult_{}".format(i), num_filters=2)
        out = []
        for j in range(2):
            out.append(decoder_mult[:, :, :, j:j + 1] * side_mult)
        mult = tf.keras.layers.Concatenate(axis=-1)(out)
        decoder_concat = convolution_block(decoder_per_class, kernel_size=1, name="output_decoder_concat_{}".format(i), num_filters=2)
        side_concat = convolution_block(side, name="output_side_concat_{}".format(i), num_filters=2)
        
        concatenated_layers_1 = tf.keras.layers.Concatenate(axis=-1)([mult, decoder_concat, side_concat])
        
        x = convolution_block(concatenated_layers_1, num_filters=num_filters, kernel_size=1,
                              name='out_concat_1_{}'.format(i))
        x = convolution_block(x, num_filters=num_filters, kernel_size=1,
                              name='out_concat_2_{}'.format(i))
        convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
        if num_classes == 1:
            return tf.keras.layers.Activation(activation='sigmoid', name=name)(convolved_layers)
        else:
            out.append(convolved_layers)
    
    output = tf.keras.layers.Concatenate(axis=-1, name='output_concatenation')(out)
    return tf.keras.layers.Activation(activation='sigmoid', name=name)(output)


def shared_concatenation_and_classification_old(decoder_output, side_outputs, num_classes, num_filters, name):
    sides = tf.keras.layers.Concatenate(axis=-1)(side_outputs)
    sides = convolution_block(sides, num_filters=3, kernel_size=1, name="sides_concatenation")
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        concatenated_layers_1 = tf.keras.layers.Concatenate(axis=-1)(
            [decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)], sides])
        x = convolution_block(concatenated_layers_1, num_filters=num_filters, kernel_size=1,
                              name='out_concat_1_{}'.format(i))
        x = convolution_block(x, num_filters=idx_mult, kernel_size=3, separable=True,
                              name='out_concat_2_{}'.format(i))
        
        if num_classes == 1:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1, name=name)(x)
            return convolved_layers
        else:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
            out.append(convolved_layers)
    
    output = tf.keras.layers.Concatenate(axis=-1, name='output_concatenation')(out)
    return tf.keras.layers.Activation(activation='sigmoid', name=name)(output)
