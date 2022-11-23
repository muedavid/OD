import tensorflow as tf


def convolution_block(block_input, num_filters=24, kernel_size=3, dilation_rate=1, strides=1, padding="same",
                      use_bias=True, seperable=False, BN=True, RELU=True, name='conv_block'):
    if seperable:
        x = tf.keras.layers.SeparableConv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, strides=strides, use_bias=use_bias,
                                            name=name + '_separable_conv')(block_input)
    else:
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                   strides=strides, use_bias=use_bias, name=name + '_conv')(block_input)
    if BN:
        x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)
    if RELU:
        x = tf.keras.layers.ReLU(name=name + '_relu')(x)
    return x


def shared_concatenation_and_classification(decoder_output, side_outputs, num_classes, num_filters, name):
    out = []
    for i in range(num_classes):
        idx_mult = int(decoder_output.shape[-1] / num_classes)
        shared_concat = [decoder_output[:, :, :, idx_mult * i:idx_mult * (i + 1)]]
        
        for j in range(len(side_outputs)):
            shared_concat.append(side_outputs[j])
        
        concatenated_layers = tf.keras.layers.Concatenate(axis=-1)(shared_concat)
        x = convolution_block(concatenated_layers, num_filters=num_filters, kernel_size=3, name='out_{}'.format(i))
        
        if num_classes == 1:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1, name=name)(x)
            return convolved_layers
        else:
            convolved_layers = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(x)
            out.append(convolved_layers)
    return tf.keras.layers.Concatenate(axis=-1, name=name)(out)
