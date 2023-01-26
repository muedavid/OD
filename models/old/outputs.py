import tensorflow as tf
from models.network_elements import utils


def shared_concatenation_and_classification_old(decoder_output, side_outputs, num_classes, num_filters_per_class,
                                                name):
    num_filters = num_filters_per_class * num_classes
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