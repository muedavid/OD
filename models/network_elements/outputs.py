import tensorflow as tf
from models.network_elements import utils


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
    
    adaptive_weight = utils.convolution_block(decoder, num_filters=num_classes * 4, kernel_size=1)
    adaptive_weight = utils.convolution_block(adaptive_weight, num_filters=num_classes * 4, kernel_size=1)
    adaptive_weight = utils.convolution_block(adaptive_weight, num_filters=num_classes * 4, kernel_size=1, RELU=False)
    
    fusion_decoder = utils.convolution_block(decoder, num_filters=num_classes)
    fusion = tf.keras.layers.Concatenate()([fusion_decoder] + sides * num_classes)
    
    output = fusion * adaptive_weight
    
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2], 4, num_classes])
    
    output = tf.reduce_sum(output, axis=3)
    
    return tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(output)
