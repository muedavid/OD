import tensorflow as tf


def convolution_block(block_input, name=None, num_filters=1, kernel_size=3, dilation_rate=1, strides=1, padding="same",
                      use_bias=True, separable=False, depthwise=False, depth_multiplier=1, BN=True, RELU=True):
    if separable and depthwise:
        raise ValueError("only one of the following arguments: separable or depthwise can be True")
    activation = "relu" if RELU else None
    if separable:
        layer_name = None if name is None else name + '_separable_conv'
        x = tf.keras.layers.SeparableConv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, strides=strides, use_bias=use_bias,
                                            name=layer_name, activation=activation)(block_input)
    elif depthwise:
        layer_name = None if name is None else name + '_depthwise_conv'
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, dilation_rate=dilation_rate,
                                            padding=padding, strides=strides, use_bias=use_bias,
                                            depth_multiplier=depth_multiplier,
                                            name=layer_name, activation=activation)(block_input)
    
    else:
        layer_name = None if name is None else name + '_conv'
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                   strides=strides, use_bias=use_bias,
                                   name=layer_name, activation=activation)(block_input)
    if BN:
        layer_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(name=layer_name)(x)
    # if RELU:
    #     layer_name = None if name is None else name + '_relu'
    #     x = tf.keras.layers.ReLU(name=layer_name)(x)
    return x

def mobile_net_v2_inverted_residual(input_layer, depth_multiplier, output_depht_multiplier=1, strides=1):
    expand = convolution_block(input_layer, num_filters=input_layer.shape[-1] * depth_multiplier, kernel_size=1)
    depth = convolution_block(expand, depthwise=True, kernel_size=3, strides=strides)
    output = convolution_block(depth, num_filters=input_layer.shape[-1] * output_depht_multiplier, RELU=False,
                               kernel_size=1)
    return output


def time_testing_concat(img_input):
    splits = 5
    split_size = 2
    conv = convolution_block(img_input, num_filters=splits * split_size, kernel_size=3)
    
    for j in range(20):
        out = []
        for i in range(splits):
            out.append(convolution_block(conv, num_filters=split_size, kernel_size=1))
        out = tf.keras.layers.Concatenate()(out)
        
        conv = convolution_block(out, num_filters=1, kernel_size=1, RELU=False)
    
    out = tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(conv)
    
    return out


def time_testing(img_input):
    splits = 5
    split_size = 2
    conv = convolution_block(img_input, num_filters=splits * split_size, kernel_size=3)
    
    for j in range(20):
        out = convolution_block(conv, num_filters=split_size * splits, kernel_size=1)
        
        conv = convolution_block(out, num_filters=1, kernel_size=1, RELU=False)
    
    out = tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(conv)
    
    return out


def time_testing_add(img_input):
    splits = 5
    split_size = 2
    conv = convolution_block(img_input, num_filters=splits * split_size, kernel_size=3)
    
    for j in range(20):
        out = []
        for i in range(splits):
            out.append(convolution_block(conv, num_filters=split_size, kernel_size=1))
        out = tf.keras.layers.Add()(out)
        
        conv = convolution_block(out, num_filters=1, kernel_size=1, RELU=False)
    
    out = tf.keras.layers.Activation(activation="sigmoid", name="out_edge")(conv)
    
    return out


