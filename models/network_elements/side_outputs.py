import tensorflow as tf
from models.network_elements import utils


def viot_side_feature(x1, output_dims, num_classes, num_filters_per_class, method="bilinear"):
    num_filters = num_filters_per_class * num_classes
    
    x1 = utils.convolution_block(x1, kernel_size=3, separable=True, num_filters=2 * num_filters)
    x1 = utils.convolution_block(x1, kernel_size=3, separable=True, num_filters=2 * num_filters)
    x1 = utils.convolution_block(x1, kernel_size=1, separable=True, num_filters=num_filters)
    if x1.shape[1] != output_dims[0]:
        x1 = tf.image.resize(x1, output_dims)
    return x1


def viot_side_feature_prior(x1, output_dims, num_classes, num_filters_per_class, method="bilinear"):
    num_filters = num_filters_per_class * num_classes
    x1 = utils.convolution_block(x1, kernel_size=3, num_filters=num_filters, separable=True)
    
    if x1.shape[1] != output_dims[0]:
        x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="SAME", kernel_initializer=tf.keras.initializers.Constant(0.5), use_bias=False)(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.image.resize(x1, output_dims)
        
    return x1


def viot_side_feature_prior_segmentation(x1, x2, output_dims_edge, output_dims_segmentation, num_classes,
                                         num_filters_per_class, method="bilinear"):
    num_filters = num_filters_per_class * num_classes
    segmentation = utils.convolution_block(x1, kernel_size=3, num_filters=10, separable=True)
    segmentation = utils.mobile_net_v2_inverted_residual(segmentation, depth_multiplier=4)
    edge = utils.convolution_block(x2, kernel_size=3, num_filters=6, separable=True)
    edge = utils.convolution_block(edge, kernel_size=1, num_filters=6)
    
    if edge.shape[1] != output_dims_edge[0]:
        edge = tf.image.resize(edge, output_dims_edge)
        edge = utils.convolution_block(edge, kernel_size=3, separable=True, num_filters=6, BN=False)
    if segmentation.shape[1] != output_dims_segmentation[0]:
        segmentation = tf.image.resize(segmentation, output_dims_segmentation)
        segmentation = utils.convolution_block(segmentation, kernel_size=3, separable=True, num_filters=6)
    return segmentation, edge


def side_feature(x, output_dims, num_classes, num_filters_per_class, method="bilinear", name=None):
    num_filters = num_filters_per_class * num_classes
    sides = []
    for layer in x:
        if layer.shape[1] != output_dims[0]:
            layer = tf.image.resize(layer, (output_dims[0], output_dims[1]), method=method)
        sides.append(layer)
    side_feature_1 = tf.keras.layers.Concatenate(axis=-1)(sides)
    
    side_feature_2 = utils.convolution_block(side_feature_1, num_filters=2 * num_filters, kernel_size=1,
                                             dilation_rate=1, BN=True, RELU=True,
                                             name=name + "_2_conv3x3")
    side_feature_3 = utils.convolution_block(side_feature_2, num_filters=int(num_filters / 2), kernel_size=3,
                                             BN=True, RELU=True, separable=True, name=name + "_3_conv3x3")
    side_feature_4 = utils.convolution_block(side_feature_3, num_filters=int(num_filters / 2), kernel_size=5,
                                             dilation_rate=1, BN=True, RELU=True, separable=True,
                                             name=name + "_4_conv3x3")
    side_feature_5 = utils.convolution_block(side_feature_2, num_filters=int(num_filters / 2), kernel_size=1,
                                             dilation_rate=1, BN=True, RELU=True,
                                             separable=True, name=name + "_5_conv3x3")
    side_feature_6 = tf.keras.layers.Concatenate(axis=-1)([side_feature_4, side_feature_5])
    side_feature_7 = utils.convolution_block(side_feature_6, num_filters=int(num_filters / 2), kernel_size=1,
                                             dilation_rate=1, BN=True, RELU=True,
                                             name=name + "_7_conv1x1")
    return side_feature_7


def casenet_side_feature(x, channels, kernel_size_transpose, stride_transpose, output_padding=None, padding='same',
                         name=None):
    x = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=(1, 1), padding='same')(x)
    return tf.keras.layers.Conv2DTranspose(channels, kernel_size=kernel_size_transpose,
                                           strides=(stride_transpose, stride_transpose), padding=padding,
                                           output_padding=output_padding, use_bias=False, name=name)(x)


def lite_edge_side_feature_extraction(side, output_shape):
    side = utils.convolution_block(side, num_filters=20, separable=True)
    side = utils.convolution_block(side, num_filters=1, kernel_size=1)
    side_output = tf.image.resize(side, output_shape)
    
    return side_output


def FENet_side_feature_extraction(side, output_shape):
    side = utils.convolution_block(side, num_filters=1, separable=True)
    side_output = tf.image.resize(side, output_shape)
    
    return side_output
