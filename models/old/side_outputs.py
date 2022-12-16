import tensorflow as tf
from models.network_elements import utils


def side_feature_edge_prior(x, output_dims, num_filters, method="bilinear", name=None):
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_1_conv3x3")
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_2_conv3x3")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_3_conv1x1")
    x = tf.image.resize(x, (int(output_dims[0] / 2), int(output_dims[1] / 2)), method=method)
    
    x = utils.convolution_block(x, num_filters=1, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                separable=True, name=name + "_4_conv1x1")
    
    if x.shape[1] != output_dims[0]:
        x = tf.image.resize(x, (output_dims[0], output_dims[1]), method=method)
    return x