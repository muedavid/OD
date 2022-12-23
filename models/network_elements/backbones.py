import traitlets.config
from tensorflow import keras
import tensorflow as tf
from numpy import array, log2
from models.network_elements import utils


def get_backbone(name="MobileNetV2", weights="imagenet", input_shape=(640, 360, 3), alpha=1,
                 output_layer=None, trainable_idx=None, input_name='in_img'):
    if output_layer is None:
        output_layer = [0, 1, 2]
    include_top = False
    if name == 'RESNet101':
        base_model = keras.applications.resnet.ResNet101(include_top=include_top, weights=weights,
                                                         input_shape=input_shape)
        
        layer_names = array(
            ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"])
        
        base_sub_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_names[3]).output)
        base_sub_model.trainable = False
        
        x = residual_block_resnet(base_sub_model.output, 512, name="conv5_block1")
        x = residual_block_resnet(x, 512, name="conv5_block2")
        output = residual_block_resnet(x, 512, name="conv5_block3")
        
        base_model = keras.Model(inputs=base_model.input, outputs=output)
    
    elif name == 'RESNet50':
        base_model = keras.applications.resnet.ResNet50(include_top=include_top, weights=weights,
                                                        input_shape=input_shape)
        
        layer_names = array(
            ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"])
        
        base_sub_model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_names[3]).output)
        base_sub_model.trainable = False
        
        x = residual_block_resnet(base_sub_model.output, 512, name="conv5_block1")
        x = residual_block_resnet(x, 512, name="conv5_block2")
        output = residual_block_resnet(x, 512, name="conv5_block3")
        
        base_model = keras.Model(inputs=base_model.input, outputs=output)
        
        # cut out down-sampling at conv5 as in CASENet paper:  # base_model_output1 = base_model.get_layer("conv4_block6_out").output  # base_model_input2 = base_model.get_layer("conv4_block5_out").input  # base_model1 = keras.Model(inputs=base_model.input, outputs=base_model_output1)  # base_model2 = keras.Model(inputs=base_model_input2, outputs=base_model.layers[-1].output)  #  # base_model2(base_model1.output)  # base_model = keras.Model(inputs=base_model1.input, outputs=base_model2.layers[-1].output)  #  # base_model.summary()
    
    elif name == 'MobileNetV2':
        base_model = keras.applications.MobileNetV2(include_top=include_top, weights=weights, input_shape=input_shape,
                                                    alpha=alpha)
        
        layer_names = array(
            ["Conv1", "expanded_conv_project_BN", "block_2_add", "block_5_add", "block_9_add", "block_12_add",
             "block_15_add", "block_16_project_BN", "out_relu"])
    
    else:
        raise ValueError("Backbone Network not defined")
    
    base_model.trainable = True
    if trainable_idx is not None:
        for layer in base_model.layers:
            layer.trainable = False
            if layer.name == layer_names[trainable_idx]:
                break
    
    if input_name == "in_img":
        names = ["base_model", "backbone"]
    else:
        names = ["base_model_prior", "backbone_prior"]
    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names[output_layer]]
    backbone = keras.Model(inputs=base_model.input, outputs=layers, name=names[0])
    
    input_model = keras.Input(shape=input_shape, name=input_name)
    x = backbone(input_model, training=True)
    backbone = keras.Model(input_model, x, name=names[1])
    
    return backbone, layer_names


def residual_block_resnet(x, num_input_filter, name='residual_block', filter_multiplication=4):
    if x.shape[-1] == num_input_filter * filter_multiplication:
        residual = x
    else:
        residual = utils.convolution_block(x, num_filters=num_input_filter * filter_multiplication, kernel_size=1,
                                           RELU=False)
    
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=1, name=name + '_1')
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=3, name=name + '_2')
    x = utils.convolution_block(x, num_filters=num_input_filter * filter_multiplication, kernel_size=1, RELU=False,
                                name=name + '_3')
    
    return keras.layers.Add(name=name + '_out')([x, residual])

def get_mobile_net(input_shape=(640, 360, 3), num_filters=8):
    input_model = keras.Input(shape=input_shape, name="in_img")
    
    conv_1 = utils.convolution_block(input_model, num_filters=8, kernel_size=3)
    conv_1 = utils.convolution_block(conv_1, depthwise=True)
    conv_1 = utils.convolution_block(conv_1, num_filters=8, kernel_size=1)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_1, strides=2, depth_multiplier=2)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, strides=1, depth_multiplier=6)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=6)
    conv_3 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=2, strides=2)
    
    return input_model, conv_1, conv_2, conv_3


def get_mobile_net_prior(input_shape=(640, 360, 3), num_filters=8):
    input_model = keras.Input(shape=input_shape, name="in_img")
    
    conv_1 = utils.convolution_block(input_model, num_filters=8, kernel_size=3)
    conv_1 = utils.convolution_block(conv_1, depthwise=True)
    conv_1 = utils.convolution_block(conv_1, num_filters=8, kernel_size=1)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_1, strides=2, depth_multiplier=2)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=8)
    # conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=6)
    conv_3 = utils.convolution_block(conv_2, RELU=False, BN=False, kernel_size=3, strides=2)
    
    return input_model, conv_1, conv_2, conv_3


def get_mobile_net_shifted(input_shape, edge_input, num_filters=8):
    input_model = keras.Input(shape=input_shape, name="in_img")
    
    edge_1 = utils.convolution_block(edge_input, kernel_size=3, num_filters=4)
    edge_1 = utils.convolution_block(edge_1, kernel_size=3, num_filters=4)
    conv_1 = utils.convolution_block(input_model, num_filters=8, kernel_size=3)
    conv_1 = utils.convolution_block(conv_1, depthwise=True)
    conv_1 = tf.keras.layers.Concatenate()([edge_1, conv_1])
    conv_1 = utils.convolution_block(conv_1, num_filters=8, kernel_size=1)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_1, strides=2, depth_multiplier=2)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, strides=1, depth_multiplier=6)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=6)
    conv_3 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=2, strides=2)
    
    return input_model, conv_1, conv_2, conv_3

def get_mobile_net_edge(input_img, input_edge):
    dim = 11
    horizontal = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    vertical = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    horizontal = tf.cast(tf.reshape(horizontal, [dim, 1, 1, 1]), tf.float32)
    vertical = tf.cast(tf.reshape(vertical, [1, dim, 1, 1]), tf.float32)
    
    edge_1 = tf.nn.conv2d(input_edge, horizontal, dilations=1, strides=[1, 1, 1, 1], padding="SAME")
    edge_1 = tf.nn.conv2d(edge_1, horizontal, dilations=1, strides=[1, 1, 1, 1], padding="SAME")
    edge_1 = tf.nn.conv2d(edge_1, vertical, dilations=1, strides=[1, 1, 1, 1], padding="SAME")
    edge_1 = tf.nn.conv2d(edge_1, vertical, dilations=1, strides=[1, 1, 1, 1], padding="SAME") + input_edge
    edge_1_x = tf.where(edge_1 >= 1.0, 1.0, -1.0)

    conv_1_edge = utils.convolution_block(input_img, num_filters=4, kernel_size=3, RELU=False)
    conv_1_edge = tf.keras.layers.Concatenate()([conv_1_edge, edge_1_x])
    conv_2_edge = utils.convolution_block(conv_1_edge, num_filters=4, kernel_size=3, RELU=False)
    conv_2_edge = utils.convolution_block(conv_2_edge, depthwise=True, kernel_size=3, strides=2)
    conv_2_edge = utils.convolution_block(conv_2_edge, num_filters=4, kernel_size=3, separable=True)
    conv_2_edge = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="SAME")(conv_2_edge)
    conv_2_edge = utils.convolution_block(conv_2_edge, num_filters=8, kernel_size=3, separable=True)
    conv_2_edge = utils.convolution_block(conv_2_edge, num_filters=8, kernel_size=3, separable=True)
    conv_2_edge = utils.convolution_block(conv_2_edge, num_filters=1, kernel_size=3, separable=True)
    conv_2_edge = tf.image.resize(conv_2_edge, (input_img.shape[1], input_img.shape[2]))
    conv_2_edge = tf.keras.layers.Activation("hard_sigmoid")(conv_2_edge)

    conv_1 = utils.convolution_block(input_img, num_filters=8, kernel_size=3, RELU=False)
    conv_1 = utils.convolution_block(conv_1, num_filters=8, kernel_size=3, separable=True)
    conv_2 = tf.keras.layers.Concatenate()([conv_1, conv_2_edge])
    conv_2 = utils.convolution_block(conv_2, num_filters=12, kernel_size=3, separable=True)
    conv_2 = utils.convolution_block(conv_2, num_filters=12, kernel_size=3, separable=True)
    conv_2 = utils.convolution_block(conv_2, num_filters=6, kernel_size=3, separable=True)

    conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(conv_2)

    
    # filters = tf.constant(
    #     [[[[1, 1.2, 1, 0]], [[2, 0.9, 0, -0.9]], [[1, 0, -1, -1.2]]],
    #      [[[0, 0.9, 2, 0.9]], [[0, 0, 0, 0]], [[0, -0.9, -2, -0.9]]],
    #      [[[-1, 0, 1, 1.2]], [[-2, -0.9, 0, 0.9]], [[-1, -1.2, -1, 0]]]],
    #     dtype=tf.float32)
    
    # conv_2 = tf.nn.conv2d(conv_2, filters, strides=[1, 1, 1, 1], padding="SAME")


    
    # filters = tf.constant(
    #     [[[[0]], [[0]], [[0]], [[1]], [[0]], [[0]], [[0]]],
    #      [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #      [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #      [[[1]], [[0]], [[0]], [[1]], [[0]], [[0]], [[1]]],
    #      [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #      [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
    #      [[[0]], [[0]], [[0]], [[1]], [[0]], [[0]], [[0]]]],
    #     dtype=tf.float32)
    
    # filters = tf.constant(
    #     [[[[0, 0, 0, 1]], [[0, 0, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 0]], [[0, 1, 0, 0]]],
    #      [[[0, 0, 0, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 0]], [[0, 1, 0, 0]], [[0, 0, 0, 0]]],
    #      [[[1, 0, 0, 0]], [[1, 0, 0, 0]], [[0, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]]],
    #      [[[0, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]], [[0, 0, 0, 0]]],
    #      [[[0, 1, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 1]]]],
    #     dtype=tf.float32)
    
    
    
    return tf.keras.layers.Activation(activation='sigmoid', name="out_edge")(conv_2), edge_1_x, conv_2_edge
