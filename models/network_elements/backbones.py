from tensorflow import keras
import tensorflow as tf
from numpy import array, log2
from models.network_elements import utils


def get_backbone(name="MobileNetV2", weights="imagenet", input_shape=(640, 360, 3), alpha=1,
                 output_layer=None, trainable_idx=None):
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
    
    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names[output_layer]]
    backbone = keras.Model(inputs=base_model.input, outputs=layers, name="base_model")
    
    input_model = keras.Input(shape=input_shape, name='in_img')
    x = backbone(input_model, training=True)
    backbone = keras.Model(input_model, x, name="backbone")
    
    return backbone, layer_names


def residual_block_resnet(x, num_input_filter, name='residual_block', filter_multiplication=4):
    if x.shape[-1] == num_input_filter * filter_multiplication:
        residual = x
    else:
        residual = utils.convolution_block(x, num_input_filter * filter_multiplication, kernel_size=1, RELU=False)
    
    x = utils.convolution_block(x, num_input_filter, kernel_size=1, name=name + '_1')
    x = utils.convolution_block(x, num_input_filter, kernel_size=3, name=name + '_2')
    x = utils.convolution_block(x, num_input_filter * filter_multiplication, kernel_size=1, RELU=False,
                                name=name + '_3')
    
    return keras.layers.Add(name=name + '_out')([x, residual])


def edge_map_preprocessing_3D(input_layer, image_layer, output_shape, num_classes):
    # for 3D convolution such that each channel has same filter
    x = tf.expand_dims(input_layer, axis=-1)
    
    x = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                               strides=1, use_bias=False, name='edge_input_1_conv')(x)
    x = tf.keras.layers.Conv3D(9, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(2, 2, 1), use_bias=False, name='edge_input_2_conv')(x)
    x = tf.keras.layers.Activation(activation='sigmoid')(x)
    
    image_layer = utils.convolution_block(image_layer, num_filters=num_classes, RELU=False,
                                          name='backbone_output_for_edge')
    image_layer = tf.keras.layers.Activation(activation='sigmoid')(image_layer)
    image_layer = tf.expand_dims(image_layer, axis=-1)
    x = tf.keras.layers.Concatenate(axis=-1)([x, image_layer])
    
    x = tf.keras.layers.Conv3D(3, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(2, 2, 1), use_bias=False, name='edge_image_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_1_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_1_relu')(x)
    
    x = tf.keras.layers.Conv3D(3, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=True, name='edge_image_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_2_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_2_relu')(x)
    
    down_sampling = int(log2(x.shape[1] / output_shape[0]).tolist())
    if down_sampling % 1 != 0.0:
        raise ValueError("input shape of the edge map must be exact dividable by the output shape of the backbone")
    for i in range(down_sampling):
        x = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                   strides=(2, 2, 1), use_bias=True, name="edge_map_down_sampling_{}".format(i))(x)
    
    x = tf.keras.layers.Conv3D(1, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=True, name='edge_backbone_out_conv')(x)
    x = tf.squeeze(x, axis=-1)
    x = tf.keras.layers.Activation(activation='sigmoid', name='edge_backbone_out_sigmoid')(x)
    return x


def edge_map_preprocessing(input_layer, image_layer, output_shape, num_classes):
    edge_map_1 = utils.convolution_block(input_layer, num_filters=num_classes, use_bias=True,
                                         name="edge_map_processing_1")
    edge_map_2 = utils.convolution_block(edge_map_1, kernel_size=3, num_filters=num_classes, use_bias=True, strides=2,
                                         name="edge_map_processing_2")
    edge_map_3 = utils.convolution_block(edge_map_2, kernel_size=5, num_filters=num_classes, use_bias=True, strides=2,
                                         name="edge_map_processing_3")
    # edge_map_4 = utils.convolution_block(edge_map_3, kernel_size=3, num_filters=num_classes, use_bias=True, strides=2,
    #                                      separable=True, name="edge_map_processing_4")
    #
    # image_layer = utils.convolution_block(image_layer, num_filters=2, RELU=True, BN=True,
    #                                       name='backbone_output_for_edge')
    # x = tf.keras.layers.Concatenate(axis=-1)([edge_map_3, image_layer])
    #
    # edge_image_1 = utils.convolution_block(x, kernel_size=3, num_filters=num_classes, use_bias=True, strides=1,
    #                                        name="edge_image_1")
    
    # edge_image_2 = utils.convolution_block(edge_image_1, kernel_size=3, num_filters=num_classes, use_bias=True, strides=2,
    #                                        separable=True, name="edge_image_2")
    # x = tf.keras.layers.Concatenate(axis=-1)([edge_map_4, edge_image_2])
    #
    # x = utils.convolution_block(x, kernel_size=3, num_filters=num_classes, use_bias=True, strides=1,
    #                             separable=True, name="edge_image_3")
    x = utils.convolution_block(edge_map_3, kernel_size=5, num_filters=num_classes, use_bias=True, strides=1,
                                name="edge_image_3")
    
    down_sampling = int(log2(x.shape[1] / output_shape[0]).tolist())
    if down_sampling % 1 != 0.0:
        raise ValueError("input shape of the edge map must be exact dividable by the output shape of the backbone")
    for i in range(down_sampling):
        x = utils.convolution_block(x, num_filters=num_classes, separable=True, strides=2,
                                    name="edge_map_down_sampling_{}".format(i))
    return x
