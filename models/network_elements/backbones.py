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
        residual = utils.convolution_block(x, num_filters=num_input_filter * filter_multiplication, kernel_size=1,
                                           RELU=False)
    
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=1, name=name + '_1')
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=3, name=name + '_2')
    x = utils.convolution_block(x, num_filters=num_input_filter * filter_multiplication, kernel_size=1, RELU=False,
                                name=name + '_3')
    
    return keras.layers.Add(name=name + '_out')([x, residual])


def edge_map_preprocessing_3D(input_layer, image_layer, output_shape, num_classes):
    # for 3D convolution such that each channel has same filter
    edge = tf.expand_dims(input_layer, axis=-1)
    
    edge = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                  strides=1, use_bias=False, name='edge_input_1_conv')(edge)
    edge = tf.keras.layers.ReLU(name='edge_input_1_relu')(edge)
    edge = tf.keras.layers.BatchNormalization(name='edge_input_1_bn')(edge)
    edge = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                  strides=(2, 2, 1), use_bias=False, name='edge_input_2_conv')(edge)
    edge = tf.keras.layers.ReLU(name='edge_input_2_relu')(edge)
    edge = tf.keras.layers.Conv3D(3, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                                  strides=(2, 2, 1), use_bias=False, name='edge_input_3_conv')(edge)
    edge = tf.keras.layers.BatchNormalization(name='edge_input_2_bn')(edge)
    
    image_layer = utils.convolution_block(image_layer, num_filters=num_classes, strides=2, kernel_size=5,
                                          name='backbone_output_for_edge_1')
    image_layer = utils.convolution_block(image_layer, num_filters=num_classes, RELU=False,
                                          name='backbone_output_for_edge_2')
    image_layer = tf.expand_dims(image_layer, axis=-1)
    x = tf.keras.layers.Concatenate(axis=-1)([edge, image_layer])
    
    x = tf.keras.layers.Conv3D(1, kernel_size=(5, 5, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=False, name='edge_image_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_1_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_1_relu')(x)
    
    x = tf.keras.layers.Conv3D(1, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                               strides=(1, 1, 1), use_bias=True, name='edge_image_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='edge_image_2_bn')(x)
    x = tf.keras.layers.ReLU(name='edge_image_2_relu')(x)
    
    down_sampling = int(log2(x.shape[1] / output_shape[0]).tolist())
    if down_sampling % 1 != 0.0:
        raise ValueError("input shape of the edge map must be exact dividable by the output shape of the backbone")
    for i in range(down_sampling):
        x = tf.keras.layers.Conv3D(3, kernel_size=(3, 3, 1), dilation_rate=1, padding='same',
                                   strides=(2, 2, 1), use_bias=True, name="edge_map_down_sampling_{}".format(i))(x)
    x = tf.squeeze(x, axis=-1)
    return x


def edge_map_preprocessing(input_layer):
    num_filters = 5
    edge_map_1 = utils.convolution_block(input_layer, kernel_size=3, num_filters=num_filters, use_bias=True,
                                         name="edge_map_processing_1")
    edge_map_2 = utils.convolution_block(edge_map_1, kernel_size=3, num_filters=num_filters, use_bias=True,
                                         strides=1, name="edge_map_processing_2")
    edge_map_3 = utils.convolution_block(edge_map_2, kernel_size=5, num_filters=num_filters, use_bias=True,
                                         name="edge_map_processing_3")
    edge_map_4 = utils.convolution_block(edge_map_3, kernel_size=5, num_filters=num_filters, use_bias=True, strides=1,
                                         separable=True, name="edge_map_processing_4")
    edge_map_5 = tf.keras.layers.Concatenate(axis=-1)([edge_map_2, edge_map_4])
    edge_map_6 = utils.convolution_block(edge_map_5, kernel_size=1, num_filters=num_filters, use_bias=True, strides=1,
                                         separable=True, name="edge_map_processing_6")
    return edge_map_6


def edge_map_preprocessing_shifted(input_layer):
    edge = tf.expand_dims(input_layer, axis=-1)
    dim = 15
    shift_up = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_down = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_left = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    shift_right = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [1, dim, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [1, dim, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, 1, dim, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, 1, dim, 1, 1]), tf.float32)
    shift_direction_to_filter = {"up": shift_up, "down": shift_down, "left": shift_left, "right": shift_right}
    shift_pattern = {"up": ["up", "left", "right"],
                     "down": ["down", "left", "right"],
                     "left": ["left"],
                     "right": ["right"]}
    
    shifted = [edge]
    for shift_direction_first, following_shift_direction in shift_pattern.items():
        conv_filter = shift_direction_to_filter[shift_direction_first]
        edge_shifted = tf.nn.conv3d(edge, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME")
        shifted.append(edge_shifted)
        for shift_direction_second in following_shift_direction:
            conv_filter = shift_direction_to_filter[shift_direction_second]
            shifted.append(tf.nn.conv3d(edge_shifted, conv_filter, strides=[1, 1, 1, 1, 1], padding="SAME"))
    
    edge_shifted = tf.keras.layers.Concatenate(axis=-1, name="concat_shifted_filter")(shifted)
    
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=1, padding="same", use_bias=True, activation="relu")(
        edge_shifted)
    edge_shifted = tf.keras.layers.BatchNormalization(name='edge_input_3_bn')(edge_shifted)
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=3, padding="same", use_bias=True,
                                          activation="relu")(
        edge_shifted)
    
    shape = tf.shape(edge_shifted)
    x = tf.reshape(edge_shifted, [shape[0], shape[1], shape[2], shape[3]*shape[4]])
    return x
