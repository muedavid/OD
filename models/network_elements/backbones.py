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
                                           RELU=False, name="residual")
    
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=1, name=name + '_1')
    x = utils.convolution_block(x, num_filters=num_input_filter, kernel_size=3, name=name + '_2')
    x = utils.convolution_block(x, num_filters=num_input_filter * filter_multiplication, kernel_size=1, RELU=False,
                                name=name + '_3')
    
    return keras.layers.Add(name=name + '_out')([x, residual])


def edge_map_preprocessing(input_layer, image_layer, output_shape, num_filters):
    edge = tf.expand_dims(input_layer, axis=-1)
    shift_up = tf.constant([0, 0, 0, 0, 1])
    shift_down = tf.constant([1, 0, 0, 0, 0])
    shift_left = tf.constant([0, 0, 0, 0, 1])
    shift_right = tf.constant([1, 0, 0, 0, 0])
    shift_up = tf.cast(tf.reshape(shift_up, [1, 5, 1, 1, 1]), tf.float32)
    shift_down = tf.cast(tf.reshape(shift_down, [1, 5, 1, 1, 1]), tf.float32)
    shift_left = tf.cast(tf.reshape(shift_left, [1, 1, 5, 1, 1]), tf.float32)
    shift_right = tf.cast(tf.reshape(shift_right, [1, 1, 5, 1, 1]), tf.float32)
    filters = [shift_up, shift_down, shift_right, shift_left]
    
    shifted = [edge]
    for i in range(len(filters)):
        shifted.append(tf.nn.conv3d(edge, filters[i], strides=[1, 1, 1, 1, 1], padding="SAME"))
    edge_shifted = tf.keras.layers.Concatenate(axis=-1)(shifted)
    edge_shifted = tf.keras.layers.Conv3D(4, kernel_size=1, padding="same", use_bias=True, activation="relu")(edge_shifted)
    
    shifted = [edge_shifted]
    for i in range(4):
        for f in filters:
            shifted.append(tf.nn.conv3d(edge_shifted[:, :, :, :, i:i + 1], f, strides=[1, 1, 1, 1, 1], padding="SAME"))
    edge_shifted = tf.keras.layers.Concatenate(axis=-1)(shifted)
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=1, padding="same", use_bias=True, activation="relu")(edge_shifted)
    edge_shifted = tf.keras.layers.Conv3D(5, kernel_size=3, padding="same", use_bias=True, activation="relu")(edge_shifted)
    
    shape = tf.shape(edge_shifted)
    x = tf.reshape(edge_shifted, [shape[0], shape[1], shape[2], shape[4]])
    print(x.shape)
    
    down_sampling = int(log2(x.shape[1] / output_shape[0]).tolist())
    if down_sampling % 1 != 0.0:
        raise ValueError("input shape of the edge map must be exact dividable by the output shape of the backbone")
    for i in range(down_sampling):
        x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, strides=2, use_bias=True,
                                    name="edge_map_down_sampling_{}".format(i))
    return x
