from tensorflow import keras
from numpy import array
from models.network_elements import utils


def backbone_edge_detection(input_shape=(640, 360, 3), num_filters=5):
    input_model = keras.Input(shape=input_shape, name="in_img")
    
    conv_1 = utils.convolution_block(input_model, num_filters=6 * num_filters, kernel_size=3, strides=2, RELU=False,
                                     BN=False)
    conv_2 = utils.mobile_net_v2_inverted_residual(conv_1, depth_multiplier=2)
    conv_3 = utils.mobile_net_v2_inverted_residual(conv_2, depth_multiplier=2, output_depth=4*num_filters)
    conv_4 = utils.mobile_net_v2_inverted_residual(conv_3, depth_multiplier=4)
    conv_5 = utils.mobile_net_v2_inverted_residual(conv_4, depth_multiplier=2, strides=2)
    conv_6 = utils.convolution_block(conv_5, num_filters=4 * num_filters, kernel_size=3, separable=True)
    
    return input_model, conv_3, conv_6


# Paper based backbones
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
