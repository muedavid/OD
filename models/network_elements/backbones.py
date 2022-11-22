from tensorflow import keras
from numpy import array
import network_elements.features as feature


# TODO: Didn't manage to set the training parameter to False. Only possbile at call of a model. So pay attention if you
#  add additional layers to trainable. Their Batch norm layers will start to update to the training data and thus change the results for the already trained head. By using only very small learning rate this problem might be avoidable.


def get_backbone(name="MobileNetV2", weights="imagenet", height=None, width=None, alpha=1, channels=3,
                 output_layer=[2, 3, 4], trainable_idx=3):
    input_shape = [height, width, channels]
    include_top = False
    if name == 'RESNet101':
        base_model = keras.applications.resnet.ResNet101(include_top=include_top, weights=weights,
                                                         input_shape=input_shape)
        preprocessing = keras.resnet.preprocess_input
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
        preprocessing = keras.applications.resnet.preprocess_input
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
        # preprocessing = keras.applications.mobilenet_v2.preprocess_input
        layer_names = array(
            ["Conv1", "expanded_conv_project_BN", "block_2_add", "block_5_add", "block_9_add", "block_12_add",
             "block_15_add", "block_16_project_BN", "out_relu"])

    else:
        raise ValueError("Backbone Network not defined")

    base_model.trainable = True
    # if trainable_idx is not None:
    #     for layer in base_model.layers:
    #         layer.trainable = False
    #         if layer.name == layer_names[trainable_idx]:
    #             break

    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names[output_layer]]
    backbone = keras.Model(inputs=base_model.input, outputs=layers, name="base_model")

    input_model = keras.Input(shape=input_shape, name='in_img')
    # x = preprocessing(input_model)
    # x = backbone(x)
    x = backbone(input_model, training=True)
    backbone = keras.Model(input_model, x, name="backbone")

    return backbone, layer_names


# def get_partially_freeze_base_model(base_model, layer_names, input_shape, trainable_idx):
#     base_freeze_output_name = layer_names[trainable_idx]
#     stop = False
#     for layer in base_model.layers:
#         base_trainable_input_name = layer.name
#         if stop:
#             break
#         if layer.name == base_freeze_output_name:
#             stop = True
#     base_model_freeze_output = base_model.get_layer(base_freeze_output_name).output
#     base_model_trainable_input = base_model.get_layer(base_freeze_output_name).input
#
#     base_model_freeze = keras.Model(inputs=base_model.input, outputs=base_model_freeze_output, name="base_model_freeze")
#     base_model_freeze.trainable = False
#     base_model_trainable = keras.Model(inputs=base_model_trainable_input, outputs=base_model.output,
#                                        name="base_model_trainable")
#     base_model_trainable.trainable = True
#
#     return base_model_freeze, base_model_trainable


def residual_block_resnet(x, num_input_filter, name='residual_block', filter_multiplication=4):
    if x.shape[-1] == num_input_filter * filter_multiplication:
        residual = x
    else:
        residual = feature.convolution_block(x, num_input_filter * filter_multiplication, kernel_size=1, RELU=False)

    x = feature.convolution_block(x, num_input_filter, kernel_size=1, name=name + '_1')
    x = feature.convolution_block(x, num_input_filter, kernel_size=3, name=name + '_2')
    x = feature.convolution_block(x, num_input_filter * filter_multiplication, kernel_size=1, RELU=False,
                                  name=name + '_3')

    return keras.layers.Add(name=name + '_out')([x, residual])
