import tensorflow as tf
from models.network_elements import utils


def pyramid_module_small_backbone(pyramid_input, num_filters=5):
    x = utils.convolution_block(pyramid_input, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                separable=True, name="skip_1")
    x_1 = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                  separable=True, name="skip_2_1")
    x_1 = utils.convolution_block(x_1, num_filters=2 * num_filters, kernel_size=3, strides=1,
                                  separable=True, name="skip_2_2")
    x_2 = utils.convolution_block(x, num_filters=2, kernel_size=1, strides=1,
                                  separable=True, name="skip_3_1")
    x = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2])
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, strides=1,
                                separable=True, name="skip_out")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                separable=True, name="img_downsampling")
    
    dims = x.shape
    
    out_1 = utils.convolution_block(x, num_filters=3, kernel_size=1, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_1")
    out_2 = utils.convolution_block(x, num_filters=3, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_2")
    out_3 = utils.convolution_block(x, num_filters=2, kernel_size=5, dilation_rate=1, BN=True, RELU=True,
                                    separable=True,
                                    name="pyramid_3")
    out_4 = utils.convolution_block(x, num_filters=1, kernel_size=7, dilation_rate=1, BN=True, RELU=True,
                                    separable=True,
                                    name="pyramid_4")
    out = tf.keras.layers.AveragePooling2D(pool_size=(14, 14), strides=14,
                                           padding='SAME')(x)
    out = utils.convolution_block(out, num_filters=2, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="pyramid_avg")
    out = tf.keras.activations.sigmoid(out)
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3, out_4])
    
    x = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=1, name="pyramid_out_1")
    
    x = utils.convolution_block(x, num_filters=num_filters, kernel_size=1, name="pyramid_out_2")
    
    x = tf.image.resize(x, (pyramid_input.shape[1], pyramid_input.shape[2]))
    
    return x


def pyramid_module(pyramid_input, num_filters=12):
    x = utils.convolution_block(pyramid_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                separable=True, name="img_downsampling")
    
    dims = x.shape
    
    out_1 = utils.convolution_block(x, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_1")
    out_2 = utils.convolution_block(x, num_filters=num_filters, kernel_size=5, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_2")
    out_3 = utils.convolution_block(x, num_filters=num_filters, kernel_size=7, dilation_rate=1, BN=True, RELU=True,
                                    name="pyramid_3")
    out = tf.keras.layers.AveragePooling2D(pool_size=(int(dims[2] / 2), int(dims[2] / 2)), strides=int(dims[2] / 4),
                                           padding='SAME')(x)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=False,
                                  name="pyramid_avg")
    out = tf.keras.activations.sigmoid(out)
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3])
    
    x = utils.convolution_block(x, num_filters=2 * num_filters, kernel_size=3, name="pyramid_out")
    
    return x


def daspp_efficient(daspp_input, num_filters=12):
    daspp_input = utils.convolution_block(daspp_input, num_filters, kernel_size=3, dilation_rate=1, strides=2,
                                          separable=True)
    
    dims = daspp_input.shape
    
    out_1 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True,
                                    RELU=True, name="daspp_1")
    out_2 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=2, BN=True,
                                    RELU=True, name="daspp_2")
    out_3 = utils.convolution_block(daspp_input, num_filters=num_filters, kernel_size=3, dilation_rate=4, BN=True,
                                    RELU=True, name="daspp_3")
    out = tf.keras.layers.AveragePooling2D(pool_size=(int(dims[2] / 2), int(dims[2] / 2)), strides=int(dims[2] / 4))(
        daspp_input)
    out = utils.convolution_block(out, num_filters=num_filters, kernel_size=3, dilation_rate=1, BN=True, RELU=True,
                                  name="daspp_avg")
    
    out_pool = tf.image.resize(out, (dims[1], dims[2]))
    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_2, out_3])
    
    x = utils.convolution_block(x, kernel_size=1, name="daspp_out")
    
    return x


def concatenate_edge_and_image_mult(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = image_layer.shape[1] / edge_layer.shape[1]
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=small_filter_num,
                                              kernel_size=int(down_sampling + 1),
                                              strides=int(down_sampling),
                                              separable=True, name="image_down_sampling")
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True, name='pyramid_module_image_1_1')
    # image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3,
    #                                              separable=True,
    #                                              name='pyramid_module_image_1_2')
    # edge_layer_concat = utils.convolution_block(edge_layer, num_filters=small_filter_num, kernel_size=3, RELU=True,
    #                                             BN=True,
    #                                             name='pyramid_module_edge_1')
    
    image = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_1")
    image = utils.convolution_block(image, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2_2")
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer])
    image_edge_concat = utils.convolution_block(x, num_filters=2 * small_filter_num, kernel_size=1,
                                                name="pyramid_module_pyramid_concat")
    
    pyramid = utils.convolution_block(image_edge_concat, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
                                      num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=2)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=2,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=2,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=2,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    pyramid_out_mult = utils.convolution_block(pyramid_out, name="pyramid_out_mult", num_filters=3, kernel_size=1)
    pyramid_out_concat = utils.convolution_block(pyramid_out, name="pyramid_out_concat", num_filters=3, kernel_size=1)
    image_mult = utils.convolution_block(image_layer, name="image_layer", num_filters=3, kernel_size=1)
    
    out = []
    for i in range(3):
        mult = tf.slice(pyramid_out_mult, begin=[0, 0, 0, i], size=[-1, -1, -1, 1],
                        name='pyramid_module_pyramid_out_{}'.format(i))
        out.append(tf.multiply(mult, image_mult, name='pyramid_module_multiplication_{}'.format(i)))
    
    after_mult = tf.keras.layers.Concatenate(axis=-1)(out)
    after_mult = utils.convolution_block(after_mult, num_filters=small_filter_num, BN=True, RELU=True,
                                         kernel_size=1,
                                         name="pyramid_module_after_mult_1")
    after_mult = utils.convolution_block(after_mult, num_filters=3, BN=True, RELU=True, kernel_size=3,
                                         separable=True, name="pyramid_module_after_mult_2")
    
    image = utils.convolution_block(image, kernel_size=3, num_filters=2,
                                    name="pyramid_module_image_2_3")
    out = tf.keras.layers.Concatenate(axis=-1)([after_mult, image, pyramid_out_concat])
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num, name="pyramid_module_out_2")
    
    return out, pyramid_out, edge_layer, image_layer_concat, after_mult


def concatenate_edge_and_image(image_layer, edge_layer, num_filters):
    small_filter_num = 5
    
    down_sampling = int(image_layer.shape[1] / edge_layer.shape[1])
    if down_sampling > 1:
        image_layer = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=down_sampling + 1,
                                              strides=down_sampling,
                                              separable=True, name="pyramid_module_image_down_sampling")
    
    image_layer_concat = utils.convolution_block(image_layer, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True,
                                                 name='pyramid_module_image_1_1')
    image_layer_concat = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3,
                                                 separable=True,
                                                 name='pyramid_module_image_1_2')
    edge_layer_concat = utils.convolution_block(edge_layer, num_filters=small_filter_num, kernel_size=3, RELU=True,
                                                BN=True,
                                                name='pyramid_module_edge_1')
    
    edge = utils.convolution_block(edge_layer_concat, num_filters=small_filter_num, kernel_size=3, separable=True,
                                   name="pyramid_module_edge_2")
    image = utils.convolution_block(image_layer_concat, num_filters=small_filter_num, kernel_size=3, separable=True,
                                    name="pyramid_module_image_2")
    
    x = tf.keras.layers.Concatenate(axis=-1)([image_layer_concat, edge_layer_concat])
    pyramid = utils.convolution_block(x, num_filters=small_filter_num, kernel_size=1,
                                      name="pyramid_module_pyramid_concat")
    
    # pyramid = utils.convolution_block(pyramid, name="pyramid_module_pyramid_down", strides=2, kernel_size=3,
    #                                   num_filters=small_filter_num, separable=True)
    pyramid_1 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_1", kernel_size=1, num_filters=4)
    pyramid_2 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_2", kernel_size=3, num_filters=4,
                                        separable=True)
    pyramid_3 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_3", kernel_size=5, num_filters=2,
                                        separable=True)
    pyramid_4 = utils.convolution_block(pyramid, name="pyramid_module_pyramid_4", kernel_size=7, num_filters=1,
                                        separable=True)
    pyramid_out = tf.keras.layers.Concatenate(axis=-1)([pyramid_1, pyramid_2, pyramid_3, pyramid_4])
    pyramid_out = utils.convolution_block(pyramid_out, name="pyramid_module_pyramid_out", num_filters=small_filter_num,
                                          kernel_size=1)
    
    pyramid_out = tf.image.resize(pyramid_out, (image_layer.shape[1], image_layer.shape[2]), method="bilinear")
    
    out = tf.keras.layers.Concatenate(axis=-1)([pyramid_out, image])
    out = utils.convolution_block(out, kernel_size=1, num_filters=2 * small_filter_num, name="pyramid_module_out_1")
    out = utils.convolution_block(out, kernel_size=3, num_filters=small_filter_num, name="pyramid_module_out_2")
    out = utils.convolution_block(out, kernel_size=1, num_filters=small_filter_num,
                                  name="pyramid_module_out_3")
    
    return out, pyramid_out, edge_layer_concat, image_layer_concat

def image_flow(img_shape):
    # TODO(DAVID) skip connection extremely important, as elsewhere vanishing gradients
    image_layer_1 = tf.keras.layers.Input(shape=img_shape, name="in_img")
    image_layer_2 = tf.keras.layers.Input(shape=img_shape, name="in_prior_img")
    image_concat = tf.keras.layers.Concatenate(axis=-1)([image_layer_1, image_layer_2])
    flow = utils.convolution_block(image_concat, kernel_size=11, num_filters=12, name="d_1",
                                   strides=2, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=9, num_filters=24, name="d_2",
                                   strides=2, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=7, num_filters=48, name="d_3",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=5, num_filters=96, name="d_4",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_5",
                                   strides=1, padding="VALID", BN=False)
    flow_x = utils.convolution_block(flow, kernel_size=1, num_filters=96, name="d_8",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow_x, kernel_size=1, num_filters=96, name="d_9",
                                   strides=1, padding="VALID", BN=False)
    flow_avg = tf.keras.layers.AveragePooling2D((4,4),strides=4)(flow)
    flow_avg = utils.convolution_block(flow_avg, kernel_size=3, num_filters=96, name="avg_1", BN=False)
    flow_avg = utils.convolution_block(flow_avg, kernel_size=3, num_filters=96, name="avg_2", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_6",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=96, name="d_7",
                                   strides=1, padding="VALID", BN=False)
    flow_avg = tf.image.resize(flow_avg, (flow.shape[1], flow.shape[2]))
    flow = tf.keras.layers.Concatenate(axis=-1)([flow, flow_avg])
    flow = utils.convolution_block(flow, kernel_size=7, num_filters=48, name="d_11",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=5, num_filters=24, name="d_12",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=3, num_filters=24, name="d_13",
                                   strides=1, padding="VALID", BN=False)
    flow = utils.convolution_block(flow, kernel_size=1, num_filters=24, name="d_14",
                                   strides=1, padding="VALID", BN=False)
    flow_x = tf.image.resize(flow_x, (80, 45))
    out = tf.image.resize(flow, (80, 45))
    out = tf.keras.layers.Concatenate(axis=-1)([flow_x, out])
    out = utils.convolution_block(out, kernel_size=1, num_filters=24, separable=True, name="out_1", BN=False)
    out = utils.convolution_block(out, kernel_size=7, num_filters=8, separable=True, name="out_2", BN=False)
    out = tf.keras.layers.Conv2D(filters=2, kernel_size=1, name="out_flow")(out)
    return image_layer_1, image_layer_2, out