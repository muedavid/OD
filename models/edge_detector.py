from tensorflow import keras
from models.network_elements import backbones, decoders, utils, pyramid_modules, side_outputs, outputs


class EdgeDetector:
    
    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.num_classes = num_classes
    
    def get_model(self):
        if self.cfg['NAME'] == 'edge_detection_without_prior':
            model = self.edge_detection_without_prior()
        elif self.cfg['NAME'] == 'edge_detection_with_prior':
            model = self.edge_detection_with_prior()
        elif self.cfg['NAME'] == 'flow':
            model = self.flow()
        elif self.cfg['NAME'] == 'flow_edge':
            model = self.flow_edge()
        elif self.cfg['NAME'] == 'time':
            model = self.time_meas()
        else:
            raise ValueError('Model Architecture not implemented')
        return model
    
    def edge_detection_without_prior(self):
        if self.num_classes == 1:
            num_filter_per_class = 5
        else:
            num_filter_per_class = 2
        
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        print(input_shape)
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        # pyramid module for detection at various scale and larger field of view
        if self.cfg['MODEL']['DILATION']:
            x = pyramid_modules.daspp_efficient(backbone.output[-1],
                                                num_classes=self.num_classes,
                                                num_filters_per_class=num_filter_per_class)
        else:
            x = pyramid_modules.pyramid_module_small_backbone(backbone.output[-1],
                                                              num_classes=self.num_classes,
                                                              num_filters_per_class=num_filter_per_class)
        
        decoder_output = decoders.decoder_small(x, output_dims=self.cfg["OUTPUT_SHAPE"],
                                                num_classes=self.num_classes,
                                                num_filters_per_class=num_filter_per_class)
        
        sides = side_outputs.side_feature([backbone.output[0], backbone.output[1]],
                                          output_dims=self.cfg["OUTPUT_SHAPE"],
                                          num_classes=self.num_classes,
                                          num_filters_per_class=num_filter_per_class, name="side")
        
        output = outputs.shared_concatenation_and_classification(decoder_output, sides, num_classes=self.num_classes,
                                            num_filters_per_class=num_filter_per_class,
                                            output_name="out_edge")
        
        model = keras.Model(inputs=backbone.input, outputs=[output, sides, decoder_output, backbone.output[-1]])
        
        return model
    
    def edge_detection_with_prior(self):
        num_filters = 10
        
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        input_edge_shape = (self.cfg["INPUT_SHAPE_MASK"][0], self.cfg["INPUT_SHAPE_MASK"][1], self.num_classes)
        
        print(input_shape)
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        input_model = keras.Input(shape=input_edge_shape, name='in_edge')
        edge_map = backbones.edge_map_preprocessing_combined(input_model)
        # edge_map = backbones.edge_map_preprocessing(input_model)
        
        # x, pyramid_out, edge_layer_concat, image_layer_concat = pyramid_modules.concatenate_edge_and_image(
        #     backbone.output[1], edge_map, num_filters=num_filters)
        x, pyramid_out, edge_layer_concat, image_layer_concat = pyramid_modules.concatenate_edge_and_image(
            backbone.output[1], edge_map, num_filters=num_filters)
        
        decoder_output = decoders.decoder_small(x, output_dims=self.cfg["OUTPUT_SHAPE"], num_filters=5)
        
        sides = side_outputs.side_feature([backbone.output[0], backbone.output[1]],
                                          output_dims=self.cfg["OUTPUT_SHAPE"],
                                          num_filters=num_filters, method="bilinear", name="side")
        
        output = utils.shared_concatenation_and_classification(decoder_output, sides, self.num_classes,
                                                               num_filters=10,
                                                               name="out_edge")
        
        # model = keras.Model(inputs=[backbone.input, input_model],
        #                     outputs=[output, sides, decoder_output, x])
        model = keras.Model(inputs=[backbone.input, input_model],
                            outputs=[output, x, pyramid_out, edge_layer_concat, image_layer_concat, decoder_output,
                                     sides])
        
        return model
    
    def flow(self):
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        
        input_1, input_2, output = pyramid_modules.image_flow(input_shape)
        
        model = keras.Model(inputs=[input_1, input_2],
                            outputs=[output])
        
        return model
    
    def flow_edge(self):
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        input_edge_shape = (self.cfg["INPUT_SHAPE_MASK"][0], self.cfg["INPUT_SHAPE_MASK"][1], 1)
        input_edge = keras.Input(shape=input_edge_shape, name='in_edge')
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        output_flow, output_image, flow_max, flow_x = pyramid_modules.flow_edge(backbone.output[-1], input_edge)
        
        model = keras.Model(inputs=[backbone.input, input_edge],
                            outputs=[output_flow, output_image, flow_max, flow_x])
        
        return model
    
    def LiteEdge(self):
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        daspp_output = pyramid_modules.daspp(backbone.output[-1])
        
        decoder_output = decoders.lite_edge_decoder(daspp_input=daspp_output, side_input=backbone.output[0])
        
        output_shape = (backbone.output.shape[1], backbone.output.shape[2])
        side_1 = side_outputs.lite_edge_side_feature_extraction(backbone.output[0], output_shape)
        side_2 = side_outputs.lite_edge_side_feature_extraction(backbone.output[1], output_shape)
        side_3 = side_outputs.lite_edge_side_feature_extraction(backbone.output[2], output_shape)
        side_4 = side_outputs.lite_edge_side_feature_extraction(backbone.output[3], output_shape)
        side_5 = side_outputs.lite_edge_side_feature_extraction(backbone.output[4], output_shape)
        
        model = keras.Model(inputs=[backbone.input],
                            outputs=[output])
        
        return model
    
    def time_meas(self):
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        
        input_model = keras.layers.Input(input_shape, name="in_img")
        output_model = utils.time_testing_add(input_model)
        
        model = keras.Model(inputs=input_model,
                            outputs=output_model)
        
        return model
    
    def edge_detection_without_prior_old(self):
        if self.num_classes == 1:
            num_filter_per_class = 4
        else:
            num_filter_per_class = 2
        
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
        print(input_shape)
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        # pyramid module for detection at various scale and larger field of view
        if self.cfg['MODEL']['DILATION']:
            x = pyramid_modules.daspp_efficient(backbone.output[-1],
                                                num_classes=self.num_classes,
                                                num_filters_per_class=num_filter_per_class)
        else:
            x = pyramid_modules.pyramid_module_small_backbone(backbone.output[-1],
                                                              num_classes=self.num_classes,
                                                              num_filters_per_class=num_filter_per_class)
        
        decoder_output = decoders.decoder_small(x, output_dims=self.cfg["OUTPUT_SHAPE"],
                                                num_classes=self.num_classes,
                                                num_filters_per_class=num_filter_per_class)
        
        sides = side_outputs.side_feature([backbone.output[0], backbone.output[1]],
                                          output_dims=self.cfg["OUTPUT_SHAPE"],
                                          num_classes=self.num_classes,
                                          num_filters_per_class=num_filter_per_class, name="side")
        
        output = outputs.viot_fusion_module(decoder_output, sides, num_classes=self.num_classes,
                                            num_filters_per_class=num_filter_per_class,
                                            output_name="out_edge")
        
        model = keras.Model(inputs=backbone.input, outputs=[output, x, sides, decoder_output])
        
        return model
