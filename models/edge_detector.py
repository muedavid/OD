from tensorflow import keras
from models.network_elements import backbones, decoders, utils, pyramid_modules


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
        else:
            raise ValueError('Model Architecture not implemented')
        return model
    
    def edge_detection_without_prior(self):
        num_filters = 10
        output_filter_mult = 2
        inside_model_filter_mult = 3
        
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
                                                num_filters=inside_model_filter_mult * self.num_classes)
        else:
            # x = pyramid_modules.pyramid_module(backbone.output[-1],
            #                                    num_filters=inside_model_filter_mult * self.num_classes)
            x = pyramid_modules.pyramid_module_small_backbone(backbone.output[-1],
                                                              num_filters=num_filters)
        
        # decoder_output = decoders.sged_decoder(x, backbone.output[1], output_dims=self.cfg["OUTPUT_SHAPE"],
        #                                        num_filters=inside_model_filter_mult * self.num_classes)
        decoder_output = decoders.decoder_small(x, output_dims=self.cfg["OUTPUT_SHAPE"], num_filters=5)
        
        # side_output_1 = decoders.sged_side_feature(backbone.output[0], output_dims=self.cfg["OUTPUT_SHAPE"],
        #                                            num_filters=output_filter_mult * self.num_classes, method="bilinear",
        #                                            name="side1")
        # side_output_2 = decoders.sged_side_feature(backbone.output[1], output_dims=self.cfg["OUTPUT_SHAPE"],
        #                                            num_filters=output_filter_mult * self.num_classes, method="bilinear",
        #                                            name="side2")
        
        sides = decoders.side_feature([backbone.output[0], backbone.output[1]], output_dims=self.cfg["OUTPUT_SHAPE"],
                                      num_filters=num_filters, method="bilinear", name="side")
        
        output = utils.shared_concatenation_and_classification(decoder_output, sides, self.num_classes,
                                                               num_filters=10,
                                                               name="out_edge")
        
        model = keras.Model(inputs=backbone.input, outputs=output)
        
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
        
        sides = decoders.side_feature([backbone.output[0], backbone.output[1]], output_dims=self.cfg["OUTPUT_SHAPE"],
                                      num_filters=num_filters, method="bilinear", name="side")
        
        output = utils.shared_concatenation_and_classification(decoder_output, sides, self.num_classes,
                                                               num_filters=10,
                                                               name="out_edge")
        
        # model = keras.Model(inputs=[backbone.input, input_model],
        #                     outputs=[output, sides, decoder_output, x])
        model = keras.Model(inputs=[backbone.input, input_model],
                            outputs=[output])
        
        return model

    def flow(self):
    
        input_shape = (self.cfg["INPUT_SHAPE_IMG"][0], self.cfg["INPUT_SHAPE_IMG"][1], 3)
    
        input_1, input_2, output = pyramid_modules.image_flow(input_shape)
    
        model = keras.Model(inputs=[input_1, input_2],
                            outputs=[output])
    
        return model
    