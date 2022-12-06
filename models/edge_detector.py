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
        else:
            raise ValueError('Model Architecture not implemented')
        return model
    
    def edge_detection_without_prior(self):
        output_filter_mult = 2
        inside_model_filter_mult = 3
        
        input_shape = (self.cfg["INPUT_SHAPE"][0], self.cfg["INPUT_SHAPE"][1], 3)
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
            x = pyramid_modules.pyramid_module(backbone.output[-1],
                                               num_filters=inside_model_filter_mult * self.num_classes)
        
        decoder_output = decoders.sged_decoder(x, [backbone.output[1]], output_dims=self.cfg["OUTPUT_SHAPE"],
                                               num_filters=inside_model_filter_mult * self.num_classes,
                                               num_output_filters=output_filter_mult * self.num_classes)
        
        side_output_1 = decoders.sged_side_feature(backbone.output[0], output_dims=self.cfg["OUTPUT_SHAPE"],
                                                   num_filters=output_filter_mult * self.num_classes, method="bilinear",
                                                   name="side1")
        side_output_2 = decoders.sged_side_feature(backbone.output[1], output_dims=self.cfg["OUTPUT_SHAPE"],
                                                   num_filters=output_filter_mult * self.num_classes, method="bilinear",
                                                   name="side2")
        
        side_outputs = [side_output_1, side_output_2]
        
        output = utils.shared_concatenation_and_classification(decoder_output, side_outputs, self.num_classes,
                                                               num_filters=output_filter_mult * self.num_classes,
                                                               name="out_edge")
        model = keras.Model(inputs=backbone.input, outputs=output)
        
        return model
    
    def edge_detection_with_prior(self):
        output_filter_mult = 2
        inside_model_filter_mult = 2
        num_filters = self.num_classes * inside_model_filter_mult
        num_filters_output = 2
        
        input_shape = (self.cfg["INPUT_SHAPE"][0], self.cfg["INPUT_SHAPE"][1], 3)
        input_edge_shape = (self.cfg["OUTPUT_SHAPE"][0], self.cfg["OUTPUT_SHAPE"][1], self.num_classes)
        
        print(input_shape)
        backbone, output_names = backbones.get_backbone(name=self.cfg["BACKBONE"]["NAME"],
                                                        weights=self.cfg["BACKBONE"]["WEIGHTS"],
                                                        input_shape=input_shape,
                                                        alpha=self.cfg["BACKBONE"]["ALPHA"],
                                                        output_layer=self.cfg["BACKBONE"]["OUTPUT_IDS"],
                                                        trainable_idx=self.cfg["BACKBONE"]["TRAIN_IDX"])
        
        input_model = keras.Input(shape=input_edge_shape, name='in_edge')
        edge_map = backbones.edge_map_preprocessing(input_model, backbone.output[1],
                                                    output_shape=backbone.output[-1].shape[1:],
                                                    num_classes=self.num_classes)
        
        x, mult = pyramid_modules.concatenate_edge_and_image(backbone.output[-1], edge_map,
                                                             num_classes=self.num_classes,
                                                             filter_mult=inside_model_filter_mult)
        
        decoder_output = decoders.sged_decoder(x, backbone.output[1],
                                               output_dims=self.cfg["OUTPUT_SHAPE"],
                                               num_filters=num_filters)
        
        side_output_1 = decoders.sged_side_feature(backbone.output[0], output_dims=self.cfg["OUTPUT_SHAPE"],
                                                   num_filters=num_filters, method="bilinear", name="side1")
        side_output_2 = decoders.sged_side_feature(backbone.output[1], output_dims=self.cfg["OUTPUT_SHAPE"],
                                                   num_filters=num_filters, method="bilinear", name="side2")
        
        side_outputs = [side_output_1, side_output_2]
        
        output = utils.shared_concatenation_and_classification(decoder_output, side_outputs, self.num_classes,
                                                               num_filters=num_filters_output, name="out_edge")
        model = keras.Model(inputs=[backbone.input, input_model], outputs=[output, mult])
        
        return model
