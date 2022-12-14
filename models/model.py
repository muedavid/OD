import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from models import model_data, edge_detector
from losses import edge_losses
from metrics import metrics
from utils import tools
from tflite_support.metadata_writers import image_segmenter
from tflite_support.metadata_writers import writer_utils


class Model:
    custom_objects = dict()
    train_model = False
    Data = None
    
    def __init__(self, config_path):
        model_config_path = os.path.join(config_path, 'model.yaml')
        self.cfg = tools.config_loader(model_config_path)
        
        tf.random.set_seed(self.cfg['SEED'])
    
    def load_data(self, dataset_name):
        self.Data = model_data.ModelData(self.cfg["NAME"], dataset_name, make_dirs=True,
                                         del_old_ckpt=self.cfg["CALLBACKS"]["DEL_OLD_CKPT"] and self.cfg['TRAIN_MODEL'],
                                         del_old_tb=self.cfg["CALLBACKS"]["DEL_OLD_TB"] and self.cfg['TRAIN_MODEL'])
        
        self.train_model = self.cfg['TRAIN_MODEL']
    
    def get_neural_network_model(self, num_classes):
        if self.cfg['MODEL']['TYPE'] == 'edge detector':
            model = edge_detector.EdgeDetector(self.cfg, num_classes=num_classes)
            return model.get_model()
    
    def get_best_model_from_checkpoints(self):
        print(self.Data.get_model_path_max_f1())
        model = tf.keras.models.load_model(self.Data.get_model_path_max_f1(),
                                           custom_objects=self.custom_objects, compile=False)
        
        if self.cfg['SAVE']:
            model.save(self.Data.paths["MODEL"])
            model.trainable = False
            model.save(self.Data.paths["TFLITE"])
        
        return model
    
    def get_loss_function(self):
        loss_functions = dict()
        
        edge_loss_cfg = self.cfg['LOSS']['edge']
        if edge_loss_cfg:
            if edge_loss_cfg['focal']['apply'] == edge_loss_cfg['sigmoid']['apply']:
                raise ValueError("Only and at least one of those edge functions should be applied")
            if edge_loss_cfg['focal']['apply']:
                focal_cfg = edge_loss_cfg['focal']
                loss_functions['out_edge'] = edge_losses.FocalLossEdges(focal_cfg['power'],
                                                                        focal_cfg['edge_loss_weighting'],
                                                                        focal_cfg['min_edge_loss_weighting'],
                                                                        focal_cfg['max_edge_loss_weighting'])
                self.custom_objects['FocalLossEdges'] = edge_losses.FocalLossEdges
            
            elif edge_loss_cfg['sigmoid']['apply']:
                sigmoid_cfg = edge_loss_cfg['sigmoid']
                loss_functions['out_edge'] = edge_losses.WeightedMultiLabelSigmoidLoss(
                    sigmoid_cfg['min_edge_loss_weighting'],
                    sigmoid_cfg['max_edge_loss_weighting'],
                    sigmoid_cfg['class_individually_weighted'])
                
                self.custom_objects['WeightedMultiLabelSigmoidLoss'] = edge_losses.WeightedMultiLabelSigmoidLoss
        if self.cfg['LOSS']['flow_edge'] or self.cfg['LOSS']['flow_scene']:
            loss_functions['out_flow'] = edge_losses.FlowLoss()
            self.custom_objects['FlowLoss'] = edge_losses.FlowLoss
        return loss_functions
    
    def get_metrics(self, num_classes):
        metric_dic = dict()
        
        if self.cfg['LOSS']['edge']:
            metric_dic['out_edge'] = [metrics.BinaryAccuracyEdges(num_classes=num_classes, classes_individually=True,
                                                                  threshold_prediction=0.5),
                                      metrics.F1Edges(num_classes=num_classes, classes_individually=True,
                                                      threshold_prediction=0.5, threshold_edge_width=0)]
            
            self.custom_objects['BinaryAccuracyEdges'] = metrics.BinaryAccuracyEdges
            self.custom_objects['F1Edges'] = metrics.F1Edges
        
        return metric_dic
    
    def get_callbacks(self):
        # freq = int(np.ceil(img_count / self.bs) * self.cfg["CALLBACKS"]["CKPT_FREQ"]) + 1
        
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath=self.Data.paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
            save_weights_only=False, save_best_only=False, monitor="val_f1", verbose=1, save_freq='epoch',
            period=self.cfg["CALLBACKS"]["CKPT_FREQ"])]
        
        if self.cfg['CALLBACKS']['LOG_TB']:
            logdir = os.path.join(self.Data.paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1))
        
        return callbacks
    
    def get_lr(self, img_count, batch_size):
        decay_step = np.ceil(img_count / batch_size) * self.cfg["EPOCHS"]
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(self.cfg['LR']['START'], decay_steps=decay_step,
                                                                    end_learning_rate=self.cfg['LR']['END'],
                                                                    power=self.cfg['LR']['POWER'])
        return lr_schedule
    
    def convert_model_to_tflite(self, model):
        if not os.path.isfile(self.Data.files['OUTPUT_TFLITE_MODEL']):
            model.trainable = False
            model.save(self.Data.paths["TFLITE"])
        converter = tf.lite.TFLiteConverter.from_saved_model(self.Data.paths["TFLITE"])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)
        
        # Save the model.
        with open(self.Data.files['OUTPUT_TFLITE_MODEL'], 'wb') as f:
            f.write(tflite_model)
        print("saved")
        labels = []
        for name, label in self.cfg['CATEGORIES'].items():
            labels.append({'name': name, 'id': label})
        sorted_labels = sorted(labels, key=lambda d: d['id'])
        
        with open(self.Data.files['OUTPUT_TFLITE_LABEL_MAP'], 'w') as f:
            for label in sorted_labels:
                name = label['name']
                f.write(name + '\n')
    
    def convert_model_to_tflite_image_segmenter(self):
        image_segmenter_writer = image_segmenter.MetadataWriter
        _MODEL_PATH = self.Data.files['OUTPUT_TFLITE_MODEL']
        _SAVE_TO_PATH = self.Data.files['OUTPUT_TFLITE_MODEL_METADATA']
        _INPUT_NORM_MEAN = 0
        _INPUT_NORM_STD = 1
        _LABEL_FILE = self.Data.files['OUTPUT_TFLITE_LABEL_MAP']
        writer = image_segmenter_writer.create_for_inference(writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN],
                                                             [_INPUT_NORM_STD], [_LABEL_FILE])
        
        # Populate the metadata into the model.
        writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)
