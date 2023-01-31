import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from models import model_data, network
from losses import edge_losses, flow_losses
from metrics import metrics
from utils import tools
from plots import edge_detection_plots
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
    
    def load_data(self, dataset_name: str):
        """
        Load the Model Data Class for saving and loading model related data to disk.
        """
        self.Data = model_data.ModelData(self.cfg["NAME"], dataset_name, make_dirs=True,
                                         del_old_ckpt=self.cfg["CALLBACKS"]["DEL_OLD_CKPT"] and self.cfg['TRAIN_MODEL'],
                                         del_old_tb=self.cfg["CALLBACKS"]["DEL_OLD_TB"] and self.cfg['TRAIN_MODEL'])
        
        self.train_model = self.cfg['TRAIN_MODEL']
    
    def get_neural_network_model(self, input_data_cfg: dict, output_data_cfg: dict):
        """
        returns the tensorflow keras model.
        """
        model = network.NN(self.cfg, input_data_cfg, output_data_cfg)
        return model.get_model()
    
    def get_best_model_from_checkpoints(self):
        """
        Checks all models saved at the checkpoints during training and loads the one with max f1 score on the validation dataset.
        """
        print(self.Data.get_model_path_max_f1())
        model = tf.keras.models.load_model(self.Data.get_model_path_max_f1(),
                                           custom_objects=self.custom_objects, compile=False)
        
        if self.cfg['SAVE']:
            model.save(self.Data.paths["MODEL"])
            model.trainable = False
            model.save(self.Data.paths["TFLITE"])
        
        return model
    
    def get_loss_function(self, output_data_cfg: dict):
        """
        loads the loss function specified in the cfg file.
        """
        loss_functions = dict()
        
        edge_loss_cfg = self.cfg['LOSS']['edge']
        if edge_loss_cfg:
            if edge_loss_cfg['focal']['apply'] == edge_loss_cfg['sigmoid']['apply']:
                raise ValueError("Only and at least one of those edge functions should be applied")
            if edge_loss_cfg['focal']['apply']:
                focal_cfg = edge_loss_cfg['focal']
                loss_functions[output_data_cfg["edge"]["name"]] = \
                    edge_losses.FocalLossEdges(power=focal_cfg['power'],
                                               edge_loss_weighting=focal_cfg['edge_loss_weighting'],
                                               min_edge_loss_weighting=focal_cfg['min_edge_loss_weighting'],
                                               max_edge_loss_weighting=focal_cfg['max_edge_loss_weighting'],
                                               padding=self.cfg['PADDING'],
                                               pixels_at_edge_without_loss=self.cfg['PIXELS_AT_EDGE_WITHOUT_LOSS'],
                                               decay=focal_cfg["decay"],
                                               focal_loss_derivative_threshold=focal_cfg[
                                                   'focal_loss_derivative_threshold'])
                self.custom_objects['FocalLossEdges'] = edge_losses.FocalLossEdges
            
            elif edge_loss_cfg['sigmoid']['apply']:
                sigmoid_cfg = edge_loss_cfg['sigmoid']
                loss_functions[output_data_cfg["edge"]["name"]] = edge_losses.WeightedMultiLabelSigmoidLoss(
                    sigmoid_cfg['min_edge_loss_weighting'],
                    sigmoid_cfg['max_edge_loss_weighting'],
                    sigmoid_cfg['class_individually_weighted'],
                    self.cfg['PADDING'], self.cfg['PIXELS_AT_EDGE_WITHOUT_LOSS'])
                
                self.custom_objects['WeightedMultiLabelSigmoidLoss'] = edge_losses.WeightedMultiLabelSigmoidLoss
        if self.cfg['LOSS']['flow']:
            loss_functions['out_flow'] = flow_losses.FlowLoss()
            self.custom_objects['FlowLoss'] = flow_losses.FlowLoss
        
        if self.cfg['LOSS']['segmentation']:
            loss_functions[output_data_cfg["segmentation"]["name"]] = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True)
        
        return loss_functions
    
    def get_metrics(self, output_data_cfg: dict):
        """
        loads the metrics specified in the cfg file.
        """
        metric_dic = dict()
        
        if self.cfg['LOSS']['edge']:
            metric_dic[output_data_cfg["edge"]["name"]] = [
                metrics.BinaryAccuracyEdges(num_classes=output_data_cfg['edge']['num_classes'],
                                            classes_individually=True,
                                            threshold_prediction=0.5,
                                            padding=self.cfg['PADDING'],
                                            pixels_at_edge_without_loss=self.cfg['PIXELS_AT_EDGE_WITHOUT_LOSS'],
                                            print_name="edge"),
                metrics.F1Edges(num_classes=output_data_cfg['edge']['num_classes'], classes_individually=True,
                                threshold_prediction=0.5,
                                padding=self.cfg['PADDING'],
                                pixels_at_edge_without_loss=self.cfg['PIXELS_AT_EDGE_WITHOUT_LOSS'],
                                print_name="edge")]
            
            self.custom_objects['BinaryAccuracyEdges'] = metrics.BinaryAccuracyEdges
            self.custom_objects['F1Edges'] = metrics.F1Edges
        if self.cfg['LOSS']['segmentation']:
            metric_dic[output_data_cfg["segmentation"]["name"]] = [tf.keras.metrics.CategoricalAccuracy()]
        return metric_dic
    
    def get_callbacks(self, f1_edge_logged: bool = True):
        """
        At the given checkpoint frequency set in the config file, the following callback are called.
        """
        # freq = int(np.ceil(img_count / self.bs) * self.cfg["CALLBACKS"]["CKPT_FREQ"]) + 1
        if f1_edge_logged:
            filepath = self.Data.paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1_edge:.4f}"
        else:
            filepath = self.Data.paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}"
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False, save_best_only=False, monitor="val_f1", verbose=1, save_freq='epoch',
            period=self.cfg["CALLBACKS"]["CKPT_FREQ"])]
        
        if self.cfg['CALLBACKS']['LOG_TB']:
            logdir = os.path.join(self.Data.paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1))
        
        return callbacks
    
    def get_lr(self, img_count: int, batch_size: int):
        """
        returns the learning rate schedule used during training
        """
        decay_step = np.ceil(img_count / batch_size) * self.cfg["EPOCHS"]
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(self.cfg['LR']['START'], decay_steps=decay_step,
                                                                    end_learning_rate=self.cfg['LR']['END'],
                                                                    power=self.cfg['LR']['POWER'])
        return lr_schedule
    
    def evaluate_and_plot_MF_score(self, model: any, dataset: any, num_classes: int, path: str,
                                   threshold_edge_width: float = 0.0):
        """
        evaluate the MF score on the given dataset.
        :param model: keras model
        :param dataset: keras dataset
        :param num_classes: performs evaluation for each class individually
        :param path: the evaluation is save as Figure to the given path
        :param threshold_edge_width: threshold on how far an edge pixel is allowed to be, such that its match still counts as true edge.
        """
        edge_detection_plots.plot_threshold_metrics_evaluation(model=model,
                                                               ds=dataset,
                                                               num_classes=num_classes,
                                                               classes_displayed_individually=True,
                                                               save=self.cfg["SAVE"],
                                                               path=path,
                                                               accuracy_y_lim_min=0.8,
                                                               padding=self.cfg["PADDING"],
                                                               pixels_at_edge_without_loss=self.cfg[
                                                                   'PIXELS_AT_EDGE_WITHOUT_LOSS'],
                                                               threshold_edge_width=threshold_edge_width)
    
    def convert_model_to_tflite(self, model: any):
        """
        convert the model to tflite and stores is to disk
        :param model: keras model
        """
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
