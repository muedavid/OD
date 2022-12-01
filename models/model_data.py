import os.path as osp
import os
import shutil


class ModelData:
    paths = dict()
    files = dict()
    
    def __init__(self, model, data, model_loaded=None, data_model_loaded=None, make_dirs=False, del_old_ckpt=False,
                 del_old_tb=True):
        self.path_definitions(model, data, model_loaded, data_model_loaded, make_dirs)
        self.clean_model_directories(del_old_ckpt, del_old_tb)
    
    def path_definitions(self, model, data, model_loaded=None, data_model_loaded=None, make_dirs=False):
        base_path_model = '/home/david/SemesterProject/Models'
        
        if data_model_loaded is None and model_loaded is not None:
            raise ValueError("Define the Dataset used to train the loaded model")
        
        self.paths = {
            'MODEL': osp.join(base_path_model, data, model) if data_model_loaded is None else osp.join(base_path_model,
                                                                                                       data_model_loaded,
                                                                                           model),
            'CKPT': osp.join(base_path_model, data, model, 'CKPT') if data_model_loaded is None else osp.join(
                base_path_model, data_model_loaded, model, 'CKPT'),
            'TBLOGS': osp.join(base_path_model, data, model, 'logs') if data_model_loaded is None else osp.join(
                base_path_model, data_model_loaded, model, 'logs'),
            'TFLITE': osp.join(base_path_model, data, model, 'TFLITE') if data_model_loaded is None else osp.join(
                base_path_model, data_model_loaded, model, 'TFLITE'),
            'FIGURES': osp.join(base_path_model, data, model, 'FIGURES') if data_model_loaded is None else osp.join(
                base_path_model, data_model_loaded, model, 'FIGURES'),
            'MODEL LOADED': osp.join(base_path_model, data_model_loaded, model_loaded) if type(
                model_loaded) == str else None}
        
        self.files = {'OUTPUT_TFLITE_MODEL': osp.join(self.paths['TFLITE'], model + '.tflite'),
                      'OUTPUT_TFLITE_MODEL_METADATA': osp.join(self.paths['TFLITE'], model + '_metadata.tflite'),
                      'OUTPUT_TFLITE_LABEL_MAP': osp.join(self.paths['TFLITE'], model + '_labels.txt'), }
        
        if make_dirs:
            for path in self.paths.keys():
                if path in ['MODEL', 'CKPT', 'TFLITE', 'TBLOGS', 'FIGURES']:
                    if not osp.exists(self.paths[path]):
                        print(path)
                        os.makedirs(self.paths[path])
    
    def clean_model_directories(self, del_old_checkpoints, del_old_tensorboard):
        if del_old_checkpoints:
            shutil.rmtree(self.paths['CKPT'])
            os.makedirs(self.paths['CKPT'])
        
        if del_old_tensorboard:
            shutil.rmtree(self.paths['TBLOGS'])
            os.makedirs(self.paths['TBLOGS'])
    
    def get_model_path_max_f1(self):
        model_path = None
        
        model_ckpt = os.listdir(self.paths['CKPT'])
        
        f1_max = 0
        for ckpt_name in model_ckpt:
            if float(ckpt_name[-4:]) >= f1_max:
                f1_max = float(ckpt_name[-4:])
                model_path = self.paths['CKPT'] + "/" + ckpt_name
        return model_path
