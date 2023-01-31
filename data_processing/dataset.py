import collections

import tensorflow as tf
import os
import os.path as osp
import numpy as np
import json

import data_processing.ds_augmentation as ds_aug
import utils.tools as tools


class DsKey:
    train = "TRAIN"
    test = "TEST"
    real_world = "REAL_WORLD"


class DataProcessing:
    image_count = 0
    img_idx = 0
    key = DsKey()
    paths = dict()
    ds_inf = dict()
    # vertices = collections.defaultdict(list)
    # vert_maps = collections.defaultdict(list)
    # vert_list = dict()
    inputs = list()
    outputs_ann = list()
    num_classes = collections.defaultdict(dict)
    input_data_cfg = dict()
    output_data_cfg = dict()
    
    def __init__(self, config_path: str):
        self.cfg = tools.config_loader(osp.join(config_path, 'dataset.yaml'))
        self.input_output_keys()
        self.rng = tf.random.Generator.from_seed(123, alg='philox')
    
    def load_dataset_information(self):
        """
        Loads the dataset.json file for each dataset type (Train, Test, IMG Only)
        """
        paths = dict()
        for key in ["TRAIN", "TEST"]:
            if self.cfg[key] is not None:
                paths[key] = osp.join(self.cfg["BASE_PATH_DATA"], self.cfg["NAME"], self.cfg[key]["NAME"])
        if self.cfg["REAL_WORLD"] is not None:
            paths["REAL_WORLD"] = self.cfg["REAL_WORLD"]["PATH"]
        self.paths['DATA'] = paths
        
        # Open JSON File with relative paths information
        for key_ds, val_ds in self.paths['DATA'].items():
            dataset_information_path = osp.join(val_ds, self.cfg["DATASET_JSON"])
            # The Dataset contains a json file
            if os.path.exists(dataset_information_path):
                f = open(dataset_information_path)
                ds_inf_tmp = json.load(f)
                f.close()
                
                # add a dictionary for image data
                self.paths[key_ds] = dict()
                self.ds_inf[key_ds] = dict()
                
                for key in ds_inf_tmp.keys():
                    # path to image data
                    if key == 'paths':
                        for key_paths, val_paths in ds_inf_tmp['paths'].items():
                            self.paths[key_ds][key_paths] = osp.join(val_ds, val_paths)
                    # additional information stored in json file
                    else:
                        self.ds_inf[key_ds][key] = ds_inf_tmp[key]
            
            # if self.cfg['VERT_LIST'] and key_ds != 'REAL_WORLD':
            #     with open(self.ds_inf['paths']['VERT'], 'r') as file:
            #         self.vert_list[key_ds] = yaml.safe_load(file)
            #     file.close()
    
    def load_dataset(self, ds_type: str, normalize: bool = True):
        """
        Loads the dataset given by ds_type and performs all sort of data processing.
        :param ds_type: Defines which of the following datasets (Train, Test, IMG only) is loaded
        :param normalize: If we should normalize the images to the range [-1, 1].
        """
        max_idx = min(self.ds_inf[ds_type]["info"]["num_frames"] - 1, self.cfg[ds_type]["MAX_IMG"] - 1)
        
        dataset = tf.data.Dataset.from_tensor_slices(range(max_idx))
        dataset = dataset.map(lambda x: self.parse_data(x, ds_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.cfg['out']['flow'] and ds_type != "REAL_WORLD":
            edge_dataset = self.load_flow_ds(ds_type, max_idx)
            dataset_combined = tf.data.Dataset.zip((dataset, edge_dataset))
            dataset = dataset_combined.map(self.combine_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        image_count = max_idx + 1
        
        print("The {mode} DS contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=ds_type))
        
        if self.cfg[ds_type]["CACHE"]:
            dataset = dataset.cache()
        if self.cfg[ds_type]["SHUFFLE"]:
            dataset = dataset.shuffle(image_count, reshuffle_each_iteration=True)
        if self.cfg[ds_type]["DATA_AUGMENTATION"]:
            dataset = dataset.map(lambda x: ds_aug.data_augmentation(x, self.rng, self.cfg[ds_type]["DATA_AUGMENTATION"]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if normalize:
            dataset = dataset.map(normalize_input_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: split_dataset_dictionary(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cfg[ds_type]["BATCH_SIZE"]:
            dataset = dataset.batch(self.cfg[ds_type]["BATCH_SIZE"])
        if self.cfg[ds_type]["PREFETCH"]:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        self.set_input_output_data_cfg()
        
        return dataset, image_count
    
    def parse_data(self, img_idx: int, ds_type: str):
        """
        Transformation function applied to each element of the dataset.
        Performs the basic data processing steps, such as image resizing, normalization, ...
        :param img_idx: current img_idx
        :param ds_type: Defines the type of the dataset: (Train, Test, IMG only)
        """
        img_idx_str = tf.strings.as_string(img_idx, width=4, fill='0')
        end_str = tf.constant(".png", dtype=tf.string)
        sep_str = tf.constant('/', dtype=tf.string)
        dataset_dict = dict()
        
        # LOAD IMG
        img_base_path = tf.constant(self.paths[ds_type]["IMG"], dtype=tf.string)
        img_path = tf.strings.join([img_base_path, sep_str, img_idx_str, end_str])
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.cfg["in"]["img"]["shape"], method="bilinear", antialias=True)
        image = tf.cast(image, tf.uint8)
        dataset_dict[self.cfg["in"]["img"]["name"]] = image
        
        if self.cfg["in"]["prior_img"] and self.paths[ds_type]["PRIOR_IMG"]:
            img_base_path = tf.constant(self.paths[ds_type]["PRIOR_IMG"], dtype=tf.string)
            img_path = tf.strings.join([img_base_path, sep_str, img_idx_str, end_str])
            image = tf.io.read_file(img_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, self.cfg["in"]["prior_img"]["shape"], method='bilinear')
            image = tf.cast(image, tf.uint8)
            dataset_dict[self.cfg["in"]["prior_img"]["name"]] = image
        
        if self.paths[ds_type]["PRIOR_ANN"]:
            # mask input:
            mask_base_path = tf.constant(self.paths[ds_type]['PRIOR_ANN'], dtype=tf.string)
            mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
            mask_input = tf.io.read_file(mask_path)
            mask_input = tf.image.decode_png(mask_input, channels=3)
            for mask_type in self.inputs:
                idx = self.ds_inf[ds_type]['info']['mask'][mask_type]
                mask = mask_input[:, :, idx:idx + 1]
                dataset_dict[self.cfg["in"][mask_type]["name"]] = \
                    self.preprocess_mask(mask, ds_type, mask_type, True)
        
        if self.paths[ds_type]["ANN"]:
            # mask output:
            mask_base_path = tf.constant(self.paths[ds_type]['ANN'], dtype=tf.string)
            mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
            mask_output = tf.io.read_file(mask_path)
            mask_output = tf.image.decode_png(mask_output, channels=3)
            for mask_type in self.outputs_ann:
                try:
                    idx = self.ds_inf[ds_type]['info']['mask'][mask_type]
                    mask = mask_output[:, :, idx:idx + 1]
                    dataset_dict[self.cfg["out"][mask_type]["name"]] = \
                        self.preprocess_mask(mask, ds_type, mask_type, False)
                except ValueError:
                    print("dataset.json file does not contain the required mask encoding: ", mask_type)

        # else:
        #     if self.paths[ds_type]["PRIOR_ANN"]:
        #         mask_base_path = tf.constant(self.paths[ds_type]['PRIOR_ANN'], dtype=tf.string)
        #         mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
        #         mask_input = tf.io.read_file(mask_path)
        #         mask_input = tf.image.decode_png(mask_input, channels=3)
        #         idx = self.ds_inf[ds_type]['info']['mask']['edge']
        #         mask = mask_input[:, :, idx:idx + 1]
        #         if self.cfg["in"]['edge'] is not None:
        #             dataset_dict[self.cfg["in"]['edge']["name"]] = \
        #                 self.preprocess_mask(mask, ds_type, "edge", True)
        #         elif self.cfg["in"]['contour'] is not None:
        #             dataset_dict[self.cfg["in"]['contour']["name"]] = \
        #                 self.preprocess_mask(mask, ds_type, "contour", True)
        
        return dataset_dict
    
    def preprocess_mask(self, mask: any, ds_type: str, mask_type: str, cfg_input_key: bool):
        """
        Performs ground truth data processing, such as resizing. Furthermore the labels are transformed based on
        the mask encoding (instance, category, binary) chosen in the config file.
        """
        cfg_first_key = "in" if cfg_input_key else "out"
        cfg_dataset = self.cfg[cfg_first_key][mask_type]
        
        # Binary: edge or not edge
        if cfg_dataset["mask_encoding"] == 2:
            mask = tf.cast(mask, tf.int32)
            mask = tf.where(mask > 0, 1, 0)
            mask = tf.cast(mask, tf.uint8)
            self.num_classes[cfg_first_key][mask_type] = 1 + (mask_type == "segmentation")
        
        # category
        elif cfg_dataset["mask_encoding"] == 1:
            mask = tf.cast(mask, tf.int32)
            
            if "cat2obj" in self.ds_inf[ds_type].keys():
                self.num_classes[cfg_first_key][mask_type] = len(self.ds_inf[ds_type]["cat2obj"]) + (
                        mask_type == "segmentation")
                for inst, cat in self.ds_inf[ds_type]["obj2cat"].items():
                    mask = tf.where(mask == int(inst), cat, mask)
                mask = tf.cast(mask, tf.uint8)
            else:
                self.num_classes[cfg_first_key][mask_type] = 1 + (mask_type == "segmentation")
                mask = tf.where(mask > 0, 1, 0)
        else:
            if "obj2cat" in self.ds_inf[ds_type].keys():
                self.num_classes[cfg_first_key][mask_type] = len(self.ds_inf[ds_type]["obj2cat"]) + (
                        mask_type == "segmentation")
            else:
                self.num_classes[cfg_first_key][mask_type] = 1 + (mask_type == "segmentation")
                mask = tf.where(mask > 0, 1, 0)
        
        # reshape:
        shape = tf.shape(mask)
        current_shape = (shape[0], shape[1])
        
        if mask_type == "segmentation":
            mask = tf.image.resize(mask, cfg_dataset["shape"], method="nearest")
            mask = apply_one_hot_encoding_to_segmentation_mask(mask, self.num_classes[cfg_first_key][mask_type])
        else:
            mask = apply_one_hot_encoding_to_mask(mask, self.num_classes[cfg_first_key][mask_type])
            mask = self.resize_label_map(mask, current_shape, self.num_classes[cfg_first_key][mask_type],
                                         cfg_dataset["shape"])
        return tf.cast(mask, tf.float32)
    
    def resize_label_map(self, label: any, current_shape_label: any, num_classes: int, mask_size: any, one_hot_encoded: bool = True):
        """
        Ground truth labels such as Edge Map are discrete values. In order to resize them, a special transformation is required.
        """
        # label 3D
        label = tf.cast(label, tf.int32)
        label = tf.expand_dims(label, axis=0)
        
        if not one_hot_encoded:
            class_range = tf.range(1, num_classes + 1)
            class_range_reshape = tf.reshape(class_range, [1, 1, 1, num_classes])
            label = tf.cast(class_range_reshape == label, dtype=tf.int32)
            pad = tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]])
            label = tf.pad(label, pad, "CONSTANT")
        
        edge_width_height = int(current_shape_label[0] / mask_size[0]) + 1
        edge_width_width = int(current_shape_label[1] / mask_size[1]) + 1
        kernel = tf.ones([edge_width_height, edge_width_width, num_classes + (one_hot_encoded is not True), 1],
                         tf.float32)
        label = tf.cast(label, tf.float32)
        label_widen = tf.nn.depthwise_conv2d(label, kernel, strides=[1, 1, 1, 1], padding="SAME")
        label_widen = tf.cast(tf.clip_by_value(label_widen, 0, 1), tf.int32)
        
        label_resized = tf.image.resize(label_widen, mask_size, method='nearest', antialias=True)
        
        if not one_hot_encoded:
            label_resized = tf.math.argmax(label_resized, axis=-1, output_type=tf.int32)
            label_resized = tf.expand_dims(label_resized, axis=-1)
        label = tf.squeeze(label_resized, axis=0)
        return label
    
    def combine_ds(self, dataset_dict: any, flow_field: any):
        """
        The flow field is loaded as an own dataset. This function combines the image and labels dataset with the flow field one.
        """
        flow_field = tf.image.resize(flow_field, self.cfg["out"]["flow"]["shape"], method='bilinear')[:, :, 0:2]
        ds_name = self.cfg["out"]["flow"]["name"]
        dataset_dict[ds_name] = flow_field
        if self.cfg["out"]["flow"]["only_edge"]:
            edge_label = tf.where(tf.reduce_sum(dataset_dict['in_edge'], axis=-1, keepdims=True) > 0, 1, 0)
            edge_label = self.resize_label_map(edge_label, (edge_label.shape[1], edge_label.shape[2]), 1,
                                               self.cfg["out"]["flow"]["shape"])
            dataset_dict[ds_name] = dataset_dict[ds_name] * tf.cast(edge_label, tf.float32)
        return dataset_dict
    
    def load_flow_ds(self, ds_type: str, max_idx: int):
        """
        Load the flow field dataset
        :param max_idx: max amount of flow field maps loaded
        :param ds_type: Defines the type of the dataset: (Train, Test, IMG only)
        """
        edge_stacked = []
        for i in range(max_idx):
            edge = np.load(self.paths[ds_type]['EDGE_FLOW'] + '/{:04}.npy'.format(i))
            edge = edge.astype(np.float32)
            edge[:, :, 0] = edge[:, :, 0] * self.cfg["in"]["edge"]["shape"][1]
            edge[:, :, 1] = edge[:, :, 1] * self.cfg["in"]["edge"]["shape"][0]
            edge_stacked.append(edge)
        edges = np.stack(edge_stacked, axis=0)
        edge_dataset = tf.data.Dataset.from_tensor_slices(edges)
        return edge_dataset
    
    def input_output_keys(self):
        """
        saves a list of inputs and outputs (edge, contour, segmentation) loaded. Defined in the config file.
        """
        candidates = ['edge', 'contour', 'segmentation']
        for c in candidates:
            if self.cfg['in'][c]:
                self.inputs.append(c)
            if self.cfg['out'][c]:
                self.outputs_ann.append(c)
    
    def set_input_output_data_cfg(self):
        """
        Defines the dictionaries: input_data_cfg and output_data_cfg.
        These two dictionaries contain relevant information for the network.
        """
        self.input_data_cfg = {
            "img": {"input_name": self.cfg["in"]["img"]["name"], "shape": self.cfg["in"]["img"]["shape"]}}
        if self.cfg["in"]["edge"]:
            self.input_data_cfg["edge"] = {"name": self.cfg["in"]["edge"]["name"],
                                           "shape": self.cfg["in"]["edge"]["shape"],
                                           "num_classes": self.num_classes["in"]["edge"]}
        if self.cfg["in"]["contour"]:
            self.input_data_cfg["edge"] = {"name": self.cfg["in"]["contour"]["name"],
                                           "shape": self.cfg["in"]["contour"]["shape"],
                                           "num_classes": self.num_classes["in"]["contour"]}
        
        self.output_data_cfg = dict()
        if self.cfg["out"]["edge"]:
            self.output_data_cfg["edge"] = {"name": self.cfg["out"]["edge"]["name"],
                                            "shape": self.cfg["out"]["edge"]["shape"],
                                            "num_classes": self.num_classes["out"]["edge"]}
        if self.cfg["out"]["contour"]:
            self.output_data_cfg["edge"] = {"name": self.cfg["out"]["contour"]["name"],
                                            "shape": self.cfg["out"]["contour"]["shape"],
                                            "num_classes": self.num_classes["out"]["contour"]}
        if self.cfg["out"]["segmentation"]:
            self.output_data_cfg["segmentation"] = {"name": self.cfg["out"]["segmentation"]["name"],
                                                    "shape": self.cfg["out"]["segmentation"]["shape"],
                                                    "num_classes": self.num_classes["out"]["segmentation"]}
        if self.cfg["out"]["flow"]:
            self.output_data_cfg["flow"] = {"name": self.cfg["out"]["flow"]["name"],
                                            "shape": self.cfg["out"]["flow"]["shape"]}


def split_dataset_dictionary(datapoint: any):
    """
    Keras model.fit() requires to split the input and output data into a tuple
    """
    datapoint_input = dict()
    datapoint_output = dict()
    for key in datapoint.keys():
        if 'in' in key:
            datapoint_input[key] = datapoint[key]
        else:
            datapoint_output[key] = datapoint[key]
    return datapoint_input, datapoint_output


def normalize_input_image(datapoint: any):
    """
    Apply image normalization to values in the range of [-1, 1]
    """
    datapoint['in_img'] = tf.cast(datapoint['in_img'], tf.float32) / 127.5 - 1.0
    return datapoint


def apply_one_hot_encoding_to_mask(mask: any, num_classes: int):
    """
    Integer encoded labels are one hot encoded (each class has own layer)
    """
    mask = tf.cast(mask, tf.int32)
    class_range = tf.range(1, num_classes + 1, dtype=tf.int32)
    class_range_reshape = tf.reshape(class_range, [1, 1, num_classes])
    mask = tf.cast(class_range_reshape == mask, dtype=tf.uint8)
    
    return mask


def apply_one_hot_encoding_to_segmentation_mask(mask, num_classes):
    """
    Integer encoded labels are one hot encoded (each class has own layer)
    """
    mask = tf.cast(mask, tf.int32)
    class_range = tf.range(0, num_classes, dtype=tf.int32)
    class_range_reshape = tf.reshape(class_range, [1, 1, num_classes])
    mask = tf.cast(class_range_reshape == mask, dtype=tf.uint8)
    
    return mask
