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
    img_only = "IMG_ONLY"


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
    
    def __init__(self, config_path):
        self.cfg = tools.config_loader(osp.join(config_path, 'dataset.yaml'))
        self.input_output_keys()
        self.rng = tf.random.Generator.from_seed(123, alg='philox')
    
    def path_definitions(self):
        paths = dict()
        for key in ["TRAIN", "TEST"]:
            if self.cfg[key] is not None:
                paths[key] = osp.join(self.cfg["BASE_PATH_DATA"], self.cfg["NAME"], self.cfg[key]["NAME"])
        if self.cfg["IMG_ONLY"] is not None:
            paths["IMG_ONLY"] = osp.join(self.cfg["BASE_PATH_DATA"], self.cfg["IMG_ONLY"]["PATH"])
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
            
            # if self.cfg['VERT_LIST'] and key_ds != 'IMG_ONLY':
            #     with open(self.ds_inf['paths']['VERT'], 'r') as file:
            #         self.vert_list[key_ds] = yaml.safe_load(file)
            #     file.close()
    
    def load_dataset(self, ds_type, normalize=True):
        max_idx = min(self.ds_inf[ds_type]["info"]["num_frames"] - 1, self.cfg[ds_type]["MAX_IMG"] - 1)
        
        dataset = tf.data.Dataset.from_tensor_slices(range(max_idx))
        dataset = dataset.map(lambda x: self.parse_data(x, ds_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.cfg['out']['flow'] and ds_type != "IMG_ONLY":
            edge_dataset = self.load_flow_ds(ds_type, max_idx)
            dataset_combined = tf.data.Dataset.zip((dataset, edge_dataset))
            dataset = dataset_combined.map(self.combine_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        image_count = max_idx + 1
        
        print("The {mode} DS contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=ds_type))
        
        if self.cfg[ds_type]["CACHE"]:
            dataset = dataset.cache()
        if self.cfg[ds_type]["SHUFFLE"]:
            dataset = dataset.shuffle(image_count, reshuffle_each_iteration=True)
        if self.cfg[ds_type]["DATA_AUG"]:
            dataset = dataset.map(lambda x: ds_aug.augment_mapping(x, self.rng, self.cfg[ds_type]["DATA_AUG"]),
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
    
    def parse_data(self, img_idx, ds_type):
        img_idx_str = tf.strings.as_string(img_idx, width=4, fill='0')
        end_str = tf.constant(".png", dtype=tf.string)
        sep_str = tf.constant('/', dtype=tf.string)
        dataset_dict = dict()
        
        # LOAD IMG
        img_base_path = tf.constant(self.paths[ds_type]["IMG"], dtype=tf.string)
        img_path = tf.strings.join([img_base_path, sep_str, img_idx_str, end_str])
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.cfg["in"]["img"]["shape"], method="area")
        image = tf.cast(image, tf.uint8)
        dataset_dict[self.cfg["in"]["img"]["name"]] = image
        
        if ds_type != 'IMG_ONLY':
            if self.cfg["in"]["prior_img"]:
                img_base_path = tf.constant(self.paths[ds_type]["PRIOR_IMG"], dtype=tf.string)
                img_path = tf.strings.join([img_base_path, sep_str, img_idx_str, end_str])
                image = tf.io.read_file(img_path)
                image = tf.image.decode_png(image, channels=3)
                image = tf.image.resize(image, self.cfg["in"]["prior_img"]["shape"], method='area')
                image = tf.cast(image, tf.uint8)
                dataset_dict[self.cfg["in"]["prior_img"]["name"]] = image
            
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
            
            # mask output:
            mask_base_path = tf.constant(self.paths[ds_type]['ANN'], dtype=tf.string)
            mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
            mask_output = tf.io.read_file(mask_path)
            mask_output = tf.image.decode_png(mask_output, channels=3)
            # unrolled for loop, execute for each statement individually and also if statements (python side effect)
            for mask_type in self.outputs_ann:
                idx = self.ds_inf[ds_type]['info']['mask'][mask_type]
                mask = mask_output[:, :, idx:idx + 1]
                dataset_dict[self.cfg["out"][mask_type]["name"]] = \
                    self.preprocess_mask(mask, ds_type, mask_type, False)
        else:
            if self.paths[ds_type]["PRIOR_ANN"]:
                mask_base_path = tf.constant(self.paths[ds_type]['PRIOR_ANN'], dtype=tf.string)
                mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
                mask_input = tf.io.read_file(mask_path)
                mask_input = tf.image.decode_png(mask_input, channels=3)
                idx = self.ds_inf[ds_type]['info']['mask']['edge']
                mask = mask_input[:, :, idx:idx + 1]
                dataset_dict[self.cfg["in"]['edge']["name"]] = \
                    self.preprocess_mask(mask, ds_type, "edge", True)
                
        return dataset_dict
    
    def preprocess_mask(self, mask, ds_type, mask_type, cfg_input_key):
        # Python side effect ok as value does not change
        cfg_first_key = "in" if cfg_input_key else "out"
        cfg_dataset = self.cfg[cfg_first_key][mask_type]
        
        # Binary: edge or not edge
        if cfg_dataset["mask_encoding"] == 2:
            mask = tf.cast(mask, tf.int32)
            mask = tf.where(mask > 0, 1, 0)
            mask = tf.cast(mask, tf.uint8)
            self.num_classes[cfg_first_key][mask_type] = 1
        
        # category
        elif cfg_dataset["mask_encoding"] == 1:
            mask = tf.cast(mask, tf.int32)
            
            for inst, cat in self.ds_inf[ds_type]["obj2cat"].items():
                mask = tf.where(mask == int(inst), cat, mask)
            mask = tf.cast(mask, tf.uint8)
            
            self.num_classes[cfg_first_key][mask_type] = len(self.ds_inf[ds_type]["cat2obj"])
            
            mask = reshape_mask(mask, self.num_classes[cfg_first_key][mask_type])
        
        else:
            self.num_classes[cfg_first_key][mask_type] = len(self.ds_inf[ds_type]["obj2cat"])
            mask = reshape_mask(mask, self.num_classes[cfg_first_key][mask_type])
        
        # reshape:
        shape = tf.shape(mask)
        current_shape = (shape[0], shape[1])
        
        if mask_type == "segmentation":
            mask = tf.image.resize(mask, cfg_dataset["shape"], method="nearest")
        else:
            mask = self.resize_label_map(mask, current_shape, self.num_classes[cfg_first_key][mask_type],
                                         cfg_dataset["shape"])
        return tf.cast(mask, tf.float32)
    
    def resize_label_map(self, label, current_shape_label, num_classes, mask_size, already_reshaped=True):
        # label 3D
        label = tf.cast(label, tf.int32)
        label = tf.expand_dims(label, axis=0)
        
        if not already_reshaped:
            class_range = tf.range(1, num_classes + 1)
            class_range_reshape = tf.reshape(class_range, [1, 1, 1, num_classes])
            label = tf.cast(class_range_reshape == label, dtype=tf.int32)
            pad = tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]])
            label = tf.pad(label, pad, "CONSTANT")
        
        edge_width_height = int(current_shape_label[0] / self.cfg["out"]["edge"]["shape"][0]) + 1
        edge_width_width = int(current_shape_label[1] / self.cfg["out"]["edge"]["shape"][1]) + 1
        kernel = tf.ones([edge_width_height, edge_width_width, num_classes + (already_reshaped is not True), 1],
                         tf.float32)
        label_pad = tf.cast(label, tf.float32)
        label_widen = tf.nn.depthwise_conv2d(label_pad, kernel, strides=[1, 1, 1, 1], padding="SAME")
        label_widen = tf.cast(tf.clip_by_value(label_widen, 0, 1), tf.int32)

        
        label_resized = tf.image.resize(label_widen, self.cfg["out"]["edge"]["shape"], method='nearest', antialias=True)
        label_resized = tf.image.resize(label_resized, mask_size, method='bilinear', antialias=True)
        
        if not already_reshaped:
            label_resized = tf.math.argmax(label_resized, axis=-1, output_type=tf.int32)
            label_resized = tf.expand_dims(label_resized, axis=-1)
        label = tf.squeeze(label_resized, axis=0)
        return label
    
    def combine_ds(self, dataset_dict, flow_field):
        flow_field = tf.image.resize(flow_field, self.cfg["out"]["flow"]["shape"], method='bilinear')
        ds_name = self.cfg["out"]["flow"]["name"]
        dataset_dict[ds_name] = flow_field
        if self.cfg["out"]["flow"]["only_edge"]:
            edge_label = tf.where(tf.reduce_sum(dataset_dict['in_edge'], axis=-1, keepdims=True) > 0, 1, 0)
            dataset_dict[ds_name] = dataset_dict[ds_name] * tf.cast(edge_label)
        return dataset_dict
    
    def load_flow_ds(self, ds_type, max_idx):
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
        candidates = ['edge', 'contour', 'segmentation']
        for c in candidates:
            if self.cfg['in'][c]:
                self.inputs.append(c)
            if self.cfg['out'][c]:
                self.outputs_ann.append(c)
    
    def set_input_output_data_cfg(self):
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


def split_dataset_dictionary(datapoint):
    datapoint_input = dict()
    datapoint_output = dict()
    for key in datapoint.keys():
        if 'in' in key:
            datapoint_input[key] = datapoint[key]
        else:
            datapoint_output[key] = datapoint[key]
    return datapoint_input, datapoint_output


def normalize_input_image(datapoint):
    datapoint['in_img'] = tf.cast(datapoint['in_img'], tf.float32) / 127.5 - 1.0
    return datapoint


def reshape_mask(mask, num_classes):
    mask = tf.cast(mask, tf.int32)
    class_range = tf.range(1, num_classes + 1, dtype=tf.int32)
    class_range_reshape = tf.reshape(class_range, [1, 1, num_classes])
    mask = tf.cast(class_range_reshape == mask, dtype=tf.uint8)
    
    return mask
