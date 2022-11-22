import collections

import tensorflow as tf
import os
import os.path as osp
import numpy as np
import json
import yaml

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
    vertices = collections.defaultdict(list)
    vert_maps = collections.defaultdict(list)
    vert_list = dict()
    inputs = list()
    outputs_ann = list()
    num_classes = dict()

    def __init__(self, input_shape, output_shape, config_path):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cfg = tools.config_loader(osp.join(config_path, 'dataset.yaml'))
        # yaml_path = osp.join(config_path, 'dataset.yaml')
        # with open(yaml_path, 'r') as file:
        #     self.cfg = yaml.safe_load(file)
        # file.close()
        self.input_output_keys()

    def path_definitions(self):
        paths = dict()
        for key in ["TRAIN", "TEST", "IMG_ONLY"]:
            if self.cfg[key] is not None:
                paths[key] = osp.join(self.cfg["BASE_PATH_DATA"], self.cfg["NAME"], self.cfg[key]["NAME"])
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

            if self.cfg['VERT_LIST']:
                with open(self.ds_inf['paths']['VERT'], 'r') as file:
                    self.vert_list[key_ds] = yaml.safe_load(file)
                file.close()

    def load_dataset(self, ds_type, shuffle=False, prefetch=False, rng=None, normalize=True):
        max_idx = min(self.ds_inf[ds_type]["info"]["num_frames"] - 1, self.cfg[ds_type]["MAX_IMG"] - 1)

        dataset = tf.data.Dataset.from_tensor_slices(range(max_idx))
        dataset = dataset.map(lambda x: self.parse_image(x, ds_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.cfg['out']['flow_edge']:
            edge_dataset = self.load_flow_ds(ds_type, max_idx, self.ds_inf[ds_type]["info"]["flow"]["edge"])
            dataset_combined = tf.data.Dataset.zip((dataset, edge_dataset))
            dataset = dataset_combined.map(self.combine_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif self.cfg['out']['flow_scene']:
            edge_dataset = self.load_flow_ds(ds_type, max_idx, self.ds_inf[ds_type]["info"]["flow"]["scene"])
            dataset_combined = tf.data.Dataset.zip((dataset, edge_dataset))
            dataset = dataset_combined.map(self.combine_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        image_count = max_idx + 1

        print("The {mode} DS contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=ds_type))

        if self.cfg[ds_type]["CACHE"]:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(image_count, reshuffle_each_iteration=True)
        if self.cfg[ds_type]["DATA_AUG"]:
            dataset = dataset.map(lambda x: ds_aug.augment_mapping(x, rng, self.cfg[ds_type]["DATA_AUG"]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if normalize:
            dataset = dataset.map(normalize_input_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: split_dataset_dictionary(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cfg[ds_type]["BATCH_SIZE"]:
            dataset = dataset.batch(self.cfg[ds_type]["BATCH_SIZE"])
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset, image_count

    def parse_image(self, img_idx, ds_type):
        img_idx_str = tf.strings.as_string(img_idx, width=4, fill='0')
        end_str = tf.constant(".png", dtype=tf.string)
        sep_str = tf.constant('/', dtype=tf.string)
        dataset_dict = dict()

        # LOAD IMG
        img_base_path = tf.constant(self.paths[ds_type]["IMG"], dtype=tf.string)
        img_path = tf.strings.join([img_base_path, sep_str, img_idx_str, end_str])
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.input_shape, method='bilinear')
        image = tf.cast(image, tf.uint8)
        dataset_dict['in_img'] = image

        # mask input:
        mask_base_path = tf.constant(self.paths[ds_type]['PRIOR_ANN'], dtype=tf.string)
        mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
        mask_input = tf.io.read_file(mask_path)
        mask_input = tf.image.decode_png(mask_input, channels=3)
        for inp in self.inputs:
            if self.cfg['in'][inp]:
                idx = self.ds_inf[ds_type]['info']['mask'][inp]
                mask = mask_input[:, :, idx:idx + 1]
                dataset_dict['in_' + inp] = self.preprocess_mask(mask, ds_type, inp)

        # mask output:
        mask_base_path = tf.constant(self.paths[ds_type]['ANN'], dtype=tf.string)
        mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
        mask_output = tf.io.read_file(mask_path)
        mask_output = tf.image.decode_png(mask_output, channels=3)
        # unrolled for loop, execute for each statement individually and also if statements (python side effect)
        for out in self.outputs_ann:
            idx = self.ds_inf[ds_type]['info']['mask'][out]
            mask = mask_output[:, :, idx:idx + 1]
            dataset_dict['out_' + out] = self.preprocess_mask(mask, ds_type, out)

        return dataset_dict

    def preprocess_mask(self, mask, ds_type, mask_type):
        # Python side effect ok as value does not change

        # category
        if self.cfg['MASK_TYPE'][mask_type] == 1:
            mask = tf.cast(mask, tf.int32)
            if mask_type == 'VERT':
                for inst, cat in self.ds_inf[ds_type]["vert2obj"].items():
                    mask = tf.where(mask == int(inst), cat, mask)
            for inst, cat in self.ds_inf[ds_type]["obj2cat"].items():
                mask = tf.where(mask == int(inst), cat, mask)
            mask = tf.cast(mask, tf.uint8)

            self.num_classes[mask_type] = len(self.ds_inf[ds_type]["cat2obj"])
        else:
            if mask_type == 'VERT':
                self.num_classes[mask_type] = len(self.ds_inf[ds_type]["vert2obj"])
            else:
                self.num_classes[mask_type] = len(self.ds_inf[ds_type]["obj2cat"])

        # reshape:
        shape = tf.shape(mask)
        current_shape = (shape[0], shape[1])

        mask = self.resize_label_map(mask, current_shape, self.num_classes[mask_type])
        return tf.cast(mask, tf.uint8)

    def resize_label_map(self, label, current_shape_label, num_classes):
        # label 3D
        label = tf.cast(label, tf.int32)
        label = tf.expand_dims(label, axis=0)
        class_range = tf.range(1, num_classes + 1)
        class_range_reshape = tf.reshape(class_range, [1, 1, 1, num_classes])
        label_re = tf.cast(class_range_reshape == label, dtype=tf.int32)
        pad = tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]])
        label_re = tf.pad(label_re, pad, "CONSTANT")

        edge_width_height = int(current_shape_label[0] / self.output_shape[0]) + 1
        edge_width_width = int(current_shape_label[1] / self.output_shape[1]) + 1
        kernel = tf.ones([edge_width_height, edge_width_width, num_classes + 1, 1], tf.float32)
        label_re = tf.cast(label_re, tf.float32)
        label_re = tf.nn.depthwise_conv2d(label_re, kernel, strides=[1, 1, 1, 1], padding="SAME")
        label_re = tf.cast(tf.clip_by_value(label_re, 0, 1), tf.int32)

        label_re = tf.image.resize(label_re, self.output_shape, method='nearest', antialias=True)
        label_re = tf.math.argmax(label_re, axis=-1, output_type=tf.int32)
        label = tf.expand_dims(label_re, axis=-1)
        label = tf.squeeze(label, axis=0)
        return label

    def combine_ds(self, dataset_dict, flow_field):
        flow_field = tf.image.resize(flow_field, self.output_shape, method='nearest')
        dataset_dict['out_flow'] = flow_field
        return dataset_dict

    def load_flow_ds(self, ds_type, max_idx, idx):
        edge_stacked = []
        for i in range(max_idx):
            edge = np.load(self.paths[ds_type]['EDGE_FLOW'] + '/{:04}.npy'.format(i))
            edge = edge[idx, :, :, :].astype(np.float32)
            edge_stacked.append(edge)
        edges = np.stack(edge_stacked, axis=0)
        edge_dataset = tf.data.Dataset.from_tensor_slices(edges)
        return edge_dataset

    def input_output_keys(self):
        candidates = ['edge', 'vert', 'cont']
        for c in candidates:
            if self.cfg['in'][c]:
                self.inputs.append(c)
        candidates = ['edge', 'vert', 'cont']
        for c in candidates:
            if self.cfg['out'][c]:
                self.outputs_ann.append(c)


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
