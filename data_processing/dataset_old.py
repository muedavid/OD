import collections

import tensorflow as tf
import tensorflow_addons as tfa
import os
import os.path as osp
import numpy as np
import json
import yaml

import data_processing.ds_augmentation as spot_light


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
        yaml_path = osp.join(config_path, 'dataset.yaml')
        with open(yaml_path, 'r') as file:
            self.ds_cfg = yaml.safe_load(file)
        file.close()
        self.input_output_keys()

    def input_output_keys(self):
        candidates = ['EDGE', 'VERT', 'CONT']
        for c in candidates:
            if self.ds_cfg['IN'][c]:
                self.inputs.append(c)
        candidates = ['EDGE', 'VERT', 'CONT']
        for c in candidates:
            if self.ds_cfg['OUT'][c]:
                self.outputs_ann.append(c)

    def path_definitions(self):
        paths = dict()
        for key in ["TRAIN", "TEST", "IMG_ONLY"]:
            if self.ds_cfg[key] is not None:
                paths[key] = osp.join(self.ds_cfg["BASE_PATH_DATA"], self.ds_cfg["NAME"], self.ds_cfg[key]["NAME"])
        self.paths['DATA'] = paths

        # Open JSON File with relative paths information
        for key_ds, val_ds in self.paths['DATA'].items():
            dataset_information_path = osp.join(val_ds, self.ds_cfg["DATASET_JSON"])
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

            if self.ds_cfg['VERT_LIST']:
                with open(self.ds_inf['paths']['VERT'], 'r') as file:
                    self.vert_list[key_ds] = yaml.safe_load(file)
                file.close()

    def load_dataset(self, ds_type):
        max_idx = min(self.ds_inf[ds_type]["info"]["num_frames"] - 1, self.ds_cfg[ds_type]["MAX_IMG"] - 1)

        edge_stacked = []
        for i in range(max_idx):
            edge = np.load(self.paths[ds_type]['EDGE_FLOW']+'/{:04}.npy'.format(i))
            edge = edge.astype(np.float32)
            edge_stacked.append(edge)
        edges = np.stack(edge_stacked, axis=0)
        edge_dataset = tf.data.Dataset.from_tensor_slices(edges)

        # if self.cfg['VERT_LIST']:
        #     self.vertices[ds_type] = self.construct_vertices_tensor(ds_type)
        #     self.vert_maps[ds_type] = self.vertices_map(ds_type)
        #     print(ds_type, len(self.vertices))

        dataset = tf.data.Dataset.from_tensor_slices(range(max_idx))
        dataset = dataset.map(lambda x: self.parse_image(x, ds_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_combined = tf.data.Dataset.zip((dataset, edge_dataset))
        dataset = dataset_combined.map(self.combine_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)


        image_count = max_idx + 1
        print("The {mode} DS contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=ds_type))

        return dataset, image_count

    def combine_ds(self, dataset_dict, flow_field):
        flow_field = tf.image.resize(flow_field, self.output_shape, method='nearest')
        dataset_dict['OUT_FLOW'] = flow_field
        return dataset_dict

    def preprocess_mask(self, mask, ds_type, mask_type):
        # instance or category labels
        # Python side effect ok as value does not change

        # category
        if self.ds_cfg['MASK_TYPE'][mask_type] == 1:
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
        dataset_dict['IN_IMG'] = image

        # mask input:
        mask_base_path = tf.constant(self.paths[ds_type]['PRIOR_ANN'], dtype=tf.string)
        mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
        mask_input = tf.io.read_file(mask_path)
        mask_input = tf.image.decode_png(mask_input, channels=3)
        for inp in self.inputs:
            if self.ds_cfg['IN'][inp]:
                idx = self.ds_inf[ds_type]['info']['mask'][inp]
                mask = mask_input[:, :, idx:idx+1]
                dataset_dict['IN_' + inp] = self.preprocess_mask(mask, ds_type, inp)

        # mask output:
        mask_base_path = tf.constant(self.paths[ds_type]['ANN'], dtype=tf.string)
        mask_path = tf.strings.join([mask_base_path, sep_str, img_idx_str, end_str])
        mask_output = tf.io.read_file(mask_path)
        mask_output = tf.image.decode_png(mask_output, channels=3)
        # unrolled for loop, execute for each statement individually and also if statements (python side effect)
        for out in self.outputs_ann:
            idx = self.ds_inf[ds_type]['info']['mask'][out]
            mask = mask_output[:, :, idx:idx+1]
            dataset_dict['OUT_' + out] = self.preprocess_mask(mask, ds_type, out)

            # else:
            #     flow_base_path = tf.constant(self.paths[ds_type]['EDGE_FLOW'], dtype=tf.string)
            #     end_str = tf.constant(".npy", dtype=tf.string)
            #     flow_path = tf.strings.join([flow_base_path, sep_str, img_idx_str, end_str])
            #     flow_field = tf.py_function(self.load_npy_as_tensor, inp=[flow_path], Tout=tf.float32)
            #     flow_field = tf.image.resize(flow_field, self.output_shape, method='bilinear')
            #     dataset_dict['OUT_FLOW'] = flow_field

        # if self.cfg['VERT_LIST']:
        #     dataset_dict["VERT_MAP"] = tf.py_function(self.add_vertices_map, inp=[img_idx, ds_type], Tout=tf.float32)

        return dataset_dict

    # def add_vertices_map(self, img_idx, ds_type):
    #     return self.vert_maps[ds_type.numpy().decode('UTF-8')][img_idx.numpy()]

    def load_npy_as_tensor(self, path):
        flow = np.load(path.numpy().decode('UTF-8'))
        flow = flow.astype(np.float32)
        return tf.convert_to_tensor(flow)

    # def vertices_map(self, ds_type):
    #     vert_maps = []
    #
    #     for img_idx in range(len(self.vert_list[ds_type]['PRIOR'])):
    #
    #         print(img_idx)
    #
    #         shape = (80, 45)
    #         map = np.zeros((shape[0], shape[1], 2))
    #
    #         vertices_prior = []
    #         label_list = []
    #         for vert in self.vert_list['ds_type']['PRIOR'][img_idx]:
    #             if vert["visible"]:
    #                 vertices_prior.append(
    #                     [vert["co"][1] / self.input_shape[0] * shape[0], vert["co"][0] / self.input_shape[1] * shape[1]])
    #                 label_list.append(vert["label"])
    #
    #         for vert in self.vert_list['ds_type']['CURRENT'][img_idx]:
    #             if vert["visible"]:
    #                 labels = vert["label"] == np.array(label_list)
    #                 prior = np.array(vertices_prior)
    #                 coordinates = prior[labels, :]
    #                 if coordinates.shape[0] != 0:
    #                     diff = np.array([vert["co"][1] / self.input_shape[0] * shape[0],
    #                                      vert["co"][0] / self.input_shape[1] * shape[1]]) - coordinates
    #                     coo = coordinates.astype(np.int32)
    #                     map[coo[0, 0], coo[0, 1], 0] = diff[0, 0]
    #                     map[coo[0, 0], coo[0, 1], 1] = diff[0, 1]
    #             vert_maps.append(map)
    #     return vert_maps

    # def construct_vertices_tensor(self, ds_type):
    #     vertices_list = []
    #
    #     # prior and current have the same length
    #     for img_idx in range(len(self.vert_list['ds_type']['PRIOR'])):
    #         vertices_dict = dict()
    #
    #         # PRIOR
    #         vertices_prior = np.full((20, 3), 0.0)
    #         vertices_current = np.full((20, 3), 0.0)
    #         label_list = np.full(20, None)
    #         vert_idx = 0
    #         for vert in self.vert_list['ds_type']['PRIOR'][img_idx]:
    #             if vert_idx < 20:
    #                 if vert["visible"]:
    #                     vertices_prior[vert_idx, 0] = vert["co"][0] / self.input_shape[1]
    #                     vertices_prior[vert_idx, 1] = vert["co"][1] / self.input_shape[0]
    #                     vertices_prior[vert_idx, 2] = 1
    #                     label_list[vert_idx] = vert["label"]
    #                     vert_idx += 1
    #
    #         for vert in self.vert_list['ds_type']['CURRENT'][img_idx]:
    #             if "visible" in vert.keys():
    #                 if vert["visible"]:
    #                     labels = vert["label"] == label_list
    #                     vertices_current[labels, 0] = vert["co"][0] / self.input_shape[1]
    #                     vertices_current[labels, 1] = vert["co"][1] / self.input_shape[0]
    #                     vertices_current[labels, 2] = 1
    #         vertices_dict['PRIOR_VERT'] = tf.convert_to_tensor(vertices_prior, dtype=tf.float32)
    #         vertices_dict['CURRENT_VERT'] = tf.convert_to_tensor(vertices_current, dtype=tf.float32)
    #         vertices_list.append(vertices_dict)
    #
    #     return vertices_list

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

    # def preprocess(self, datapoint, ds_type):
    #     # Preprocessing Layer already added into model Pipeline: model expects input of 0-255, 3 channels
    #     # datapoint['IMG'] = tf.cast(datapoint['IMG'], tf.float32) / 127.5 - 1
    #     # datapoint['IMG'] = tf.cast(datapoint['IMG'], tf.float32)
    #
    #     # datapoint['IMG'] = tf.image.resize(datapoint['IMG'], shape_input, method='nearest')
    #     # datapoint['IMG'] = tf.cast(datapoint['IMG'], tf.uint8)
    #
    #     ds_dict = {}
    #
    #     for key, val in datapoint.items():
    #         if 'IMG' in key:
    #
    #         elif 'FLOW' not in key:
    #             shape = tf.shape(val)
    #             current_shape = (shape[0], shape[1])
    #
    #             # INSTANCE:
    #             # Python side effect ok as value does not change
    #             key_annotation = None
    #             if 'EDGE' in key:
    #                 key_annotation = 'EDGE'
    #             if 'CONT' in key:
    #                 key_annotation = 'CONT'
    #             if 'VERT' in key:
    #                 key_annotation = 'VERT'
    #             if self.cfg["MASK_TYPE"][key_annotation] == 0:
    #                 self.num_classes = len(self.ds_inf[ds_type]["obj2cat"])
    #
    #             # CAT
    #             # Python side effect ok as value does not change
    #             elif self.cfg["MASK_TYPE"] == 1:
    #                 self.num_classes = len(self.ds_inf[ds_type]["cat2obj"])
    #                 val = tf.cast(val, tf.int32)
    #                 for inst, cat in self.ds_inf[ds_type]["obj2cat"].items():
    #                     val = tf.where(val == int(inst), cat, val)
    #                 # val = tf.where(val == 1, 1, 0)
    #                 val = tf.cast(val, tf.uint8)
    #
    #             val = self.resize_label_map(val, current_shape)
    #             val = tf.cast(val, tf.uint8)
    #
    #         ds_dict[key] = val
    #
    #     return ds_dict

    def dataset_processing(self, ds, ds_type, shuffle=False, prefetch=False, rng=None, normalize=False, img_count=0,
                           prior=False):
        if self.ds_cfg[ds_type]["CACHE"]:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(img_count, reshuffle_each_iteration=True)
        if self.ds_cfg[ds_type]["DATA_AUG"]:
            ds = ds.map(lambda x: augment_mapping(x, rng, self.ds_cfg[ds_type]["DATA_AUG"]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if normalize:
            ds = ds.map(normalize_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: split_dataset_dictionary(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.ds_cfg[ds_type]["BATCH_SIZE"]:
            ds = ds.batch(self.ds_cfg[ds_type]["BATCH_SIZE"])
        if prefetch:
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


def split_dataset_dictionary(datapoint):
    datapoint_input = dict()
    datapoint_output = dict()
    for key in datapoint.keys():
        if 'IN' in key:
            datapoint_input[key] = datapoint[key]
        else:
            datapoint_output[key] = datapoint[key]
    return datapoint_input, datapoint_output


def value_augmentation_spot_light(shape, value, strength_spot):
    shape = shape.numpy()

    uniform_value_diff = np.random.uniform(-value, value)
    mask = np.zeros(shape)
    mask[:, :, 2] = uniform_value_diff
    strength = np.random.uniform(0.0, strength_spot)
    mask_raw = spot_light.generate_spot_light_mask(mask_size=(shape[1], shape[0]))
    mask[:, :, 2] = mask[:, :, 2] + strength * mask_raw
    mask[:, :, 1] = -strength * mask_raw
    return mask


def augment_mapping(datapoint, rng, aug_param):
    if aug_param["blur"]:
        sigma = np.random.uniform(0, aug_param["sigma"])
        datapoint['IN_IMG'] = tfa.image.gaussian_filter2d(datapoint['IN_IMG'], (5, 5), sigma)

    if aug_param["noise_std"] != 0:
        datapoint['IN_IMG'] = tf.cast(datapoint['IN_IMG'], tf.float32)
        datapoint['IN_IMG'] = tf.keras.layers.GaussianNoise(aug_param["noise_std"])(datapoint['IN_IMG'], training=True)
        datapoint['IN_IMG'] = tf.clip_by_value(datapoint['IN_IMG'], 0, 255.0)
        datapoint['IN_IMG'] = tf.cast(datapoint['IN_IMG'], tf.uint8)

    seed = rng.make_seeds(2)[0]
    datapoint['IN_IMG'] = tf.image.stateless_random_contrast(datapoint['IN_IMG'], aug_param["contrast_factor"],
                                                          1 / aug_param["contrast_factor"], seed)
    seed = rng.make_seeds(2)[0]
    datapoint['IN_IMG'] = tf.image.stateless_random_brightness(datapoint['IN_IMG'], aug_param["brightness"], seed)

    seed = rng.make_seeds(2)[0]
    datapoint['IN_IMG'] = tf.image.stateless_random_hue(datapoint['IN_IMG'], aug_param["hue"], seed)
    seed = rng.make_seeds(2)[0]
    datapoint['IN_IMG'] = tf.image.stateless_random_saturation(datapoint['IN_IMG'], aug_param["saturation"],
                                                            1 / aug_param["saturation"], seed)
    seed = rng.make_seeds(2)[0]

    # convert to HSV
    datapoint['IN_IMG'] = tf.image.convert_image_dtype(datapoint['IN_IMG'], tf.float32)
    datapoint['IN_IMG'] = tf.image.rgb_to_hsv(datapoint['IN_IMG'])

    mask = tf.py_function(value_augmentation_spot_light,
                          inp=[datapoint['IN_IMG'].shape, aug_param["value"], aug_param["strength_spot"]], Tout=tf.float32)
    gaussian_noise = tf.random.stateless_uniform([1], seed, minval=0, maxval=aug_param["gaussian_value"])
    mask = tf.keras.layers.GaussianNoise(gaussian_noise)(mask, training=True)
    datapoint['IN_IMG'] = mask + datapoint['IN_IMG']
    datapoint['IN_IMG'] = tf.clip_by_value(datapoint['IN_IMG'], 0.0, 1.0)

    # convert back to RGB of uint8: [0,255]
    datapoint['IN_IMG'] = tf.image.hsv_to_rgb(datapoint['IN_IMG'])
    datapoint['IN_IMG'] = tf.image.convert_image_dtype(datapoint['IN_IMG'], tf.uint8, saturate=True)

    # seed = rng.make_seeds(2)[0]
    # for key in datapoint.keys():
    #     if 'VERT' not in key:
    #         datapoint[key] = tf.image.stateless_random_flip_left_right(datapoint[key], seed)
    #         if seed[0] > 5:
    #             print("x")

    # seed = rng.make_seeds(2)[0]
    # for key in datapoint.keys():
    #     if 'VERT' not in key:
    #         datapoint[key] = tf.image.stateless_random_flip_up_down(datapoint[key], seed)

    return datapoint


def normalize_fun(datapoint):
    datapoint['IN_IMG'] = tf.cast(datapoint['IN_IMG'], tf.float32) / 127.5 - 1.0
    return datapoint
