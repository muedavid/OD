import json
import time
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pycocotools.coco import _isArrayLike, maskUtils


class COCO:
    def __init__(self, coco_path=None):
        # load cfg
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.coco_path = coco_path
        if not coco_path == None:
            annotation_file = os.path.join(coco_path, "coco_annotations.json")
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def get_image_paths(self, max_idx=None):
        img_paths = []
        idx = 1
        for img in self.imgs.values():
            img_paths.append(os.path.join(self.coco_path, img['file_name']))
            if idx == max_idx:
                break
        return img_paths

    def get_annotation_mask(self, idx):
        mask = np.zeros((self.anns[1]["height"], self.anns[1]["width"], 1))
        for annotation in self.imgToAnns[idx]:

            print(annotation)

            # mask
            if type(annotation['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects([annotation['segmentation']], annotation['height'], annotation['width'])

                m = maskUtils.decode(rle)*int(annotation['category_id'])

                mask += m

        return mask
