import COCO
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import skimage.io as io

paths = {}
paths["COCO"] = "/home/david/BlenderProc/BlenderProcBlock/output/coco_data"
paths["COCO_FILE"] = os.path.join(paths["COCO"],"coco_annotations.json")


coco = COCO.COCO(paths["COCO"])

img_paths = coco.get_image_paths()
mask = coco.get_annotation_mask(0)
print(img_paths)
