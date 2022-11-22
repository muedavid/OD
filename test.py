#!for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import cv2

import data_processing.dataset as dataset
import data_processing.model_files as model_files
import network_elements.backbones as backbones
import network_elements.features as features
import network_elements.losses as losses
import network_elements.metrics as metrics
import utils.visualize as visualize
import utils.tools as tools
import utils.learning_utils as learning_utils
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#np.set_printoptions(threshold=sys.maxsize)

# Load model configs
config_path = os.path.join(os.getcwd(), 'configs','edge')
model_config_path = os.path.join(config_path, 'model.yaml')
model_cfg = tools.config_loader(model_config_path)

# Initialize Dataset loader & parse arguments if given from command line
DP = dataset.DataProcessing(model_cfg["INPUT_SHAPE"], model_cfg["OUTPUT_SHAPE"], config_path)
LU = learning_utils.LearningUtil(DP.cfg['TRAIN']['BATCH_SIZE'], config_path)
tools.parser(model_cfg, DP.cfg)

# Get Path definitions where data is stored and the model should be stored
DP.path_definitions()
MF = model_files.ModelFiles()
MF.path_definitions(model_cfg["NAME"], DP.cfg["NAME"], make_dirs=True)
MF.clean_model_directories(LU.cfg["CALLBACKS"]["DEL_OLD_CKPT"], LU.cfg["CALLBACKS"]["DEL_OLD_TB"])

# SEED:
tf.random.set_seed(model_cfg['SEED'])

if LU.cfg['TRAIN_MODEL']:
    rng = tf.random.Generator.from_seed(123, alg='philox')

    train_ds, img_count_train = DP.load_dataset(DP.key.train, shuffle=True, prefetch=True, rng=rng)

test_ds, img_count_test = DP.load_dataset(DP.key.test, shuffle=False, prefetch=True)


# for inp, out in test_ds.take(1):
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     # plt.imshow(tf.keras.preprocessing.image.array_to_img(inp["IN_IMG"][0, :, :, :]))
#     plt.imshow(inp["in_edge"][0, :, :, 0], cmap='gray', vmin=0, vmax=4)
#     plt.subplot(2, 2, 2)
#     plt.imshow(out["out_edge"][0, :, :, 0], cmap='gray', vmin=0, vmax=4)
#
#     inp = inp["in_edge"][0, :, :, 0].numpy()
#     inp = cv2.cvtColor(inp*50, cv2.COLOR_GRAY2RGB)
#     output = out["out_edge"][0, :, :, 0].numpy()
#     output = cv2.cvtColor(output * 50, cv2.COLOR_GRAY2RGB)
#     y_flow = (out["out_flow"][0, :, :, 1] * out['out_flow'].shape[2]).numpy()
#     x_flow = (out["out_flow"][0, :, :, 0] * out['out_flow'].shape[1]).numpy()
#     for row in range(0, x_flow.shape[0], 3):
#         for col in range(0, y_flow.shape[1], 3):
#             if x_flow[row, col] != 0:
#                 start_point = np.array([col, row])
#                 end_point = start_point + np.array([x_flow[row, col], y_flow[row, col]], np.int32)
#                 color = (0, 255, 0)
#                 thickness = 1
#                 inp = cv2.arrowedLine(inp, start_point, end_point, color=color, thickness=thickness)
#                 output = cv2.arrowedLine(output, start_point, end_point, color=color, thickness=thickness)
#
#     # vert = out["OUT_VERT"]
#     # for i in range(9, 20):
#     #     vert = tf.where(vert == i, 255, vert)
#     # plt.imshow(vert[0, :, :, 0], cmap='gray', vmin=0, vmax=100)
#     plt.subplot(2, 2, 3)
#     plt.imshow(inp)
#     plt.subplot(2, 2, 4)
#     plt.imshow(output)
#
#     plt.show()

# for inp, out in test_ds.take(1):
#     img_addon = np.zeros(model_cfg["OUTPUT_SHAPE"])
#     for i in range(inp['input_PRIOR_VERT'][0].shape[0]):
#         row = tf.cast(inp['input_PRIOR_VERT'][0, i, 1]*model_cfg["OUTPUT_SHAPE"][0], tf.int32)
#         col = tf.cast(inp['input_PRIOR_VERT'][0, i, 0]*model_cfg["OUTPUT_SHAPE"][1], tf.int32)
#         img_addon[row, col] = 10
#     for i in range(out['output_CURRENT_VERT'][0].shape[0]):
#         row = tf.cast(out['output_CURRENT_VERT'][0, i, 1]*model_cfg["OUTPUT_SHAPE"][0], tf.int32)
#         col = tf.cast(out['output_CURRENT_VERT'][0, i, 0]*model_cfg["OUTPUT_SHAPE"][1], tf.int32)
#         img_addon[row, col] = 10
#     img = inp['input_PRIOR_ANN'][0, :, :, 0] + out['output_ANN'][0, :, :, 0] + tf.convert_to_tensor(img_addon, tf.uint8)
#     plt.imshow(img[:, :], cmap='gray', vmin=0, vmax=10)
#     plt.show()


# if TRAIN_MODEL:
#     output_dims = tuple(model_cfg["OUTPUT_SHAPE"])
#     print(output_dims)
#
#     # BACKBONE
#     backbone, output_names = backbones.get_backbone(name=model_cfg["BACKBONE"]["NAME"],
#                                                     weights=model_cfg["BACKBONE"]["WEIGHTS"],
#                                                     height=model_cfg["INPUT_SHAPE"][0],
#                                                     width=model_cfg["INPUT_SHAPE"][1],
#                                                     alpha=model_cfg["BACKBONE"]["ALPHA"],
#                                                     output_layer=model_cfg["BACKBONE"]["OUTPUT_IDS"],
#                                                     trainable_idx=model_cfg["BACKBONE"]["TRAIN_IDX"])
#
#     # DASPP
#     daspp = features.DASPP_dilation(backbone.output[-1])
#
#     # Decoder
#     decoded = features.decoder(daspp, backbone.output[-1], output_dims=output_dims, NUM_CLASSES=DP.num_classes,
#                                num_side_filters=6)
#
#     # SIDE FEATURES
#     # TODO: Upsampling: Nearest NEIGHBOUR ?
#     upsample_side_1 = features.side_feature_SGED(backbone.output[0], output_dims=output_dims,
#                                                  num_classes=DP.num_classes, method="bilinear", name="side1")
#     upsample_side_2 = features.side_feature_SGED(backbone.output[1], output_dims=output_dims,
#                                                  num_classes=DP.num_classes, method="bilinear", name="side2")
#     # upsample_side_3 = features.side_feature_SGED(backbone.output[2], output_dims=output_dims ,interpolation="bilinear", name="side3")
#
#     # TODO: adaptive weight fusion ?
#     # CONCATENATE
#     side_outputs = [upsample_side_1, upsample_side_2, decoded]
#     # concat = features.shared_concatenation(side_outputs,NUM_CLASSES)
#     # output = features.fused_classification(concat,NUM_CLASSES,name="output")
#     output = features.shared_concatenation_fused_classification(side_outputs, DP.num_classes, name="output")
#     model = tf.keras.Model(inputs=backbone.input, outputs=output)
#
#     # model.layers[-1]._name = "output"
#     model.summary()

# if TRAIN_MODEL:
#     # learning rate schedule
#     base_learning_rate = 0.0015
#     end_learning_rate = 0.0005
#     decay_step = np.ceil(img_count_train / DP.cfg[DP.key.train]["BATCH_SIZE"]) * model_cfg["EPOCHS"]
#     lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(base_learning_rate, decay_steps=decay_step,
#                                                                 end_learning_rate=end_learning_rate, power=0.9)
#
#     frequency = int(
#         np.ceil(img_count_train / DP.cfg[DP.key.train]["BATCH_SIZE"]) * model_cfg["CALLBACKS"]["CKPT_FREQ"]) + 1
#
#     logdir = os.path.join(MF.paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
#     callbacks = [tf.keras.callbacks.ModelCheckpoint(
#         filepath=MF.paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
#         save_weights_only=False, save_best_only=False, monitor="val_f1", verbose=1, save_freq='epoch', period=5),
#         tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)]
#
#     # compile model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss,
#                   metrics=[metrics.BinaryAccuracyEdges(threshold_prediction=0),
#                            metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)])
#
#     history = model.fit(train_ds, epochs=model_cfg["EPOCHS"], validation_data=test_ds, callbacks=callbacks)
#
# model_ckpt = os.listdir(MF.paths['CKPT'])
#
# f1_max = 0
# for ckpt_name in model_ckpt:
#     if float(ckpt_name[-4:]) > f1_max:
#         f1_max = float(ckpt_name[-4:])
#         model_path = MF.paths['CKPT'] + "/" + ckpt_name
#
#         print(model_path)
#
# custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges, "F1Edges": metrics.F1Edges, "<lambda>": loss}
#
# model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#
# if TRAIN_MODEL:
#     plot_losses = ["loss", "loss"]
#     plot_metrics = ["accuracy_edges", "f1", "recall", "precision"]
#
#     path = os.path.join(MF.paths["FIGURES"], "training.svg")
#
#     visualize.plot_training_results(res=history.history, losses=plot_losses, metrics=plot_metrics,
#                                     save=model_cfg["SAVE"], path=path)
#
# ### Maximum F1 Score:
# # TODO: FAILS right now: fix
# # if not TRAIN_MODEL:
# #     step_width = 0.05
# #     threshold_range = [0.05, 0.95]
# #     threshold_array = np.arange(threshold_range[0], threshold_range[1] + step_width, step_width)
# #     threshold_array = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.975])
# #
# #     path_metrics_evaluation_plot = os.path.join(MF.paths["FIGURES"], "threshold_metrics_evaluation_test_ds.svg")
# #     threshold_f1_max = visualize.plot_threshold_metrics_evaluation_class(model=model, ds=test_ds,
# #                                                                          num_classes=DP.num_classes,
# #                                                                          threshold_array=threshold_array,
# #                                                                          threshold_edge_width=0, save=model_cfg["SAVE"],
# #                                                                          path=path_metrics_evaluation_plot)
#
# if not TRAIN_MODEL:
#     i = 0
#     for img, label in test_ds.take(1):
#         img, label = img, label
#
#         threshold = 0.5
#
#         predictions = model.predict(img)
#         predictions = tools.predict_class_postprocessing(predictions, threshold=threshold)
#
#         path = os.path.join(MF.paths["FIGURES"], "img_test_threshold_{}_{}".format(threshold, i))
#         visualize.plot_images(images=img, labels=label, predictions=predictions, save=model_cfg["SAVE"], path=path,
#                               batch_size=3)
#
#         # threshold = threshold_f1_max
#         # path = os.path.join(MF.paths["FIGURES"], "img_test_ods_{}".format(i))
#         # visualize.plot_images(images=img, labels=label, predictions=predictions, save=model_cfg["SAVE"], path=path,
#         #                       batch_size=3)
#
#         i += 1
#
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics={
#     'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
#                metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})
#
# if model_cfg["SAVE"]:
#     model.save(MF.paths["MODEL"])
#
#     custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges, "F1Edges": metrics.F1Edges, "<lambda>": loss}
#
#     model = tf.keras.models.load_model(MF.paths["MODEL"], custom_objects=custom_objects)
