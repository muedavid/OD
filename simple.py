import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import argparse
import yaml

import data_processing.dataset as dataset
import data_processing.model_files as model_files
import network_elements.backbones as backbones
import network_elements.features as features
import network_elements.losses as losses
import network_elements.metrics as metrics
import network_elements.visualize as visualize
import network_elements.tools as tools

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=False, default=None)
parser.add_argument('--data', type=str, required=False, default=None)

parser.add_argument('--bs', type=int, required=False, default=None)
parser.add_argument('--idx', type=int, required=False, default=None)
parser.add_argument('--epoch', type=int, required=False, default=None)

parser.add_argument('--train_model', action='store_true', default=False)
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--sigmoid', action='store_true', default=False)
parser.add_argument('--focal', action='store_true', default=False)

parser.add_argument('--beta_upper', type=float, required=False, default=None)
parser.add_argument('--gamma', type=float, required=False, default=None)
parser.add_argument('--alpha', type=float, required=False, default=None)

args = parser.parse_args()

# Generall Parameters
TRAIN_MODEL = True  # args.train_model
SEED = None

# LOSS
weighted_multi_label_sigmoid_edge_loss = args.sigmoid
# focal_loss = args.focal
focal_loss = True

beta_upper = 0.5 if args.beta_upper is None else args.beta_upper
beta_lower = 1.0 - beta_upper
gamma = 2.0 if args.gamma is None else args.gamma
alpha = 2.0 if args.alpha is None else args.alpha
class_weighted = True
weighted_beta = True

config_path = os.path.join(os.getcwd(), 'configs')
model_config_path = os.path.join(config_path, 'model.yaml')

with open(model_config_path, 'r') as file:
    model_cfg = yaml.safe_load(file)

tf.random.set_seed(SEED)

DP = dataset.DataProcessing(model_cfg["INPUT_SHAPE_IMG"], model_cfg["OUTPUT_SHAPE"], config_path)
DP.path_definitions()

MF = model_files.ModelFiles()
MF.path_definitions(model_cfg["NAME"], DP.cfg["NAME"], make_dirs=True)
MF.clean_model_directories(model_cfg["CALLBACKS"]["DEL_OLD_CKPT"], model_cfg["CALLBACKS"]["DEL_OLD_TB"])

if TRAIN_MODEL:
    rng = tf.random.Generator.from_seed(123, alg='philox')

    train_ds, img_count_train = DP.load_dataset(DP.key.train)
    train_ds = DP.dataset_processing(train_ds, DP.key.train, shuffle=True, prefetch=True, img_count=img_count_train,
                                     rng=rng)

test_ds, img_count_test = DP.load_dataset(DP.key.test)
test_ds = DP.dataset_processing(test_ds, DP.key.test, shuffle=False, prefetch=True, img_count=img_count_test)

if weighted_multi_label_sigmoid_edge_loss:
    loss = lambda y_true, y_pred: losses.weighted_multi_label_sigmoid_loss(y_true, y_pred, beta_lower=beta_lower,
                                                                           beta_upper=beta_upper,
                                                                           class_weighted=class_weighted)
elif focal_loss:
    loss = lambda y_true, y_pred: losses.focal_loss_edges(y_true, y_pred, gamma=gamma, alpha=alpha,
                                                          weighted_beta=weighted_beta, beta_lower=beta_lower,
                                                          beta_upper=beta_upper, class_weighted=class_weighted)
else:
    raise ValueError("either FocalLoss or WeightedMultiLabelSigmoidLoss must be True")

if TRAIN_MODEL:
    output_dims = model_cfg["OUTPUT_SHAPE"]

    # BACKBONE
    backbone, output_names = backbones.get_backbone(name=model_cfg["BACKBONE"]["NAME"],
                                                    weights=model_cfg["BACKBONE"]["WEIGHTS"],
                                                    height=model_cfg["INPUT_SHAPE_IMG"][0],
                                                    width=model_cfg["INPUT_SHAPE_IMG"][1],
                                                    alpha=model_cfg["BACKBONE"]["ALPHA"],
                                                    output_layer=model_cfg["BACKBONE"]["OUTPUT_IDS"],
                                                    trainable_idx=model_cfg["BACKBONE"]["TRAIN_IDX"])

    # DASPP
    daspp = features.DASPP_dilation(backbone.output[-1])

    # Decoder
    decoded = features.decoder(daspp, backbone.output[-1], output_dims=output_dims, NUM_CLASSES=DP.num_classes,
                               num_side_filters=6)

    # SIDE FEATURES
    # TODO: Upsampling: Nearest NEIGHBOUR ?
    upsample_side_1 = features.side_feature_SGED(backbone.output[0], output_dims=output_dims,
                                                 num_classes=DP.num_classes, method="bilinear", name="side1")
    upsample_side_2 = features.side_feature_SGED(backbone.output[1], output_dims=output_dims,
                                                 num_classes=DP.num_classes, method="bilinear", name="side2")
    # upsample_side_3 = features.side_feature_SGED(backbone.output[2], output_dims=output_dims ,interpolation="bilinear", name="side3")

    # TODO: adaptive weight fusion ?
    # CONCATENATE
    side_outputs = [upsample_side_1, upsample_side_2, decoded]
    # concat = features.shared_concatenation(side_outputs,NUM_CLASSES)
    # output = features.fused_classification(concat,NUM_CLASSES,name="output")
    output = features.shared_concatenation_fused_classification(side_outputs, DP.num_classes, name="output_ANN")
    model = tf.keras.Model(inputs=backbone.input, outputs=output)

    # model.layers[-1]._name = "output"
    model.summary()

if TRAIN_MODEL:
    # learning rate schedule
    base_learning_rate = 0.0015
    end_learning_rate = 0.0005
    decay_step = np.ceil(img_count_train / DP.cfg[DP.key.train]["BATCH_SIZE"]) * model_cfg["EPOCHS"]
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(base_learning_rate, decay_steps=decay_step,
                                                                end_learning_rate=end_learning_rate, power=0.9)

    frequency = int(
        np.ceil(img_count_train / DP.cfg[DP.key.train]["BATCH_SIZE"]) * model_cfg["CALLBACKS"]["CKPT_FREQ"]) + 1

    logdir = os.path.join(MF.paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=MF.paths["CKPT"] + "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
        save_weights_only=False, save_best_only=False, monitor="val_f1", verbose=1, save_freq='epoch', period=5),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)]

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss,
                  metrics=[metrics.BinaryAccuracyEdges(threshold_prediction=0),
                           metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)])

    history = model.fit(train_ds, epochs=model_cfg["EPOCHS"], validation_data=test_ds, callbacks=callbacks)

model_ckpt = os.listdir(MF.paths['CKPT'])

f1_max = 0
for ckpt_name in model_ckpt:
    if float(ckpt_name[-4:]) > f1_max:
        f1_max = float(ckpt_name[-4:])
        model_path = MF.paths['CKPT'] + "/" + ckpt_name

        print(model_path)

custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges, "F1Edges": metrics.F1Edges, "<lambda>": loss}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

if TRAIN_MODEL:
    plot_losses = ["loss", "loss"]
    plot_metrics = ["accuracy_edges", "f1", "recall", "precision"]

    path = os.path.join(MF.paths["FIGURES"], "training.svg")

    visualize.plot_training_results(res=history.history, losses=plot_losses, metrics=plot_metrics,
                                    save=model_cfg["SAVE"], path=path)

### Maximum F1 Score:
# TODO: FAILS right now: fix
# if not TRAIN_MODEL:
#     step_width = 0.05
#     threshold_range = [0.05, 0.95]
#     threshold_array = np.arange(threshold_range[0], threshold_range[1] + step_width, step_width)
#     threshold_array = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.975])
#
#     path_metrics_evaluation_plot = os.path.join(MF.paths["FIGURES"], "threshold_metrics_evaluation_test_ds.svg")
#     threshold_f1_max = visualize.plot_threshold_metrics_evaluation_class(model=model, ds=test_ds,
#                                                                          num_classes=DP.num_classes,
#                                                                          threshold_array=threshold_array,
#                                                                          threshold_edge_width=0, save=model_cfg["SAVE"],
#                                                                          path=path_metrics_evaluation_plot)

if not TRAIN_MODEL:
    i = 0
    for img, label in test_ds.take(1):
        img, label = img, label

        threshold = 0.5

        predictions = model.predict(img)
        predictions = tools.predict_class_postprocessing(predictions, threshold=threshold)

        path = os.path.join(MF.paths["FIGURES"], "img_test_threshold_{}_{}".format(threshold, i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=model_cfg["SAVE"], path=path,
                              batch_size=3)

        # threshold = threshold_f1_max
        # path = os.path.join(MF.paths["FIGURES"], "img_test_ods_{}".format(i))
        # visualize.plot_images(images=img, labels=label, predictions=predictions, save=model_cfg["SAVE"], path=path,
        #                       batch_size=3)

        i += 1

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics={
    'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
               metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})

if model_cfg["SAVE"]:
    model.save(MF.paths["MODEL"])

    custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges, "F1Edges": metrics.F1Edges, "<lambda>": loss}

    model = tf.keras.models.load_model(MF.paths["MODEL"], custom_objects=custom_objects)
