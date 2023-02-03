import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from metrics import metrics
from utils import tools
import cv2


def plot_edges(images=None, prior=None, labels_edge=None, labels_segmentation=None, predictions_edge=None, save=False,
               predictions_segmentation=None, path=None, batch_size=0, num_exp=None):
    if labels_edge is not None:
        num_classes_labels_edge = labels_edge.shape[-1]
        labels_edge = tools.squeeze_labels_to_single_dimension(labels_edge)
    if labels_segmentation is not None:
        num_classes_labels_segmentation = labels_segmentation.shape[-1]
        labels_segmentation = tools.squeeze_labels_to_single_dimension(labels_segmentation)
    if prior is not None:
        num_classes_prior = prior.shape[-1]
        prior = tools.squeeze_labels_to_single_dimension(prior)
    
    num_classes_predictions_edge = 0
    if predictions_edge is not None:
        num_classes_predictions_edge = predictions_edge.shape[-1]
    num_classes_predictions_segmentation = 0
    if predictions_segmentation is not None:
        num_classes_predictions_segmentation = predictions_segmentation.shape[-1]
    
    if num_exp:
        num_exp = min(num_exp, batch_size)
    else:
        num_exp = batch_size
    
    rows = 1 + (prior is not None) + (labels_edge is not None) + (
            labels_segmentation is not None) + num_classes_predictions_edge * (
                   predictions_edge is not None) + (num_classes_predictions_segmentation + 2) * (
                   predictions_segmentation is not None)
    cols = num_exp
    plt.figure(figsize=(8 * cols, 8 * rows))
    
    subplot_idx = 1
    for i in range(num_exp):
        plt.subplot(rows, cols, subplot_idx)
        plt.title("Images")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i, :, :, :]))
        plt.axis('off')
        subplot_idx += 1
    if prior is not None:
        for i in range(num_exp):
            plt.subplot(rows, num_exp, subplot_idx)
            plt.title("Prior Edge Maps")
            plt.imshow(prior[i, :, :, 0], cmap='gray', vmin=0, vmax=num_classes_prior)
            plt.axis('off')
            subplot_idx += 1
    if labels_edge is not None:
        for i in range(num_exp):
            plt.subplot(rows, num_exp, subplot_idx)
            plt.title("Ground Truth Edge")
            plt.imshow(labels_edge[i, :, :, 0], cmap='gray', vmin=0, vmax=num_classes_labels_edge)
            plt.axis('off')
            subplot_idx += 1
    if labels_segmentation is not None:
        for i in range(num_exp):
            plt.subplot(rows, num_exp, subplot_idx)
            plt.title("Ground Truth Segmentation")
            plt.imshow(labels_segmentation[i, :, :, 0], cmap='gray', vmin=0, vmax=num_classes_labels_segmentation)
            plt.axis('off')
            subplot_idx += 1
    if predictions_edge is not None:
        for j in range(num_classes_predictions_edge):
            for i in range(num_exp):
                plt.subplot(rows, num_exp, subplot_idx)
                plt.title("Estimation of class: {}".format(j + 1))
                plt.imshow(predictions_edge[i, :, :, j], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                subplot_idx += 1
    if predictions_segmentation is not None:
        for j in range(num_classes_predictions_segmentation):
            for i in range(num_exp):
                plt.subplot(rows, num_exp, subplot_idx)
                plt.title("Estimation of class: {}".format(j + 1))
                plt.imshow(predictions_segmentation[i, :, :, j], cmap='gray', vmin=-2, vmax=2)
                plt.axis('off')
                subplot_idx += 1
    
    if save:
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".svg", bbox_inches='tight')
    
    plt.draw()
    plt.show()


def plot_threshold_metrics_evaluation(model, ds, num_classes, classes_displayed_individually=False,
                                      accuracy_y_lim_min=0.95, padding=0, num_pixels_region_of_attraction=0,
                                      threshold_edge_width=0, save=False, path=None):
    step_width = 0.1
    threshold_array = np.arange(step_width, 1 - step_width, step_width)
    
    if classes_displayed_individually:
        num_classes_dimension = num_classes + 1
    else:
        num_classes_dimension = 1
    
    f1_score = np.zeros((num_classes_dimension, threshold_array.shape[0]))
    precision_score = np.zeros((num_classes_dimension, threshold_array.shape[0]))
    recall_score = np.zeros((num_classes_dimension, threshold_array.shape[0]))
    accuracy_score = np.zeros((num_classes_dimension, threshold_array.shape[0]))
    
    for i in range(threshold_array.shape[0]):
        # threshold_prediction = np.log(threshold_array[i]) - np.log(1 - threshold_array[i])
        threshold_prediction = threshold_array[i]
        
        model.compile(loss={'out_edge': []},
                      metrics={'out_edge': [metrics.BinaryAccuracyEdges(threshold_prediction=threshold_prediction,
                                                                        num_classes=num_classes,
                                                                        classes_individually=classes_displayed_individually,
                                                                        padding=padding,
                                                                        num_pixels_region_of_attraction=num_pixels_region_of_attraction),
                                            metrics.F1Edges(threshold_prediction=threshold_prediction,
                                                            num_classes=num_classes,
                                                            classes_individually=classes_displayed_individually,
                                                            padding=padding,
                                                            num_pixels_region_of_attraction=num_pixels_region_of_attraction,
                                                            threshold_edge_width=threshold_edge_width)]})

        evaluate = model.evaluate(ds, verbose=0, return_dict=True)
        accuracy_score[0, i] = evaluate["accuracy"]
        f1_score[0, i] = evaluate["f1_edges"]
        precision_score[0, i] = evaluate["precision_edges"]
        recall_score[0, i] = evaluate["recall_edges"]
        for j in range(1, num_classes_dimension):
            accuracy_score[j, i] = evaluate["accuracy_edges_" + str(j)]
            f1_score[j, i] = evaluate["f1_edges_" + str(j)]
            precision_score[j, i] = evaluate["precision_edges_" + str(j)]
            recall_score[j, i] = evaluate["recall_edges_" + str(j)]

    
    max_f1_score_idx = np.argmax(f1_score, axis=1)
    max_f1_score = np.amax(f1_score, axis=1)
    max_accuracy_score_idx = np.argmax(accuracy_score, axis=1)
    max_accuracy_score = np.amax(accuracy_score, axis=1)
    print("Max Accuracy Score = {:.3f} at {:.3f}".format(max_accuracy_score[0],
                                                         threshold_array[max_accuracy_score_idx[0]]))
    print("MF1 = {:.3f}".format(max_f1_score[0]))
    if classes_displayed_individually:
        for j in range(1, num_classes_dimension):
            print("MF1_{} = {:.3f}, ODS_{} = {:.3f}".format(j, max_f1_score[j], j,
                                                            threshold_array[max_f1_score_idx[j]]))
    
    # define figure structure
    max_plots_for_each_row = 3
    shape = (int((num_classes_dimension + 1) / max_plots_for_each_row) + 1, max_plots_for_each_row)
    fig = plt.figure(figsize=(5 * shape[1], 4 * shape[0]))
    title = "MF = {:.3f} ".format(max_f1_score[0])
    for j in range(1, num_classes_dimension):
        title = title + "MF_{} = {:.3f}, ODS_{} = {:.3f} ".format(j, max_f1_score[j], j,
                                                                  threshold_array[max_f1_score_idx[j]])
    
    fig.suptitle(title)
    accuracy_plot = plt.subplot2grid(shape=shape, loc=(0, 0))
    accuracy_plot.plot(threshold_array, accuracy_score[0, :], label='Avg')
    for j in range(1, num_classes_dimension):
        accuracy_plot.plot(threshold_array, accuracy_score[j, :], label='Class_{}'.format(j))
    accuracy_plot.set_xlabel("Threshold")
    accuracy_plot.set_ylabel("Accuracy")
    accuracy_plot.legend(loc='lower right')
    accuracy_plot.set_ylim([accuracy_y_lim_min, 1])
    accuracy_plot.set_xlim([0, 1])
    for j in range(num_classes_dimension):
        idx = j + 1
        row = int(idx / max_plots_for_each_row)
        plot = plt.subplot2grid(shape=shape, loc=(row, idx - row * max_plots_for_each_row))
        plot.plot(threshold_array, f1_score[j, :], label="F1")
        plot.plot(threshold_array, precision_score[j, :], label="Precision")
        plot.plot(threshold_array, recall_score[j, :], label="Recall")
        plot.legend(loc='lower right')
        plot.set_xlabel("Threshold")
        if j == 0:
            plot.set_title("Average")
        else:
            plot.set_title("Class " + str(j))
        plot.set_ylim([0, 1])
        plot.set_xlim([0, 1])
    
    if save:
        plt.savefig(path, bbox_inches='tight')
    
    plt.draw()
    
    ods = []
    
    for j in range(1, num_classes_dimension):
        ods.append(threshold_array[max_f1_score_idx[j]])
    return ods
