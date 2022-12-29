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
                plt.imshow(predictions_segmentation[i, :, :, j], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                subplot_idx += 1
        
        prediction_max = np.where(predictions_segmentation > 0.5, 1.0, 0.0)
        prediction_max = tools.squeeze_labels_to_single_dimension(prediction_max)
        prediction_max_gt = \
            prediction_max + tf.cast(
                tf.image.resize(labels_edge * 10, (prediction_max.shape[1], prediction_max.shape[2])), tf.int32)
        
        prior_resized = tf.where(tf.cast(prior, tf.float32) >= 1.0, 1.0, 0.0)
        prior_resized = tf.image.resize(prior_resized, (prediction_max.shape[1], prediction_max.shape[2]))
        prior_resized = tf.where(prior_resized > 0.7, 1.0, 0.0)
        prediction_max_prior = prediction_max + tf.cast(prior_resized, tf.int32)*20
        for i in range(num_exp):
            plt.subplot(rows, num_exp, subplot_idx)
            plt.title("with ground truth")
            plt.imshow(prediction_max_gt[i, :, :, 0], cmap='gray', vmin=0, vmax=8)
            plt.axis('off')
            subplot_idx += 1
        for i in range(num_exp):
            plt.subplot(rows, num_exp, subplot_idx)
            plt.title("with ground truth")
            plt.imshow(prediction_max_prior[i, :, :, 0], cmap='gray', vmin=0, vmax=8)
            plt.axis('off')
            subplot_idx += 1
    
    if save:
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".svg", bbox_inches='tight')
    
    plt.draw()
    plt.show()


def plot_threshold_metrics_evaluation(model, ds, num_classes, classes_displayed_individually=False,
                                      accuracy_y_lim_min=0.95, save=False,
                                      path=None, threshold_edge_width=0):
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
                                                                        classes_individually=classes_displayed_individually),
                                            metrics.F1Edges(threshold_prediction=threshold_prediction,
                                                            threshold_edge_width=threshold_edge_width,
                                                            num_classes=num_classes,
                                                            classes_individually=classes_displayed_individually)]})
        
        evaluate = model.evaluate(ds, verbose=0)
        for j in range(num_classes_dimension):
            accuracy_score[j, i] = evaluate[1 + j]
            f1_score[j, i] = evaluate[1 + num_classes_dimension + j]
            precision_score[j, i] = evaluate[1 + 2 * num_classes_dimension + j]
            recall_score[j, i] = evaluate[1 + 3 * num_classes_dimension + j]
    
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


def get_flow(flow, numpy=False):
    if numpy:
        x_flow = np.abs(flow[:, :, :, 0])
        y_flow = np.abs(flow[:, :, :, 1])
    else:
        x_flow = np.abs((flow[:, :, :, 0]).numpy())
        y_flow = np.abs((flow[:, :, :, 1]).numpy())
    return x_flow, y_flow


def plot_flow_field(images=None, prior=None, flow_ground_truth=None, flow_prediction=None, batch_size=0, num_exp=None,
                    num_classes=1):
    flow_max = 4.0
    
    if num_exp:
        num_exp = min(num_exp, batch_size)
    else:
        num_exp = batch_size
    
    # mask = np.zeros(shape=[x_flow.shape[0], x_flow.shape[1], x_flow.shape[2], 3], dtype=np.int32)
    # already_drawn = np.zeros(shape=x_flow.shape, dtype=np.int32)
    #
    # for batch in range(0, x_flow.shape[0]):
    #     for row in range(0, x_flow.shape[1]):
    #         for col in range(0, y_flow.shape[2]):
    #             if x_flow[batch, row, col] != 0 and already_drawn[batch, row, col] == 0:
    #                 start_point = np.array([col, row])
    #                 end_point = start_point + np.array([x_flow[batch, row, col], y_flow[batch, row, col]], np.int32)
    #                 color = (255, 255, 255)
    #                 thickness = 1
    #                 mask[batch, :, :, :] = cv2.arrowedLine(mask[batch, :, :, :], start_point, end_point, color=color,
    #                                                        thickness=thickness)
    #                 already_drawn[batch, max(row - pad, 0):min(row + pad, x_flow.shape[1] - 1),
    #                 max(col - pad, 0):min(col + pad, x_flow.shape[2] - 1)] = 1
    
    rows = 1 + (prior is not None) + 2 * (flow_ground_truth is not None) + 2 * (flow_prediction is not None)
    cols = num_exp
    plt.figure(figsize=(8 * cols, 8 * rows))
    for i in range(num_exp):
        plt.subplot(rows, cols, i + 1)
        plt.title("Images")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i, :, :, :]))
        plt.axis('off')
        if prior is not None:
            plt.subplot(rows, num_exp, num_exp + i + 1)
            plt.title("Prior Edge Maps")
            plt.imshow(tf.keras.preprocessing.image.array_to_img(prior[i, :, :, :]))
            plt.axis('off')
        if flow_ground_truth is not None:
            x_flow_gt, y_flow_gt = get_flow(flow_ground_truth)
            
            plt.subplot(rows, num_exp, (1 + (prior is not None)) * num_exp + i + 1)
            plt.title("Ground Truth Flow x Direction")
            plt.imshow(x_flow_gt[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
            plt.subplot(rows, num_exp, (1 + (prior is not None)) * num_exp + num_exp + i + 1)
            plt.title("Ground Truth Flow y Direction")
            plt.imshow(y_flow_gt[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
        if flow_prediction is not None:
            x_flow_pred, y_flow_pred = get_flow(flow_prediction, numpy=True)
            
            plt.subplot(rows, num_exp,
                        (1 + (prior is not None) + 2 * (flow_ground_truth is not None)) * num_exp + i + 1)
            plt.title("Flow x Direction")
            plt.imshow(x_flow_pred[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
            plt.subplot(rows, num_exp,
                        (1 + (prior is not None) + 2 * (flow_ground_truth is not None)) * num_exp + num_exp + i + 1)
            plt.title("Flow y Direction")
            plt.imshow(y_flow_pred[i, :, :], cmap='gray', vmin=0, vmax=flow_max)
            plt.axis('off')
