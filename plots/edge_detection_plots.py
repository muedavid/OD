import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from metrics import metrics


def plot_edges(images=None, labels=None, predictions=None, save=False, path=None, batch_size=0, num_exp=None,
               num_classes=0):
    # TODO: think about plotting output after sigmoid as values are more meaningfull and scale too.
    
    if num_exp:
        num_exp = min(num_exp, batch_size)
    else:
        num_exp = batch_size
    
    rows = 1 + (labels is not None) + num_classes * (predictions is not None)
    cols = num_exp
    plt.figure(figsize=(8 * cols, 8 * rows))
    for i in range(num_exp):
        plt.subplot(rows, cols, i + 1)
        plt.title("Images")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i, :, :, :]))
        plt.axis('off')
        if labels is not None:
            plt.subplot(rows, num_exp, num_exp + i + 1)
            plt.title("Ground Truth")
            plt.imshow(labels[i, :, :, 0], cmap='gray', vmin=0, vmax=num_classes)
            plt.axis('off')
        if predictions is not None:
            for j in range(num_classes):
                plt.subplot(rows, num_exp, (1 + (labels is not None) + j) * num_exp + i + 1)
                plt.title("Estimation of class: {}".format(j + 1))
                plt.imshow(predictions[i, :, :, j], cmap='gray', vmin=-5, vmax=5)
                plt.axis('off')
    
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
        threshold_prediction = np.log(threshold_array[i]) - np.log(1 - threshold_array[i])
        
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
        row = int(idx/max_plots_for_each_row)
        plot = plt.subplot2grid(shape=shape, loc=(row, idx - row*max_plots_for_each_row))
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
